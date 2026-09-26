from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-scale15m-q1-attention-modal-20260926-p"
ENGINEERING_SEED = 20_261_022
BATCH = 768
HEADS = 8
HEAD_DIM = 32
QUERY_LEN = 1
KEY_LENGTHS = (1, 8, 16, 32, 64)
WARMUP = 5
REPEATS = 20

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
)
app = modal.App("tam-rlt-systems-scale15m-q1-attention-20260926-p")


def _encode(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=32768,
    timeout=20 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_scale15m_q1_attention_p",
        "engineering_seed": ENGINEERING_SEED,
        "requires_kernel_profile_job_id": "rlt-systems-scale15m-kernel-profile-modal-20260926-o",
        "batch": BATCH,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "query_len": QUERY_LEN,
        "key_lengths": list(KEY_LENGTHS),
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"q1-attention preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("q1-attention profile requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(ENGINEERING_SEED)
    torch.cuda.manual_seed_all(ENGINEERING_SEED)
    dtype = torch.bfloat16

    backend_specs = [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("math", SDPBackend.MATH),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ]

    def run_once(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: Any, backward: bool) -> torch.Tensor:
        with sdpa_kernel(backends=[backend]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        if backward:
            out.float().square().mean().backward()
        return out

    def timed(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: Any, backward: bool
    ) -> dict[str, Any]:
        try:
            for _ in range(WARMUP):
                q.grad = None; k.grad = None; v.grad = None
                run_once(q, k, v, backend, backward)
            torch.cuda.synchronize(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPEATS):
                q.grad = None; k.grad = None; v.grad = None
                run_once(q, k, v, backend, backward)
            end.record()
            torch.cuda.synchronize(device)
            total_ms = float(start.elapsed_time(end))
            return {
                "supported": True,
                "total_ms": total_ms,
                "repeats": REPEATS,
                "mean_ms": total_ms / REPEATS,
            }
        except Exception as exc:
            torch.cuda.synchronize(device)
            return {
                "supported": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    results: dict[str, Any] = {}
    numerical: dict[str, Any] = {}

    for key_len in KEY_LENGTHS:
        gen = torch.Generator(device="cuda").manual_seed(ENGINEERING_SEED + key_len)
        q0 = torch.randn(BATCH, HEADS, QUERY_LEN, HEAD_DIM, device=device, dtype=dtype, generator=gen)
        k0 = torch.randn(BATCH, HEADS, key_len, HEAD_DIM, device=device, dtype=dtype, generator=gen)
        v0 = torch.randn(BATCH, HEADS, key_len, HEAD_DIM, device=device, dtype=dtype, generator=gen)

        per_backend: dict[str, Any] = {}
        for name, backend in backend_specs:
            q = q0.detach().clone().requires_grad_(True)
            k = k0.detach().clone().requires_grad_(True)
            v = v0.detach().clone().requires_grad_(True)
            per_backend[name] = {
                "forward": timed(q, k, v, backend, backward=False),
                "forward_backward": timed(q, k, v, backend, backward=True),
            }
        results[str(key_len)] = per_backend

        # Numerical comparison to Flash at representative lengths.
        if key_len in (32, 64):
            refs: dict[str, Any] = {}
            qf = q0.detach().clone().requires_grad_(True)
            kf = k0.detach().clone().requires_grad_(True)
            vf = v0.detach().clone().requires_grad_(True)
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                flash_out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=False)
            flash_loss = flash_out.float().square().mean()
            flash_loss.backward()
            refs["flash_loss"] = float(flash_loss.detach().cpu())

            for name, backend in backend_specs[1:]:
                qc = q0.detach().clone().requires_grad_(True)
                kc = k0.detach().clone().requires_grad_(True)
                vc = v0.detach().clone().requires_grad_(True)
                try:
                    with sdpa_kernel(backends=[backend]):
                        out = F.scaled_dot_product_attention(qc, kc, vc, is_causal=False)
                    loss = out.float().square().mean()
                    loss.backward()
                    refs[name] = {
                        "supported": True,
                        "output_max_abs_vs_flash": float((out - flash_out).abs().max().item()),
                        "loss_abs_vs_flash": abs(float(loss.detach().cpu()) - float(flash_loss.detach().cpu())),
                        "q_grad_max_abs_vs_flash": float((qc.grad - qf.grad).abs().max().item()),
                        "k_grad_max_abs_vs_flash": float((kc.grad - kf.grad).abs().max().item()),
                        "v_grad_max_abs_vs_flash": float((vc.grad - vf.grad).abs().max().item()),
                        "bit_identical_output": bool(torch.equal(out, flash_out)),
                    }
                except Exception as exc:
                    refs[name] = {
                        "supported": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            numerical[str(key_len)] = refs

    # Rank supported forward+backward candidates at length 32, the decoder SWA steady-state case.
    steady = results["32"]
    ranked = []
    flash_ms = steady["flash"]["forward_backward"].get("mean_ms")
    for name, row in steady.items():
        fb = row["forward_backward"]
        if fb.get("supported"):
            ms = float(fb["mean_ms"])
            ranked.append({
                "backend": name,
                "mean_forward_backward_ms": ms,
                "speedup_vs_flash": None if flash_ms is None else float(flash_ms) / ms,
            })
    ranked.sort(key=lambda x: x["mean_forward_backward_ms"])

    return _encode({
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": "PASS_SCALE15M_Q1_ATTENTION_BACKEND_PROFILE",
        "engineering_only": True,
        "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "engineering_seed": ENGINEERING_SEED,
        "started_unix": started,
        "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "shape": {
            "batch": BATCH,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "query_len": QUERY_LEN,
            "key_lengths": list(KEY_LENGTHS),
        },
        "timings": results,
        "numerical_vs_flash": numerical,
        "steady_state_len32_ranking": ranked,
        "interpretation_ceiling": (
            "Engineering microbenchmark only. It identifies whether PyTorch SDPA backends are "
            "better suited than FlashAttention for RLT's repeated q_len=1 decoder attention. "
            "Numerical comparisons are reported explicitly; no scientific quality claim is implied."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote q1-attention result must be base64 text")
    print("RLT_SYSTEMS_SCALE15M_Q1_ATTENTION_P_RESULT_B64=" + result)
