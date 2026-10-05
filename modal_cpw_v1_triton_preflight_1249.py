from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics
import time

import modal

from tam_research.cpw_v1.protocol import (
    EXPECTED_CPW_V1_PARAMETERS,
    GRAD_ACCUM_STEPS,
    MICRO_BATCH_SIZE,
    SEQ_LEN,
    TRANSFORMER_PARAMETERS,
)

ISSUE = 1249
SEED = 1_249_001
TOKENS = 1_000_000
RESULT_ROOT = "/vol/cpw-v1/triton-scan-preflight-v1"
APP_NAME = "tam-research-cpw-v1-triton-scan-1249"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "datasets>=4.0,<5",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "numpy>=2.0,<3",
        "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/vol": volume},
)
def ensure_data() -> dict:
    from tam_research.data import prepare_fineweb
    result = prepare_fineweb(
        "/vol/data/fineweb-edu-gpt2",
        train_tokens=25_000_000,
        val_tokens=2_000_000,
    )
    volume.commit()
    return result


def _summary(r: dict) -> dict:
    return {
        "architecture": r["architecture"],
        "seed": int(r["seed"]),
        "parameters": int(r["parameters"]),
        "nll": float(r["final_eval"]["nll"]),
        "training_tps": float(r["training_tokens_per_second"]),
        "total_compute_seconds": float(r["total_compute_seconds"]),
        "peak_vram_gib": float(r["peak_vram_gb"]),
    }


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def run_preflight(source_sha: str) -> dict:
    import torch

    from architectures.cortex_s.affine_scan_triton_candidate import (
        affine_scan_triton_candidate,
        triton_available,
    )
    from tam_research.cpw_v1.train import train_cpw_v1_candidate
    from tam_research.cpw_v1_fast.model import fast_parameter_count
    from tam_research.cpw_v1_fast.train import train_cpw_v1_fast
    from tam_research.models import diagonal_affine_scan
    from tam_research.train import train_language_model

    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise RuntimeError(f"invalid source SHA: {source_sha!r}")
    if not triton_available():
        raise RuntimeError("Triton unavailable on H100 image")
    if fast_parameter_count() != EXPECTED_CPW_V1_PARAMETERS:
        raise RuntimeError("fast backend parameter count drift")

    root = Path(RESULT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    attempt_path = root / "ATTEMPT.json"
    result_path = root / "RESULT.json"
    failure_path = root / "FAILURE.json"
    if attempt_path.exists() or result_path.exists() or failure_path.exists():
        raise RuntimeError("CPW-v1 Triton preflight namespace already consumed")

    attempt = {
        "issue": ISSUE,
        "source_sha": source_sha,
        "seed": SEED,
        "tokens": TOKENS,
        "shape": [64, 512, 64],
        "retry_authorized": False,
        "scientific_claim_allowed": False,
        "breakthrough_claim_allowed": False,
        "started_unix": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    volume.commit()
    print("CPW_V1_TRITON_ATTEMPT=" + json.dumps(attempt, sort_keys=True), flush=True)

    try:
        device = torch.device("cuda")
        torch.manual_seed(SEED)
        a0 = torch.sigmoid(
            torch.randn(64, 512, 64, device=device, dtype=torch.bfloat16)
        )
        b0 = (
            0.05
            * torch.randn(64, 512, 64, device=device, dtype=torch.bfloat16)
        )

        ar = a0.detach().clone().requires_grad_(True)
        br = b0.detach().clone().requires_grad_(True)
        yr = diagonal_affine_scan(ar, br)
        lr = yr.float().square().mean()
        gar, gbr = torch.autograd.grad(lr, (ar, br))

        af = a0.detach().clone().requires_grad_(True)
        bf = b0.detach().clone().requires_grad_(True)
        yf = affine_scan_triton_candidate(af, bf, None)
        lf = yf.float().square().mean()
        gaf, gbf = torch.autograd.grad(lf, (af, bf))

        correctness = {
            "output_max_abs": float((yr.float() - yf.float()).abs().max()),
            "output_mean_abs": float((yr.float() - yf.float()).abs().mean()),
            "grad_a_max_abs": float((gar.float() - gaf.float()).abs().max()),
            "grad_b_max_abs": float((gbr.float() - gbf.float()).abs().max()),
            "finite": bool(
                torch.isfinite(yf).all()
                and torch.isfinite(gaf).all()
                and torch.isfinite(gbf).all()
            ),
        }
        correctness["pass"] = bool(
            correctness["finite"]
            and correctness["output_max_abs"] <= 0.015625
            and correctness["grad_a_max_abs"] <= 0.004
            and correctness["grad_b_max_abs"] <= 0.008
        )
        print(
            "CPW_V1_TRITON_CORRECTNESS="
            + json.dumps(correctness, sort_keys=True),
            flush=True,
        )
        if not correctness["pass"]:
            raise RuntimeError(f"Triton correctness gate failed: {correctness}")

        def bench(fn, warmup: int = 5, iters: int = 25) -> float:
            a = a0.detach().clone().requires_grad_(True)
            b = b0.detach().clone().requires_grad_(True)
            for _ in range(warmup):
                a.grad = None
                b.grad = None
                y = fn(a, b)
                y.float().square().mean().backward()
            torch.cuda.synchronize()
            samples = []
            for _ in range(iters):
                a.grad = None
                b.grad = None
                started = time.perf_counter()
                y = fn(a, b)
                y.float().square().mean().backward()
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - started)
            return statistics.median(samples)

        production_scan_s = bench(lambda a, b: diagonal_affine_scan(a, b))
        triton_scan_s = bench(
            lambda a, b: affine_scan_triton_candidate(a, b, None)
        )
        scan_speedup = production_scan_s / max(triton_scan_s, 1e-12)
        scan = {
            "production_median_s": production_scan_s,
            "triton_median_s": triton_scan_s,
            "speedup": scan_speedup,
        }
        print("CPW_V1_TRITON_SCAN_TIMING=" + json.dumps(scan, sort_keys=True), flush=True)

        train_root = str(root / "training")
        transformer = train_language_model(
            architecture="transformer",
            seed=SEED,
            data_dir="/vol/data/fineweb-edu-gpt2",
            run_root=str(Path(train_root) / "transformer"),
            token_budget=TOKENS,
            seq_len=SEQ_LEN,
            micro_batch_size=MICRO_BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            eval_every_tokens=TOKENS,
            checkpoint_every_tokens=TOKENS,
            resume=False,
            compile_model=False,
        )
        volume.commit()

        production = train_cpw_v1_candidate(
            architecture="cpwv1",
            seed=SEED,
            data_dir="/vol/data/fineweb-edu-gpt2",
            run_root=str(Path(train_root) / "production"),
            token_budget=TOKENS,
            seq_len=SEQ_LEN,
            micro_batch_size=MICRO_BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
        )
        volume.commit()

        fast = train_cpw_v1_fast(
            seed=SEED,
            data_dir="/vol/data/fineweb-edu-gpt2",
            run_root=str(Path(train_root) / "fast"),
            token_budget=TOKENS,
            seq_len=SEQ_LEN,
            micro_batch_size=MICRO_BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
        )
        volume.commit()

        t = _summary(transformer)
        p = _summary(production)
        f = _summary(fast)
        training = {
            "transformer": t,
            "production_cpw": p,
            "fast_cpw": f,
            "fast_vs_production_tps": f["training_tps"] / max(p["training_tps"], 1e-12),
            "fast_vs_transformer_tps": f["training_tps"] / max(t["training_tps"], 1e-12),
            "fast_minus_production_nll": f["nll"] - p["nll"],
        }

        promising = bool(
            correctness["pass"]
            and scan_speedup > 1.0
            and training["fast_vs_production_tps"] > 1.0
            and training["fast_vs_transformer_tps"] >= 0.85
            and abs(training["fast_minus_production_nll"]) <= 0.02
            and int(f["parameters"]) == EXPECTED_CPW_V1_PARAMETERS
            and int(t["parameters"]) == TRANSFORMER_PARAMETERS
        )
        final = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "classification": (
                "CPW_V1_TRITON_PROMISING"
                if promising
                else "CPW_V1_TRITON_INSUFFICIENT"
            ),
            "correctness": correctness,
            "scan": scan,
            "training": training,
            "scientific_claim_allowed": False,
            "breakthrough_claim_allowed": False,
            "completed_unix": time.time(),
        }
        result_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V1_TRITON_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
        return final
    except BaseException as exc:
        failure = {
            "issue": ISSUE,
            "source_sha": source_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scientific_interpretation": False,
            "retry_authorized": False,
            "failed_unix": time.time(),
        }
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        volume.commit()
        print("CPW_V1_TRITON_FAILURE=" + json.dumps(failure, sort_keys=True), flush=True)
        raise


@app.local_entrypoint()
def main(source_sha: str) -> None:
    ensure_data.remote()
    result = run_preflight.remote(source_sha)
    print("CPW_V1_TRITON_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
