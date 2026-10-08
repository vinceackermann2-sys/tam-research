from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-gated-scan-optimizer-qualitysec-modal-20261008-al"
ENGINEERING_SEED = 20_261_055
BATCH_SEED = 20_271_055
EVAL_SEED = 20_291_055
COMPILE_PROBE_SEED = 20_301_055
SEQ_LEN = 64
BATCH_SIZE = 64
TIME_BUDGET_SECONDS = 60.0
EVAL_BATCH_SIZE = 64
EVAL_BATCHES = 32
EXPECTED_PARAMETERS = 15_129_344
WEIGHT_DECAY = 0.1
WARMUP_TIME_RATIO = 0.02
EXPECTED_TRAIN_SHA256 = "0f9ea99680f532f990ca64fabbd02b46bba2c4f8077f868545a03f5e6907c907"
EXPECTED_VAL_SHA256 = "e6b3fb209a80a2cebe22924c3c8af26b30fc4afd322a1a49c71af8b968057db2"

CANDIDATES = (
    {"name": "cosine_lr3e-4", "schedule": "cosine", "learning_rate": 3e-4},
    {"name": "cosine_lr1e-3", "schedule": "cosine", "learning_rate": 1e-3},
    {"name": "cosine_lr3e-3", "schedule": "cosine", "learning_rate": 3e-3},
    {"name": "constant_lr3e-4", "schedule": "constant", "learning_rate": 3e-4},
    {"name": "constant_lr1e-3", "schedule": "constant", "learning_rate": 1e-3},
    {"name": "constant_lr3e-3", "schedule": "constant", "learning_rate": 3e-3},
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11", "numpy>=2.0,<3", "datasets>=4.0,<5",
        "transformers>=4.55,<5", "tokenizers>=0.21,<1", "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-gated-scan-optimizer-qualitysec-20261008-al")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _candidate_manifest() -> list[dict[str, Any]]:
    return [
        {
            "name": str(c["name"]),
            "schedule": str(c["schedule"]),
            "learning_rate": float(c["learning_rate"]),
        }
        for c in CANDIDATES
    ]


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=95 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, parameter_count
    from experiments.rlt.model_gated_scan import GatedScanLightStateRLT
    from experiments.rlt.train_lightstate_scale15m_pair_4m_a import CompilableLightStateRLT
    from tam_research.data import TokenBin, prepare_fineweb

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_gated_scan_optimizer_qualitysec_al",
        "engineering_seed": ENGINEERING_SEED,
        "batch_seed": BATCH_SEED,
        "eval_seed": EVAL_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "requires_ak_realdata_job_id": "rlt-systems-gated-scan-realdata-modal-20261008-ak",
        "requires_aj_throughput_job_id": "rlt-systems-gated-scan-throughput-modal-20261008-aj",
        "requires_cpu_tests_run_id": 37750663298,
        "batch_size": BATCH_SIZE,
        "time_budget_seconds_each": TIME_BUDGET_SECONDS,
        "seq_len": SEQ_LEN,
        "train_tokens": 6_000_000,
        "val_tokens": 500_000,
        "training_data_seed": 20_260_924,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "candidates": _candidate_manifest(),
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"gated-scan optimizer quality/sec preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("gated-scan optimizer quality/sec calibration requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )

    data_dir = Path("/tmp/rlt-gated-scan-optimizer-qualitysec-al")
    meta = prepare_fineweb(
        str(data_dir),
        train_tokens=job["train_tokens"],
        val_tokens=job["val_tokens"],
        seed=job["training_data_seed"],
    )
    train_sha = _sha256(data_dir / "train.bin")
    val_sha = _sha256(data_dir / "val.bin")
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(f"train hash mismatch: {train_sha}")
    if val_sha != EXPECTED_VAL_SHA256:
        raise RuntimeError(f"val hash mismatch: {val_sha}")
    train_data = TokenBin(str(data_dir / "train.bin"))
    val_data = TokenBin(str(data_dir / "val.bin"))

    def seed_all(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def new_base() -> GatedScanLightStateRLT:
        seed_all(ENGINEERING_SEED)
        model = GatedScanLightStateRLT(cfg).to(device)
        if parameter_count(model) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"gated-scan parameter drift: {parameter_count(model)}")
        return model

    # Strict wrapper equivalence before any calibration.
    reference = new_base()
    candidate_base = new_base()
    candidate_base.load_state_dict(reference.state_dict())
    candidate = CompilableLightStateRLT(candidate_base)
    gen0 = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED)
    x0 = torch.randint(0, cfg.vocab_size, (2, SEQ_LEN), generator=gen0, dtype=torch.long).to(device)
    y0 = torch.randint(0, cfg.vocab_size, (2, SEQ_LEN), generator=gen0, dtype=torch.long).to(device)
    reference.zero_grad(set_to_none=True)
    candidate_base.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ref_logits = reference(x0)
        cand_logits = candidate(x0)
        ref_loss = F.cross_entropy(ref_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
        cand_loss = F.cross_entropy(cand_logits.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
    ref_loss.backward()
    cand_loss.backward()
    torch.cuda.synchronize(device)
    logits_abs = float((ref_logits - cand_logits).abs().max().item())
    loss_abs = float((ref_loss - cand_loss).abs().item())
    grad_abs = 0.0
    for rp, cp in zip(reference.parameters(), candidate_base.parameters()):
        if rp.grad is None or cp.grad is None:
            if rp.grad is not cp.grad:
                grad_abs = float("inf")
                break
            continue
        grad_abs = max(grad_abs, float((rp.grad - cp.grad).abs().max().item()))
    wrapper_exact = logits_abs == 0.0 and loss_abs == 0.0 and grad_abs == 0.0
    del reference, candidate_base, candidate, ref_logits, cand_logits
    torch.cuda.empty_cache()
    if not wrapper_exact:
        raise RuntimeError("CompilableLightStateRLT wrapper lost exact semantic equivalence")

    def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def evaluate_eager(model: nn.Module) -> dict[str, float]:
        model.eval()
        gen = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
        losses: list[float] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(EVAL_BATCHES):
                x, y = val_data.batch(EVAL_BATCH_SIZE, SEQ_LEN, gen, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
                losses.append(float(loss.detach().cpu()))
        torch.cuda.synchronize(device)
        nll = sum(losses) / len(losses)
        return {
            "nll": nll,
            "perplexity": math.exp(min(nll, 20.0)),
            "eval_tokens": EVAL_BATCHES * EVAL_BATCH_SIZE * SEQ_LEN,
            "seconds": time.perf_counter() - t0,
            "execution": "eager_no_grad",
        }

    def current_lr(spec: dict[str, Any], elapsed: float) -> float:
        lr = float(spec["learning_rate"])
        schedule = str(spec["schedule"])
        if schedule == "constant":
            return lr
        if schedule != "cosine":
            raise RuntimeError(f"unknown schedule: {schedule}")
        frac = min(max(elapsed / TIME_BUDGET_SECONDS, 0.0), 1.0)
        if frac < WARMUP_TIME_RATIO:
            scale = max(frac / WARMUP_TIME_RATIO, 1e-3)
        else:
            progress = (frac - WARMUP_TIME_RATIO) / (1.0 - WARMUP_TIME_RATIO)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr * scale

    base = new_base()
    initial_state = snapshot(base)
    initial_eval = evaluate_eager(base)
    module = CompilableLightStateRLT(base)
    compiled = torch.compile(module, fullgraph=True, dynamic=False, mode="default")

    # Compile one fixed-batch training graph before the timed candidates.
    compile_gen = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED + BATCH_SIZE)
    x_compile, y_compile = train_data.batch(BATCH_SIZE, SEQ_LEN, compile_gen, device)
    base.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    compile_t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        compile_logits = compiled(x_compile)
        compile_loss = F.cross_entropy(
            compile_logits.float().reshape(-1, cfg.vocab_size), y_compile.reshape(-1)
        )
    compile_loss.backward()
    torch.cuda.synchronize(device)
    compile_training_step_seconds = time.perf_counter() - compile_t0
    compile_training_step_loss = float(compile_loss.detach().cpu())
    base.load_state_dict(initial_state, strict=True)
    base.zero_grad(set_to_none=True)

    rows: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        base.load_state_dict(initial_state, strict=True)
        base.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(
            base.parameters(),
            lr=float(spec["learning_rate"]),
            betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY,
            fused=True,
        )
        batch_gen = torch.Generator(device="cpu").manual_seed(BATCH_SEED)
        base.train()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        steps = 0
        tokens = 0
        last_loss = float("nan")
        last_lr = float(spec["learning_rate"])
        training_nonfinite = False

        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= TIME_BUDGET_SECONDS and steps > 0:
                break
            optimizer.zero_grad(set_to_none=True)
            x, y = train_data.batch(BATCH_SIZE, SEQ_LEN, batch_gen, device)
            last_lr = current_lr(spec, elapsed)
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled(x)
                loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
            last_loss = float(loss.detach().cpu())
            if not math.isfinite(last_loss):
                training_nonfinite = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optimizer.step()
            steps += 1
            tokens += BATCH_SIZE * SEQ_LEN

        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        final_eval = evaluate_eager(base)
        finite_final_nll = math.isfinite(float(final_eval["nll"]))
        rows.append({
            "candidate": {
                "name": str(spec["name"]),
                "schedule": str(spec["schedule"]),
                "learning_rate": float(spec["learning_rate"]),
            },
            "batch_size": BATCH_SIZE,
            "parameters": parameter_count(base),
            "steps": steps,
            "tokens_seen": tokens,
            "actual_train_seconds": train_seconds,
            "tokens_per_second": tokens / max(train_seconds, 1e-9),
            "optimizer_steps_per_second": steps / max(train_seconds, 1e-9),
            "last_train_loss": last_loss,
            "last_lr": last_lr,
            "peak_vram_gb": peak,
            "training_nonfinite": training_nonfinite,
            "final_eval": final_eval,
            "finite_final_nll": finite_final_nll,
            "eligible_for_selection": finite_final_nll and not training_nonfinite,
        })
        del optimizer
        torch.cuda.empty_cache()

    eligible = [r for r in rows if r["eligible_for_selection"]]
    if not eligible:
        raise RuntimeError("all gated-scan optimizer candidates were ineligible")
    best = min(eligible, key=lambda r: float(r["final_eval"]["nll"]))

    return _encode({
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": "PASS_SCALE15M_GATED_SCAN_OPTIMIZER_QUALITYSEC_CALIBRATION",
        "engineering_only": True,
        "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "engineering_seed": ENGINEERING_SEED,
        "batch_seed": BATCH_SEED,
        "eval_seed": EVAL_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "started_unix": started,
        "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "data": {
            "metadata": meta,
            "train_sha256": train_sha,
            "val_sha256": val_sha,
        },
        "wrapper_semantic_equivalence": {
            "passed": wrapper_exact,
            "logits_max_abs": logits_abs,
            "loss_abs": loss_abs,
            "grad_max_abs": grad_abs,
        },
        "protocol": {
            "batch_size": BATCH_SIZE,
            "time_budget_seconds_each_candidate": TIME_BUDGET_SECONDS,
            "seq_len": SEQ_LEN,
            "candidates": _candidate_manifest(),
            "weight_decay": WEIGHT_DECAY,
            "warmup_time_ratio_for_cosine": WARMUP_TIME_RATIO,
            "compile_once_fixed_batch_before_candidates": True,
            "compile_time_excluded": True,
            "compile_training_step_seconds": compile_training_step_seconds,
            "compile_training_step_loss": compile_training_step_loss,
            "reset_identical_initial_weights_before_each_candidate": True,
            "same_batch_stream_seed": True,
            "same_eval_stream_seed": True,
            "validation_execution": "eager_no_grad",
        },
        "initial_eval": initial_eval,
        "results": rows,
        "best_by_final_nll": {
            "candidate": best["candidate"],
            "final_nll": best["final_eval"]["nll"],
            "steps": best["steps"],
            "tokens_seen": best["tokens_seen"],
            "tokens_per_second": best["tokens_per_second"],
            "optimizer_steps_per_second": best["optimizer_steps_per_second"],
        },
        "interpretation_ceiling": (
            "Engineering-only light-state RLT optimizer/schedule calibration at the independently selected "
            "batch size 64. It freezes the light-state setting for a later fresh-seed equal-GPU-time "
            "comparison and is not architecture-level scientific evidence."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote gated-scan optimizer quality/sec result must be base64 text")
    print("RLT_SYSTEMS_GATED_SCAN_OPTIMIZER_QUALITYSEC_AL_RESULT_B64=" + result)
