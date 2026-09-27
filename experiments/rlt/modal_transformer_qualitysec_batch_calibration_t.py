from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-scale15m-transformer-qualitysec-batch-modal-20260927-t"
ENGINEERING_SEED = 20_261_026
BATCH_SEED = 20_271_026
EVAL_SEED = 20_291_026
SEQ_LEN = 64
BATCH_SIZES = (64, 128, 256, 512, 768)
TIME_BUDGET_SECONDS = 60.0
EVAL_BATCH_SIZE = 64
EVAL_BATCHES = 32
EXPECTED_PARAMETERS = 15_129_344
EXPECTED_TRAIN_SHA256 = "0f9ea99680f532f990ca64fabbd02b46bba2c4f8077f868545a03f5e6907c907"
EXPECTED_VAL_SHA256 = "e6b3fb209a80a2cebe22924c3c8af26b30fc4afd322a1a49c71af8b968057db2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11", "numpy>=2.0,<3", "datasets>=4.0,<5",
        "transformers>=4.55,<5", "tokenizers>=0.21,<1", "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-scale15m-transformer-qualitysec-batch-20260927-t")


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


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=45 * 60,
    retries=0,
    max_containers=1,
)
def run_profile_remote(job_json: str) -> str:
    import torch
    import torch.nn.functional as F

    from tam_research.data import TokenBin, prepare_fineweb
    from tam_research.models import ModelConfig, ResearchLM, parameter_count

    started = time.time()
    job = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_scale15m_transformer_qualitysec_batch_t",
        "engineering_seed": ENGINEERING_SEED,
        "batch_seed": BATCH_SEED,
        "eval_seed": EVAL_SEED,
        "requires_compute_match_job_id": "rlt-compute-matched-scale15m-60s-modal-20260926-a",
        "batch_sizes": list(BATCH_SIZES),
        "time_budget_seconds_each": TIME_BUDGET_SECONDS,
        "seq_len": SEQ_LEN,
        "train_tokens": 6_000_000,
        "val_tokens": 500_000,
        "training_data_seed": 20_260_924,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if job != expected:
        raise RuntimeError(f"transformer quality/sec batch preregistration mismatch: {job}")
    if not torch.cuda.is_available():
        raise RuntimeError("transformer quality/sec calibration requires CUDA")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    cfg = ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938, architecture="transformer",
    )

    data_dir = Path("/tmp/transformer-qualitysec-batch-t")
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

    def new_model() -> ResearchLM:
        seed_all(ENGINEERING_SEED)
        m = ResearchLM(cfg).to(device)
        if parameter_count(m) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Transformer parameter drift: {parameter_count(m)}")
        return m

    def lr_at(elapsed: float) -> float:
        frac = min(max(elapsed / TIME_BUDGET_SECONDS, 0.0), 1.0)
        warmup = 0.02
        if frac < warmup:
            scale = max(frac / warmup, 1e-3)
        else:
            progress = (frac - warmup) / (1.0 - warmup)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 3e-4 * scale

    def evaluate_eager(model: ResearchLM) -> dict[str, float]:
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

    def calibrate(batch_size: int) -> dict[str, Any]:
        base = new_model()
        initial_eval = evaluate_eager(base)
        compiled = torch.compile(base, fullgraph=True, dynamic=False, mode="default")
        opt = torch.optim.AdamW(
            base.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
        )
        batch_gen = torch.Generator(device="cpu").manual_seed(BATCH_SEED)

        # Compile forward/backward without updating weights.
        x, y = train_data.batch(batch_size, SEQ_LEN, batch_gen, device)
        base.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        ct0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = compiled(x)
            loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        torch.cuda.synchronize(device)
        compile_step_seconds = time.perf_counter() - ct0
        base.zero_grad(set_to_none=True)

        batch_gen = torch.Generator(device="cpu").manual_seed(BATCH_SEED)
        base.train()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        steps = 0
        tokens = 0
        last_loss = float("nan")
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= TIME_BUDGET_SECONDS and steps > 0:
                break
            opt.zero_grad(set_to_none=True)
            x, y = train_data.batch(batch_size, SEQ_LEN, batch_gen, device)
            lr = lr_at(elapsed)
            for group in opt.param_groups:
                group["lr"] = lr
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled(x)
                loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            opt.step()
            steps += 1
            tokens += batch_size * SEQ_LEN
            last_loss = float(loss.detach().cpu())

        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        final_eval = evaluate_eager(base)

        out = {
            "batch_size": batch_size,
            "parameters": parameter_count(base),
            "initial_eval": initial_eval,
            "final_eval": final_eval,
            "steps": steps,
            "tokens_seen": tokens,
            "actual_train_seconds": train_seconds,
            "tokens_per_second": tokens / max(train_seconds, 1e-9),
            "optimizer_steps_per_second": steps / max(train_seconds, 1e-9),
            "last_train_loss": last_loss,
            "peak_vram_gb": peak,
            "compile_training_step_seconds": compile_step_seconds,
        }
        del compiled, opt, base
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return out

    rows = [calibrate(b) for b in BATCH_SIZES]
    best = min(rows, key=lambda x: float(x["final_eval"]["nll"]))
    return _encode({
        "schema": 1,
        "job_id": JOB_ID,
        "status": "complete",
        "classification": "PASS_SCALE15M_TRANSFORMER_QUALITYSEC_BATCH_CALIBRATION",
        "engineering_only": True,
        "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "engineering_seed": ENGINEERING_SEED,
        "batch_seed": BATCH_SEED,
        "eval_seed": EVAL_SEED,
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
        "protocol": {
            "batch_sizes": list(BATCH_SIZES),
            "time_budget_seconds_each": TIME_BUDGET_SECONDS,
            "seq_len": SEQ_LEN,
            "learning_rate": 3e-4,
            "warmup_ratio_by_elapsed_time": 0.02,
            "schedule": "cosine_by_elapsed_post_compile_training_time",
            "compile_time_excluded": True,
            "validation_execution": "eager_no_grad_to_avoid_per_batch_compile_graphs",
            "same_initialization_seed": True,
            "same_batch_stream_seed": True,
            "same_eval_stream_seed": True,
        },
        "results": rows,
        "best_by_final_nll": {
            "batch_size": best["batch_size"],
            "final_nll": best["final_eval"]["nll"],
            "tokens_seen": best["tokens_seen"],
            "steps": best["steps"],
            "tokens_per_second": best["tokens_per_second"],
            "optimizer_steps_per_second": best["optimizer_steps_per_second"],
        },
        "interpretation_ceiling": (
            "Engineering-only Transformer hyperparameter calibration. It selects batch size for "
            "validation quality per fixed post-compile GPU training time at exact 15.1M parameters. "
            "It is not architecture-level scientific evidence."
        ),
    })


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_profile_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("remote Transformer quality/sec batch calibration result must be base64 text")
    print("RLT_SYSTEMS_SCALE15M_TRANSFORMER_QUALITYSEC_BATCH_T_RESULT_B64=" + result)
