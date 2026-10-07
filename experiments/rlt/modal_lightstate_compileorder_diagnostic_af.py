from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-lightstate-compileorder-modal-20261007-af"
ENGINEERING_SEED = 20_261_048
BATCH_SEED = 20_271_048
EVAL_SEED = 20_291_048
COMPILE_PROBE_SEED = 20_301_048
SEQ_LEN = 64
BATCH_SIZE = 64
TIME_BUDGET_SECONDS = 20.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
EVAL_BATCH_SIZE = 64
EVAL_BATCHES = 32
EXPECTED_PARAMETERS = 15_129_344
EXPECTED_TRAIN_SHA256 = "0f9ea99680f532f990ca64fabbd02b46bba2c4f8077f868545a03f5e6907c907"
EXPECTED_VAL_SHA256 = "e6b3fb209a80a2cebe22924c3c8af26b30fc4afd322a1a49c71af8b968057db2"
VARIANTS = ("train_compile_first", "compiled_eval_first")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11", "numpy>=2.0,<3", "datasets>=4.0,<5",
        "transformers>=4.55,<5", "tokenizers>=0.21,<1", "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-ls-compileorder-20261007-af")


def _encode(value: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def _decode(value: str) -> dict[str, Any]:
    obj = json.loads(base64.urlsafe_b64decode(value.encode()))
    if not isinstance(obj, dict):
        raise TypeError("decoded remote diagnostic payload must be a dict")
    return obj


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_variant_impl(job_json: str, variant: str) -> str:
    import torch
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, parameter_count
    from experiments.rlt.model_light_state import LightStateRecurrentTransformer
    from experiments.rlt.train_lightstate_scale15m_pair_4m_a import CompilableLightStateRLT
    from tam_research.data import TokenBin, prepare_fineweb
    from tam_research.train import seed_all

    started = time.time()
    runtime: dict[str, Any] | None = None
    try:
        if variant not in VARIANTS:
            raise ValueError(f"unknown compile-order variant: {variant}")
        job = json.loads(job_json)
        expected = {
            "schema": 1,
            "job_id": JOB_ID,
            "task": "systems_lightstate_compileorder_af",
            "engineering_seed": ENGINEERING_SEED,
            "batch_seed": BATCH_SEED,
            "eval_seed": EVAL_SEED,
            "compile_probe_seed": COMPILE_PROBE_SEED,
            "training_data_seed": 20_260_924,
            "requires_optimizer_calibration_job_id": "rlt-systems-lightstate-optimizer-qualitysec-modal-20261007-ae",
            "requires_compute_match_job_id": "rlt-lightstate-compute-matched-optimizer-tuned-scale15m-60s-modal-20261007-b",
            "variants": list(VARIANTS),
            "time_budget_seconds_each": TIME_BUDGET_SECONDS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "schedule": "constant",
            "seq_len": SEQ_LEN,
            "train_tokens": 6_000_000,
            "val_tokens": 500_000,
            "expected_parameters": EXPECTED_PARAMETERS,
            "automatic_retry": False,
        }
        if job != expected:
            raise RuntimeError(f"compile-order diagnostic preregistration mismatch: {job}")
        if not torch.cuda.is_available():
            raise RuntimeError("compile-order diagnostic requires CUDA")
        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")

        data_dir = Path(f"/tmp/rlt-ls-compileorder-af-{variant}")
        meta = prepare_fineweb(
            str(data_dir),
            train_tokens=job["train_tokens"],
            val_tokens=job["val_tokens"],
            seed=job["training_data_seed"],
        )
        train_sha = _sha256(data_dir / "train.bin")
        val_sha = _sha256(data_dir / "val.bin")
        if train_sha != EXPECTED_TRAIN_SHA256 or val_sha != EXPECTED_VAL_SHA256:
            raise RuntimeError(f"pinned data hash mismatch: train={train_sha} val={val_sha}")
        train_data = TokenBin(str(data_dir / "train.bin"))
        val_data = TokenBin(str(data_dir / "val.bin"))

        cfg = RLTConfig(
            vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
            max_seq_len=128, ff_mult=4, swa_window=32,
        )
        seed_all(ENGINEERING_SEED)
        base = LightStateRecurrentTransformer(cfg).to(device)
        if parameter_count(base) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"light-state parameter drift: {parameter_count(base)}")
        compiled = torch.compile(
            CompilableLightStateRLT(base), fullgraph=True, dynamic=False, mode="default"
        )

        def eager_eval() -> dict[str, float]:
            base.eval()
            gen = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
            losses: list[float] = []
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    x, y = val_data.batch(EVAL_BATCH_SIZE, SEQ_LEN, gen, device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = base(x)
                        loss = F.cross_entropy(
                            logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1)
                        )
                    losses.append(float(loss.detach().cpu()))
            torch.cuda.synchronize(device)
            nll = sum(losses) / len(losses)
            return {"nll": nll, "seconds": time.perf_counter() - t0}

        def compiled_eval() -> dict[str, float]:
            base.eval()
            gen = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
            losses: list[float] = []
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    x, y = val_data.batch(EVAL_BATCH_SIZE, SEQ_LEN, gen, device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = compiled(x)
                        loss = F.cross_entropy(
                            logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1)
                        )
                    losses.append(float(loss.detach().cpu()))
            torch.cuda.synchronize(device)
            nll = sum(losses) / len(losses)
            return {"nll": nll, "seconds": time.perf_counter() - t0}

        if variant == "train_compile_first":
            initial_eval = eager_eval()
        else:
            initial_eval = compiled_eval()

        # Match the scientific trainer's compile-with-backward probe but exclude it from timed training.
        base.train()
        base.zero_grad(set_to_none=True)
        compile_gen = torch.Generator(device="cpu").manual_seed(COMPILE_PROBE_SEED)
        x0, y0 = train_data.batch(BATCH_SIZE, SEQ_LEN, compile_gen, device)
        torch.cuda.synchronize(device)
        compile_t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits0 = compiled(x0)
            loss0 = F.cross_entropy(logits0.float().reshape(-1, cfg.vocab_size), y0.reshape(-1))
        loss0.backward()
        torch.cuda.synchronize(device)
        compile_seconds = time.perf_counter() - compile_t0
        compile_loss = float(loss0.detach().cpu())
        base.zero_grad(set_to_none=True)

        optimizer = torch.optim.AdamW(
            base.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY, fused=True,
        )
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
            optimizer.zero_grad(set_to_none=True)
            x, y = train_data.batch(BATCH_SIZE, SEQ_LEN, batch_gen, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled(x)
                loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
            last_loss = float(loss.detach().cpu())
            if not math.isfinite(last_loss):
                raise RuntimeError(f"non-finite train loss in {variant}: {last_loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optimizer.step()
            steps += 1
            tokens += BATCH_SIZE * SEQ_LEN
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - t0
        final_eval = eager_eval()

        return _encode({
            "schema": 1,
            "job_id": JOB_ID,
            "status": "complete",
            "variant": variant,
            "engineering_only": True,
            "scientific_execution": False,
            "engineering_seed": ENGINEERING_SEED,
            "batch_seed": BATCH_SEED,
            "eval_seed": EVAL_SEED,
            "compile_probe_seed": COMPILE_PROBE_SEED,
            "runtime": runtime,
            "data": {"metadata": meta, "train_sha256": train_sha, "val_sha256": val_sha},
            "initial_eval": initial_eval,
            "compile_probe": {"seconds": compile_seconds, "loss": compile_loss},
            "training": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "schedule": "constant",
                "steps": steps,
                "tokens_seen": tokens,
                "actual_train_seconds": train_seconds,
                "tokens_per_second": tokens / max(train_seconds, 1e-9),
                "optimizer_steps_per_second": steps / max(train_seconds, 1e-9),
                "last_loss": last_loss,
                "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            },
            "final_eval": final_eval,
            "finished_unix": time.time(),
        })
    except Exception as exc:
        return _encode({
            "schema": 1,
            "job_id": JOB_ID,
            "status": "failed",
            "variant": variant,
            "engineering_only": True,
            "scientific_execution": False,
            "started_unix": started,
            "finished_unix": time.time(),
            "runtime": runtime,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scientific_conclusion": None,
        })


@app.function(
    image=image, gpu="A100-80GB", cpu=4.0, memory=65536,
    timeout=55 * 60, retries=0, max_containers=1,
)
def run_train_compile_first(job_json: str) -> str:
    return _run_variant_impl(job_json, "train_compile_first")


@app.function(
    image=image, gpu="A100-80GB", cpu=4.0, memory=65536,
    timeout=55 * 60, retries=0, max_containers=1,
)
def run_compiled_eval_first(job_json: str) -> str:
    return _run_variant_impl(job_json, "compiled_eval_first")


@app.local_entrypoint()
def main(job_path: str) -> None:
    raw = Path(job_path).read_text()
    first = _decode(run_train_compile_first.remote(raw))
    second = _decode(run_compiled_eval_first.remote(raw))
    status = "complete" if first.get("status") == second.get("status") == "complete" else "failed"
    payload: dict[str, Any] = {
        "schema": 1,
        "job_id": JOB_ID,
        "status": status,
        "engineering_only": True,
        "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "results": {
            "train_compile_first": first,
            "compiled_eval_first": second,
        },
        "scientific_conclusion": None,
    }
    if status == "complete":
        a = float(first["training"]["tokens_per_second"])
        b = float(second["training"]["tokens_per_second"])
        ratio = b / max(a, 1e-9)
        payload["derived"] = {
            "compiled_eval_first_over_train_compile_first_tps": ratio,
            "train_compile_first_over_compiled_eval_first_tps": a / max(b, 1e-9),
            "absolute_tps_difference": b - a,
        }
        payload["classification"] = (
            "CONFIRMED_LARGE_COMPILEORDER_THROUGHPUT_EFFECT"
            if ratio < 0.80 or ratio > 1.25
            else "NO_LARGE_COMPILEORDER_THROUGHPUT_EFFECT"
        )
        payload["interpretation_ceiling"] = (
            "Engineering-only compiler/runtime diagnostic. It tests whether compiled-evaluation-before-training "
            "changes measured light-state training throughput. It cannot establish architecture quality or a "
            "scientific model advantage."
        )
    print("RLT_SYSTEMS_LIGHTSTATE_COMPILEORDER_AF_RESULT_B64=" + _encode(payload))
