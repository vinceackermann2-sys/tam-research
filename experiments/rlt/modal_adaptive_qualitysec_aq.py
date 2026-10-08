from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import modal

JOB_ID = "rlt-systems-adaptive-realdata-modal-20261008-aq"
ENGINEERING_SEED = 20_261_060
BATCH_SEED = 20_271_060
EVAL_SEED = 20_291_060
COMPILE_PROBE_SEED = 20_301_060
SEQ_LEN = 64
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
EVAL_BATCHES = 32
TIME_BUDGET_SECONDS = 60.0
EXPECTED_PARAMETERS = 15_129_344
TRAIN_TOKENS = 6_000_000
VAL_TOKENS = 500_000
TRAINING_DATA_SEED = 20_260_924
EXPECTED_TRAIN_SHA = "0f9ea99680f532f990ca64fabbd02b46bba2c4f8077f868545a03f5e6907c907"
EXPECTED_VAL_SHA = "e6b3fb209a80a2cebe22924c3c8af26b30fc4afd322a1a49c71af8b968057db2"
CANDIDATES = (
    ("adaptive_scan", "cosine", 1e-3),
    ("residual_scan", "cosine", 1e-3),
    ("transformer_control", "cosine", 1e-3),
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11", "numpy>=2.0,<3",
        "datasets>=4.0,<5", "transformers>=4.55,<5",
        "tokenizers>=0.21,<1", "huggingface-hub>=0.34,<1",
    )
    .add_local_python_source("tam_research", "experiments")
)
app = modal.App("tam-rlt-systems-adaptive-realdata-20261008-aq")


def _encode(obj: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4.0,
    memory=65536,
    timeout=95 * 60,
    retries=0,
    max_containers=1,
)
def run_quality_remote(job_json: str) -> str:
    import torch
    import torch.nn.functional as F

    from experiments.rlt.model import RLTConfig, parameter_count
    from experiments.rlt.model_gated_scan import (
        associative_affine_scan, sequential_affine_reference,
    )
    from experiments.rlt.model_gated_scan_residual import ResidualGatedScanRLT
    from experiments.rlt.model_gated_scan_adaptive import AdaptiveResidualGatedScanRLT
    from tam_research.data import TokenBin, prepare_fineweb
    from tam_research.models import ModelConfig, ResearchLM
    from tam_research.train import seed_all

    started = time.time()
    prereg = json.loads(job_json)
    expected = {
        "schema": 1,
        "job_id": JOB_ID,
        "task": "systems_adaptive_realdata_qualitysec_aq",
        "engineering_seed": ENGINEERING_SEED,
        "batch_seed": BATCH_SEED,
        "eval_seed": EVAL_SEED,
        "compile_probe_seed": COMPILE_PROBE_SEED,
        "requires_ap_scientific_job_id": "rlt-residual-gated-scan-scientific-scale15m-60s-modal-20261008-ap",
        "requires_adaptive_cpu_tests_run_id": 37844155104,
        "candidate_protocols": [
            {"name": name, "schedule": schedule, "learning_rate": lr}
            for name, schedule, lr in CANDIDATES
        ],
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_batches": EVAL_BATCHES,
        "time_budget_seconds_each": TIME_BUDGET_SECONDS,
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "training_data_seed": TRAINING_DATA_SEED,
        "expected_parameters_each": EXPECTED_PARAMETERS,
        "automatic_retry": False,
    }
    if prereg != expected:
        raise RuntimeError(f"AQ residual quality/sec preregistration mismatch: {prereg}")
    if not torch.cuda.is_available():
        raise RuntimeError("AQ requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )
    transf_cfg = ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938,
        architecture="transformer",
    )
    classes = {
        "adaptive_scan": (AdaptiveResidualGatedScanRLT, cfg),
        "residual_scan": (ResidualGatedScanRLT, cfg),
        "transformer_control": (ResearchLM, transf_cfg),
    }

    data_dir = Path("/tmp/rlt-adaptive-realdata-aq")
    meta = prepare_fineweb(
        str(data_dir), train_tokens=TRAIN_TOKENS,
        val_tokens=VAL_TOKENS, seed=TRAINING_DATA_SEED,
    )
    train_sha = _sha256(data_dir / "train.bin")
    val_sha = _sha256(data_dir / "val.bin")
    if train_sha != EXPECTED_TRAIN_SHA or val_sha != EXPECTED_VAL_SHA:
        raise RuntimeError(f"Pinned data mismatch: {train_sha}, {val_sha}")
    train = TokenBin(str(data_dir / "train.bin"))
    val = TokenBin(str(data_dir / "val.bin"))

    # Check the mathematical scan identity on GPU at the actual runtime.
    gen = torch.Generator(device="cuda").manual_seed(COMPILE_PROBE_SEED)
    aa = torch.sigmoid(torch.randn(2, 17, 8, generator=gen, device=device))
    bb = torch.randn(2, 17, 8, generator=gen, device=device) * 0.1
    state = torch.randn(8, generator=gen, device=device) * 0.1
    torch.testing.assert_close(
        associative_affine_scan(aa, bb, state),
        sequential_affine_reference(aa, bb, state),
        atol=2e-6, rtol=2e-6,
    )

    def new_model(name: str):
        seed_all(ENGINEERING_SEED)
        cls, model_cfg = classes[name]
        model = cls(model_cfg).to(device)
        count = parameter_count(model)
        if count != EXPECTED_PARAMETERS:
            raise RuntimeError(f"{name} parameter drift: {count}")
        return model

    def evaluate(model) -> dict[str, float | int | str]:
        model.eval()
        generator = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
        losses = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(EVAL_BATCHES):
                x, y = val.batch(EVAL_BATCH_SIZE, SEQ_LEN, generator, device)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, 50_257), y.reshape(-1)
                    )
                losses.append(float(loss.detach().cpu()))
        torch.cuda.synchronize(device)
        nll = sum(losses) / len(losses)
        return {
            "nll": nll, "perplexity": math.exp(min(nll, 20.0)),
            "eval_tokens": EVAL_BATCHES * EVAL_BATCH_SIZE * SEQ_LEN,
            "seconds": time.perf_counter() - t0, "execution": "eager_no_grad",
        }

    def train_candidate(name: str, schedule: str, learning_rate: float) -> dict[str, Any]:
        model = new_model(name)
        initial_eval = evaluate(model)
        compiled = torch.compile(model, fullgraph=True, dynamic=False, mode="default")
        opt = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.95),
            weight_decay=0.1, fused=True,
        )

        # Compile without changing weights or consuming timed training budget.
        compile_generator = torch.Generator(device="cpu").manual_seed(
            COMPILE_PROBE_SEED
        )
        x0, y0 = train.batch(BATCH_SIZE, SEQ_LEN, compile_generator, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        c0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits0 = compiled(x0)
            loss0 = F.cross_entropy(
                logits0.float().reshape(-1, 50_257), y0.reshape(-1)
            )
        loss0.backward()
        torch.cuda.synchronize(device)
        compile_seconds = time.perf_counter() - c0
        compile_loss = float(loss0.detach().cpu())
        model.zero_grad(set_to_none=True)

        batch_generator = torch.Generator(device="cpu").manual_seed(BATCH_SEED)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        steps = 0
        last_loss = float("nan")
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= TIME_BUDGET_SECONDS and steps > 0:
                break
            if schedule == "cosine":
                progress = min(elapsed / TIME_BUDGET_SECONDS, 1.0)
                warmup = 0.02
                if progress < warmup:
                    lr = learning_rate * max(progress / warmup, 1e-4)
                else:
                    position = min((progress - warmup) / (1.0 - warmup), 1.0)
                    lr = learning_rate * 0.5 * (1 + math.cos(math.pi * position))
                for group in opt.param_groups:
                    group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            x, y = train.batch(BATCH_SIZE, SEQ_LEN, batch_generator, device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = compiled(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, 50_257), y.reshape(-1)
                )
            last_loss = float(loss.detach().cpu())
            if not math.isfinite(last_loss):
                raise RuntimeError(f"nonfinite training loss: {name} {last_loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1

        torch.cuda.synchronize(device)
        actual_seconds = time.perf_counter() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        final_eval = evaluate(model)
        if not math.isfinite(float(final_eval["nll"])):
            raise RuntimeError(f"nonfinite validation NLL: {name}")
        result = {
            "name": name, "parameters": parameter_count(model),
            "schedule": schedule, "learning_rate": learning_rate,
            "initial_eval": initial_eval, "final_eval": final_eval,
            "compile_probe": {"seconds": compile_seconds, "loss": compile_loss},
            "actual_train_seconds": actual_seconds,
            "steps": steps, "tokens_seen": steps * BATCH_SIZE * SEQ_LEN,
            "tokens_per_second": steps * BATCH_SIZE * SEQ_LEN / actual_seconds,
            "last_train_loss": last_loss,
            "peak_vram_gb": peak_vram_gb,
        }
        del compiled, opt, model
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return result

    rows = [train_candidate(*candidate) for candidate in CANDIDATES]
    results = {row["name"]: row for row in rows}
    adaptive = results["adaptive_scan"]
    residual = results["residual_scan"]
    transformer = results["transformer_control"]
    if not all(row["parameters"] == EXPECTED_PARAMETERS for row in rows):
        raise RuntimeError("AQ model count drift")
    return _encode({
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "PASS_ADAPTIVE_GATED_SCAN_REALDATA_ENGINEERING_QUALITYSEC",
        "engineering_only": True, "scientific_execution": False,
        "breakthrough_claim_supported": False,
        "engineering_seed": ENGINEERING_SEED, "batch_seed": BATCH_SEED,
        "eval_seed": EVAL_SEED, "compile_probe_seed": COMPILE_PROBE_SEED,
        "started_unix": started, "finished_unix": time.time(),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "data": {"metadata": meta, "train_sha256": train_sha, "val_sha256": val_sha},
        "gpu_scan_vs_serial_passed": True,
        "exact_parameter_match_verified": True,
        "protocol": {
            "candidates": [
                {"name": name, "schedule": schedule, "learning_rate": lr}
                for name, schedule, lr in CANDIDATES
            ],
            "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
            "time_budget_seconds_each": TIME_BUDGET_SECONDS,
            "eval_batch_size": EVAL_BATCH_SIZE, "eval_batches": EVAL_BATCHES,
            "same_initialization_seed": True,
            "same_batch_stream_seed": True, "same_eval_stream_seed": True,
            "compile_time_excluded": True,
            "training_graph_compiled_before_timed_training": True,
            "validation_execution": "eager_no_grad",
            "weight_decay": 0.1, "gradient_clip_norm": 1.0,
            "adaptive_optimizer_borrowed_from_AO_not_independently_calibrated": True,
            "residual_optimizer_frozen_from_AO": True,
            "transformer_optimizer_frozen_from_prior_transformer_calibration": True,
            "adaptive_is_new_non_equivalent_architecture": True,
        },
        "results": rows,
        "derived": {
            "adaptive_minus_transformer_final_nll":
                adaptive["final_eval"]["nll"] - transformer["final_eval"]["nll"],
            "adaptive_minus_residual_final_nll":
                adaptive["final_eval"]["nll"] - residual["final_eval"]["nll"],
            "adaptive_over_transformer_tps":
                adaptive["tokens_per_second"] / transformer["tokens_per_second"],
            "adaptive_over_residual_tps":
                adaptive["tokens_per_second"] / residual["tokens_per_second"],
        },
        "interpretation_ceiling": (
            "AQ single engineering seed, 15,129,344 parameters, pinned FineWeb-Edu D, "
            "equal 60-second postcompile windows with cosine LR 1e-3 for all three "
            "architectures. Adaptive LR borrowed from AO residual and is not "
            "independently calibrated. This is an engineering screen only, not a "
            "scientific demonstration, replication, breakthrough, or scale gate."
        ),
    })

@app.function(image=image, cpu=2.0, timeout=300, retries=0, max_containers=1)
def verify_image_remote() -> str:
    import datasets
    import numpy
    import torch
    import transformers

    from experiments.rlt.model import RLTConfig, parameter_count
    from experiments.rlt.model_gated_scan_residual import ResidualGatedScanRLT
    from experiments.rlt.model_gated_scan_adaptive import AdaptiveResidualGatedScanRLT
    from tam_research.models import ModelConfig, ResearchLM
    from tam_research.data import TokenBin, prepare_fineweb
    from tam_research.train import seed_all

    torch.set_num_threads(2)
    cfg = RLTConfig(
        vocab_size=50_257, d_model=256, n_heads=8, n_stages=2,
        max_seq_len=128, ff_mult=4, swa_window=32,
    )
    transf_cfg = ModelConfig(
        vocab_size=50_257, d_model=256, n_layers=3, n_heads=8,
        max_seq_len=128, ff_mult=4, ff_inner=938, architecture="transformer",
    )
    counts = {}
    for name, cls, model_cfg in [
        ("adaptive_scan", AdaptiveResidualGatedScanRLT, cfg),
        ("residual_scan", ResidualGatedScanRLT, cfg),
        ("transformer_control", ResearchLM, transf_cfg),
    ]:
        seed_all(ENGINEERING_SEED)
        model = cls(model_cfg)
        counts[name] = parameter_count(model)
        if counts[name] != EXPECTED_PARAMETERS:
            raise RuntimeError(f"AQ {name} parameter mismatch")
        with torch.no_grad():
            out = model(torch.tensor([[5, 6]], dtype=torch.long))
        if out.shape != (1, 2, 50_257) or not torch.isfinite(out).all():
            raise RuntimeError(f"AQ {name} CPU forward failure")
    return _encode({
        "status": "pass", "gpu_requested": False,
        "cpu_forward_shapes_passed": True,
        "parameter_counts": counts,
        "numpy_version": str(numpy.__version__),
        "torch_version": str(torch.__version__),
        "datasets_version": str(datasets.__version__),
        "transformers_version": str(transformers.__version__),
    })


@app.local_entrypoint()
def verify_dependencies() -> None:
    result = verify_image_remote.remote()
    if not isinstance(result, str):
        raise TypeError("AQ image preflight must return base64 string")
    print("RLT_SYSTEMS_ADAPTIVE_AQ_PREFLIGHT_B64=" + result)


@app.local_entrypoint()
def main(job_path: str) -> None:
    result = run_quality_remote.remote(Path(job_path).read_text())
    if not isinstance(result, str):
        raise TypeError("AQ remote result must be base64 string")
    print("RLT_SYSTEMS_ADAPTIVE_AQ_RESULT_B64=" + result)
