from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import modal

# PREPARATION-ONLY HARNESS UNTIL THE EXACT OWNER-ONLY PAID TRIGGER IS
# SEPARATELY AUTHORIZED AND OPENED.
PHASE = "full-v11-compiled-explicit-fp32-ce"
TRIGGER_TITLE = "[modal-cortex-s-100m-2b-full-v11-compiled-explicit-fp32-ce]"
FULL_TRAINING_SEED = 2_026_091_014
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/full-v11-compiled-explicit-fp32-ce"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

V11_PREFLIGHT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v11-compiled-explicit-fp32-ce"
V11_PREFLIGHT_SOURCE_SHA = "be5c0c19c7ebf3339e0c8be80158eb52b1a3dae2"
V11_PREFLIGHT_SOURCE_TREE = "2820a74ad1ea4e694cbf0a08ce42011e9e483d6e"
V11_PREFLIGHT_HARNESS_SHA = "dcbec42960139757bf0cc528a099edea433edd66"
V11_PREFLIGHT_SEED = 2_026_091_013
V11_PREFLIGHT_STATUS = "SYSTEMS_PREFLIGHT_PASS"
V11_PREFLIGHT_CLASSIFICATION = "ENGINEERING_SYSTEMS_PREFLIGHT_V11_COMPILED_EXPLICIT_FP32_CE_ONLY"
V11_EXPECTED_TPS = 287_002.5286255846
V11_EXPECTED_COMPILE_SECONDS = 371.187105083
V11_EXPECTED_PROJECTED_FULL_SECONDS = 8_036.624794902304
V11_EXPECTED_PEAK_VRAM_GIB = 37.06346607208252

TRAIN_IMPLEMENTATION_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"
TRAIN_CORPUS_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_CORPUS_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_CORPUS_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"

EXPECTED_CORTEX_PARAMS = 101_778_112
TRAIN_TOKENS = 2_000_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
TOKENS_PER_STEP = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
TOTAL_OPTIMIZER_STEPS = 30_518
FULL_BATCH_TOKEN_EXPOSURES = 2_000_027_648
EVAL_EVERY_TOKENS = 200_000_000
CHECKPOINT_EVERY_TOKENS = 200_000_000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.02
MODEL_COMPILE_MODE = "max-autotune-no-cudagraphs"
LOSS_COMPILE_MODE = "max-autotune-no-cudagraphs"
LOSS_FULLGRAPH = True
MAX_PROJECTED_FULL_SECONDS = 8_500.0
HARD_FULL_TIMEOUT_SECONDS = 10_000

FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)
RETIRED_OR_CONSUMED_ENGINEERING_SEEDS = (
    910_001,
    2_026_090_901,
    2_026_090_902,
    2_026_090_903,
    2_026_090_904,
    2_026_090_905,
    2_026_090_906,
    2_026_090_907,
    2_026_091_001,
    2_026_091_002,
    2_026_091_003,
    2_026_091_004,
    2_026_091_005,
    2_026_091_006,
    2_026_091_007,
    2_026_091_008,
    2_026_091_009,
    2_026_091_010,
    2_026_091_011,
    2_026_091_012,
    2_026_091_013,
)

APP_NAME = "cortex-s-v0-100m-2b-full-v11-compiled-explicit-fp32-ce"
VOLUME_NAME = "tam-research-data"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
    .add_local_python_source("architectures")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _full_sha(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase SHA")
    return normalized


def _validate_payload(source_sha: str, source_tree: str, harness_sha: str) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    forbidden = set(FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS) | set(RETIRED_OR_CONSUMED_ENGINEERING_SEEDS)
    if FULL_TRAINING_SEED in forbidden:
        raise RuntimeError("v11 full-training seed is consumed, retired, or scientifically reserved")
    if TOKENS_PER_STEP != 65_536:
        raise RuntimeError("token-per-step contract drift")
    if TOTAL_OPTIMIZER_STEPS * TOKENS_PER_STEP != FULL_BATCH_TOKEN_EXPOSURES:
        raise RuntimeError("full-batch exposure contract drift")
    if TOTAL_OPTIMIZER_STEPS != math.ceil(TRAIN_TOKENS / TOKENS_PER_STEP):
        raise RuntimeError("optimizer-step contract drift")
    return source, tree, harness


def _load_authoritative_v11_result() -> dict[str, Any]:
    result_path = Path(V11_PREFLIGHT_ROOT) / "RESULT.json"
    if not result_path.exists():
        raise FileNotFoundError("authoritative v11 RESULT.json is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    exact = {
        "status": V11_PREFLIGHT_STATUS,
        "classification": V11_PREFLIGHT_CLASSIFICATION,
        "source_sha": V11_PREFLIGHT_SOURCE_SHA,
        "source_tree": V11_PREFLIGHT_SOURCE_TREE,
        "harness_sha": V11_PREFLIGHT_HARNESS_SHA,
        "engineering_seed": V11_PREFLIGHT_SEED,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
        "seed_8100_used": False,
        "reserved_scientific_seeds_used": False,
    }
    for key, expected in exact.items():
        if result.get(key) != expected:
            raise RuntimeError(f"authoritative v11 result drift for {key}: {result.get(key)!r}")
    dispatcher = result.get("dispatcher_proof") or {}
    if dispatcher.get("candidate_calls") != 24 or dispatcher.get("legacy_fallback_calls") != 0:
        raise RuntimeError("authoritative v11 Triton dispatcher proof drift")
    proof = result.get("compiled_explicit_fp32_ce_cuda_proof") or {}
    if not proof.get("loss_matches_explicit_fp32_reference"):
        raise RuntimeError("authoritative v11 explicit-FP32 loss proof is not PASS")
    if not proof.get("gradient_matches_explicit_fp32_reference"):
        raise RuntimeError("authoritative v11 explicit-FP32 gradient proof is not PASS")
    if float(proof.get("max_gradient_abs_diff", float("inf"))) != 0.0:
        raise RuntimeError("authoritative v11 gradient proof drift")
    calibration = result.get("calibration") or {}
    if calibration.get("execution") != "compiled" or calibration.get("compile_error") not in (None, ""):
        raise RuntimeError("authoritative v11 execution/compile contract drift")
    expected_numbers = {
        "training_tokens_per_second": V11_EXPECTED_TPS,
        "compile_seconds": V11_EXPECTED_COMPILE_SECONDS,
        "projected_full_seconds": V11_EXPECTED_PROJECTED_FULL_SECONDS,
        "peak_vram_gb": V11_EXPECTED_PEAK_VRAM_GIB,
    }
    for key, expected in expected_numbers.items():
        actual = float(calibration.get(key, float("nan")))
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
            raise RuntimeError(f"authoritative v11 calibration drift for {key}: {actual}")
    if float(calibration["projected_full_seconds"]) > MAX_PROJECTED_FULL_SECONDS:
        raise RuntimeError("authoritative v11 projection no longer satisfies the frozen gate")
    return result


def _verify_frozen_inputs() -> dict[str, Any]:
    import inspect

    import architectures.cortex_s.compiled_explicit_fp32_ce_v1 as candidate
    import architectures.cortex_s.experiments.scale100m_2b.protocol as protocol
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.language_model import parameter_count
    from architectures.cortex_s.production_scan_integration_v1 import (
        build_memory_lean_grouped_triton_scan_cortex_100m,
        integration_status as scan_status,
    )

    training_path = Path(inspect.getsourcefile(training_module) or "")
    if not training_path.exists() or _sha256(training_path) != TRAIN_IMPLEMENTATION_SHA256:
        raise RuntimeError("frozen production trainer fingerprint drift")

    data_root = Path(DATA_DIR)
    hashes = {
        "train_sha256": _sha256(data_root / "train.bin"),
        "val_sha256": _sha256(data_root / "val.bin"),
        "meta_sha256": _sha256(data_root / "meta.json"),
    }
    expected_hashes = {
        "train_sha256": TRAIN_CORPUS_SHA256,
        "val_sha256": VAL_CORPUS_SHA256,
        "meta_sha256": META_CORPUS_SHA256,
    }
    if hashes != expected_hashes:
        raise RuntimeError(f"frozen corpus fingerprint drift: {hashes}")

    meta = json.loads((data_root / "meta.json").read_text(encoding="utf-8"))
    if meta.get("train_tokens") != TRAIN_TOKENS or meta.get("val_tokens") != 5_000_000:
        raise RuntimeError("frozen corpus token counts drift")
    if meta.get("tokenizer") != "gpt2" or meta.get("dtype") != "uint16":
        raise RuntimeError("frozen corpus tokenizer/dtype drift")

    candidate_status = candidate.integration_status()
    required_candidate = {
        "systems_variant": "compiled_explicit_fp32_ce_v1",
        "loss_compile_mode": LOSS_COMPILE_MODE,
        "loss_fullgraph": True,
        "explicit_full_logits_fp32_cast_preserved": True,
        "cross_entropy_semantics_preserved": True,
        "single_host_readback_preserved": True,
        "liger_used": False,
    }
    for key, expected in required_candidate.items():
        if candidate_status.get(key) != expected:
            raise RuntimeError(f"v11 candidate drift for {key}")

    scan = scan_status()
    if scan.get("systems_variant") != "memory_lean_grouped_v4_triton_scan_v1":
        raise RuntimeError("production scan integration identity drift")
    if scan.get("cuda_backend") != "affine_scan_triton_candidate":
        raise RuntimeError("production scan backend drift")

    frozen_protocol = {
        "TRAIN_TOKENS": TRAIN_TOKENS,
        "SEQ_LEN": SEQ_LEN,
        "MICRO_BATCH_SIZE": MICRO_BATCH_SIZE,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        "TOTAL_OPTIMIZER_STEPS": TOTAL_OPTIMIZER_STEPS,
        "FULL_BATCH_TOKEN_EXPOSURES": FULL_BATCH_TOKEN_EXPOSURES,
        "EVAL_EVERY_TOKENS": EVAL_EVERY_TOKENS,
        "CHECKPOINT_EVERY_TOKENS": CHECKPOINT_EVERY_TOKENS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_RATIO": WARMUP_RATIO,
    }
    for name, expected in frozen_protocol.items():
        if getattr(protocol, name) != expected:
            raise RuntimeError(f"frozen protocol drift for {name}: {getattr(protocol, name)!r}")
    if training_module.COMPILE_MODE != MODEL_COMPILE_MODE:
        raise RuntimeError("production model compile mode drift")

    model = build_memory_lean_grouped_triton_scan_cortex_100m()
    if parameter_count(model) != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError("CORTEX-S parameter count drift")
    return {
        **hashes,
        "metadata": meta,
        "candidate_status": candidate_status,
        "scan_status": scan,
        "parameter_count": parameter_count(model),
        "trainer_sha256": TRAIN_IMPLEMENTATION_SHA256,
    }


@app.function(image=base_image, cpu=4, memory=8192, timeout=30 * 60, volumes={"/vol": volume})
def verify_zero_gpu(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("reserved v11 full-training namespace already exists; fail closed")
    v11_result = _load_authoritative_v11_result()
    frozen = _verify_frozen_inputs()
    payload = {
        "status": "PASS",
        "classification": "ZERO_GPU_FULL_2B_V11_PREPARATION_GATE",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "full_training_seed_reserved": FULL_TRAINING_SEED,
        "result_root": RESULT_ROOT,
        "gpu_allocated": False,
        "authoritative_v11_result": {
            "status": v11_result["status"],
            "classification": v11_result["classification"],
            "source_sha": v11_result["source_sha"],
            "source_tree": v11_result["source_tree"],
            "harness_sha": v11_result["harness_sha"],
            "engineering_seed": v11_result["engineering_seed"],
            "projected_full_seconds": v11_result["calibration"]["projected_full_seconds"],
        },
        "frozen_inputs": frozen,
        "training_contract": {
            "train_tokens": TRAIN_TOKENS,
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "tokens_per_step": TOKENS_PER_STEP,
            "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
            "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
            "eval_every_tokens": EVAL_EVERY_TOKENS,
            "checkpoint_every_tokens": CHECKPOINT_EVERY_TOKENS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "model_compile_mode": MODEL_COMPILE_MODE,
            "loss_compile_mode": LOSS_COMPILE_MODE,
            "loss_fullgraph": LOSS_FULLGRAPH,
            "hard_timeout_seconds": HARD_FULL_TIMEOUT_SECONDS,
            "resume_authorized": False,
        },
        "full_training_authorized_by_zero_gpu_gate": False,
        "scientific_claim_authorized": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.function(image=base_image, cpu=2, memory=2048, timeout=10 * 60, volumes={"/vol": volume})
def reserve_full_dispatch(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    gate_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "FULL_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not gate_path.exists():
        raise RuntimeError("zero-GPU full-training gate is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "PASS",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "full_training_seed_reserved": FULL_TRAINING_SEED,
    }.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"zero-GPU full gate mismatch for {key}")
    if marker_path.exists() or result_path.exists():
        raise RuntimeError("v11 full-training dispatch/result already consumed")
    marker = {
        "status": "FULL_DISPATCH_CONSUMED",
        "classification": "ENGINEERING_SINGLE_ATTEMPT_FULL_2B_V11",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "full_training_seed": FULL_TRAINING_SEED,
        "result_root": RESULT_ROOT,
        "h100_allocation_started": False,
        "resume_authorized": False,
        "scientific_claim_authorized": False,
        "marked_unix": time.time(),
    }
    _atomic_write(marker_path, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


def _save_v11_checkpoint(
    *,
    path: Path,
    model: Any,
    optimizer: Any,
    step: int,
    tokens_seen: int,
    generator: Any,
    source_sha: str,
) -> None:
    import torch
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    payload = {
        "experiment": "cortex-s-v11-full-2b-compiled-explicit-fp32-ce",
        "classification": "ENGINEERING_FULL_2B_TRAINING_CHECKPOINT_ONLY",
        "source_sha": source_sha,
        "architecture": "cortex_s",
        "seed": FULL_TRAINING_SEED,
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "batch_generator_state": generator.get_state(),
        "config": asdict(training_module.CORTEX_100M_CONFIG),
        "execution": "compiled",
        "resume_authorized": False,
        "scientific_claim_authorized": False,
    }
    tmp = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(path)


def _run_full_training(source_sha: str) -> dict[str, Any]:
    import torch

    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.compiled_explicit_fp32_ce_v1 import compiled_explicit_fp32_ce_training_builder
    from architectures.cortex_s.language_model import parameter_count
    from architectures.cortex_s.production_scan_integration_v1 import memory_lean_grouped_triton_scan_training_builder
    from tam_research.data import TokenBin
    from tam_research.train import cosine_lr

    if not torch.cuda.is_available():
        raise RuntimeError("v11 full 2B training requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("v11 full 2B training requires H100/SM90-class CUDA")

    run_dir = Path(RESULT_ROOT)
    started_path = run_dir / "ATTEMPT_STARTED.json"
    result_path = run_dir / "RESULT.json"
    success_path = run_dir / "SUCCESS.json"
    if started_path.exists() or result_path.exists() or success_path.exists():
        raise RuntimeError("v11 full-training namespace already contains terminal/attempt state")

    _atomic_write(
        started_path,
        {
            "status": "ATTEMPT_STARTED",
            "source_sha": source_sha,
            "seed": FULL_TRAINING_SEED,
            "tokens": TRAIN_TOKENS,
            "execution": "compiled",
            "started_unix": time.time(),
            "resume_authorized": False,
            "scientific_claim_authorized": False,
        },
    )
    volume.commit()

    torch.set_float32_matmul_precision("high")
    training_module.seed_all(FULL_TRAINING_SEED)

    with memory_lean_grouped_triton_scan_training_builder():
        with compiled_explicit_fp32_ce_training_builder():
            model = training_module.build_cortex_100m().to(device)
            optimizer = training_module._make_optimizer(model)
            compile_start = time.perf_counter()
            runner = training_module._compile_model(model)

            warm_generator = torch.Generator(device="cpu").manual_seed(FULL_TRAINING_SEED + 99_999)
            warm_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
            training_module._one_optimizer_step(
                model=model,
                runner=runner,
                optimizer=optimizer,
                train_data=warm_data,
                generator=warm_generator,
                device=device,
                lr=LEARNING_RATE,
            )

            # Reset all trainable/scientific state after compile warmup. The warmup
            # generator is separate and never advances the actual training stream.
            training_module.seed_all(FULL_TRAINING_SEED)
            model = training_module.build_cortex_100m().to(device)
            optimizer = training_module._make_optimizer(model)
            runner = training_module._compile_model(model)
            torch.cuda.synchronize(device)
            compile_seconds = max(time.perf_counter() - compile_start, 0.0)

            if parameter_count(model) != EXPECTED_CORTEX_PARAMS:
                raise RuntimeError("parameter-count drift before full training")

            train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
            val_data = TokenBin(str(Path(DATA_DIR) / "val.bin"))
            generator = torch.Generator(device="cpu").manual_seed(FULL_TRAINING_SEED + 10_000)
            warmup_steps = max(1, int(TOTAL_OPTIMIZER_STEPS * WARMUP_RATIO))
            next_eval = EVAL_EVERY_TOKENS
            next_checkpoint = CHECKPOINT_EVERY_TOKENS
            curve: list[dict[str, Any]] = []
            training_seconds = 0.0
            eval_seconds = 0.0
            checkpoint_seconds = 0.0
            tokens_seen = 0
            latest_path = run_dir / "latest.pt"
            progress_path = run_dir / "PROGRESS.json"

            torch.cuda.reset_peak_memory_stats(device)
            wall_start = time.perf_counter()
            last_train_loss = float("nan")
            for step in range(TOTAL_OPTIMIZER_STEPS):
                torch.cuda.synchronize(device)
                train_start = time.perf_counter()
                lr = cosine_lr(step, TOTAL_OPTIMIZER_STEPS, warmup_steps, LEARNING_RATE)
                last_train_loss = training_module._one_optimizer_step(
                    model=model,
                    runner=runner,
                    optimizer=optimizer,
                    train_data=train_data,
                    generator=generator,
                    device=device,
                    lr=lr,
                )
                torch.cuda.synchronize(device)
                training_seconds += max(time.perf_counter() - train_start, 0.0)
                tokens_seen = min(TRAIN_TOKENS, (step + 1) * TOKENS_PER_STEP)

                evaluated = False
                if tokens_seen >= next_eval or tokens_seen >= TRAIN_TOKENS:
                    eval_start = time.perf_counter()
                    ev = training_module.evaluate(
                        model,
                        runner,
                        val_data,
                        batches=20,
                        seed=FULL_TRAINING_SEED + 20_000,
                    )
                    torch.cuda.synchronize(device)
                    eval_seconds += max(time.perf_counter() - eval_start, 0.0)
                    curve.append(
                        {
                            "step": step + 1,
                            "tokens_seen": tokens_seen,
                            "training_seconds": training_seconds,
                            "wall_seconds": time.perf_counter() - wall_start,
                            "nll": ev["nll"],
                            "perplexity": ev["perplexity"],
                            "last_train_loss": last_train_loss,
                            "lr": lr,
                        }
                    )
                    while next_eval <= tokens_seen:
                        next_eval += EVAL_EVERY_TOKENS
                    evaluated = True

                checkpointed = False
                if tokens_seen >= next_checkpoint or tokens_seen >= TRAIN_TOKENS:
                    checkpoint_start = time.perf_counter()
                    _save_v11_checkpoint(
                        path=latest_path,
                        model=model,
                        optimizer=optimizer,
                        step=step + 1,
                        tokens_seen=tokens_seen,
                        generator=generator,
                        source_sha=source_sha,
                    )
                    checkpoint_seconds += max(time.perf_counter() - checkpoint_start, 0.0)
                    while next_checkpoint <= tokens_seen:
                        next_checkpoint += CHECKPOINT_EVERY_TOKENS
                    checkpointed = True

                if evaluated or checkpointed:
                    _atomic_write(
                        progress_path,
                        {
                            "status": "IN_PROGRESS",
                            "source_sha": source_sha,
                            "seed": FULL_TRAINING_SEED,
                            "step": step + 1,
                            "tokens_seen": tokens_seen,
                            "last_train_loss": last_train_loss,
                            "curve": curve,
                            "resume_authorized": False,
                            "scientific_claim_authorized": False,
                        },
                    )
                    volume.commit()

            final_eval_start = time.perf_counter()
            final_eval = training_module.evaluate(
                model,
                runner,
                val_data,
                batches=50,
                seed=FULL_TRAINING_SEED + 30_000,
            )
            torch.cuda.synchronize(device)
            eval_seconds += max(time.perf_counter() - final_eval_start, 0.0)
            wall_seconds = max(time.perf_counter() - wall_start, 1e-9)
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            training_tps = TRAIN_TOKENS / max(training_seconds, 1e-9)
            baseline = dict(training_module.BASELINE_TRANSFORMER)

            return {
                "status": "COMPLETE",
                "classification": "ENGINEERING_FULL_2B_V11_TRAINING_ONLY",
                "scientific_claim_authorized": False,
                "breakthrough_claim_allowed": False,
                "source_sha": source_sha,
                "architecture": "cortex_s",
                "parameters": parameter_count(model),
                "seed": FULL_TRAINING_SEED,
                "tokens_seen": TRAIN_TOKENS,
                "full_batch_token_exposures": FULL_BATCH_TOKEN_EXPOSURES,
                "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
                "seq_len": SEQ_LEN,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "execution": "compiled",
                "compile_seconds": compile_seconds,
                "training_seconds": training_seconds,
                "eval_seconds": eval_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "wall_seconds": wall_seconds,
                "total_compute_seconds": wall_seconds + compile_seconds,
                "training_tokens_per_second": training_tps,
                "peak_vram_gb": peak_vram_gb,
                "last_train_loss": last_train_loss,
                "final_eval": final_eval,
                "curve": curve,
                "router": model.router_stats(),
                "state_policy": "reset_per_random_training_sequence",
                "resume_authorized": False,
                "historical_transformer": baseline,
                "descriptive_historical_comparison_only": {
                    "equal_token_nll_delta_cortex_minus_transformer": (
                        final_eval["nll"] - float(baseline["final_nll"])
                    ),
                    "equal_token_nll_cortex_better": final_eval["nll"] < float(baseline["final_nll"]),
                    "throughput_ratio_cortex_over_transformer": (
                        training_tps / float(baseline["training_tokens_per_second"])
                    ),
                    "same_seed": False,
                    "same_pretraining_bytes": True,
                    "same_optimizer_hparams": True,
                    "same_context_and_global_batch": True,
                    "note": (
                        "The Transformer is a historical control and used a different seed. "
                        "This result is engineering training evidence only and authorizes no scientific claim."
                    ),
                },
                "checkpoint": str(latest_path),
                "gpu_name": torch.cuda.get_device_name(device),
                "training_contract": {
                    "train_tokens": TRAIN_TOKENS,
                    "tokens_per_step": TOKENS_PER_STEP,
                    "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
                    "eval_every_tokens": EVAL_EVERY_TOKENS,
                    "checkpoint_every_tokens": CHECKPOINT_EVERY_TOKENS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO,
                    "model_compile_mode": MODEL_COMPILE_MODE,
                    "loss_compile_mode": LOSS_COMPILE_MODE,
                    "loss_fullgraph": LOSS_FULLGRAPH,
                },
            }


@app.function(
    image=base_image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=HARD_FULL_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
)
def h100_full_training(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    marker_path = root / "FULL_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not marker_path.exists():
        raise RuntimeError("durable full-training dispatch marker is missing")
    if result_path.exists():
        raise RuntimeError("v11 full-training RESULT.json already exists; refusing duplicate execution")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "FULL_DISPATCH_CONSUMED",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "full_training_seed": FULL_TRAINING_SEED,
    }.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"full dispatch marker mismatch for {key}")
    marker["h100_allocation_started"] = True
    marker["h100_function_started_unix"] = time.time()
    _atomic_write(marker_path, marker)
    volume.commit()

    try:
        training = _run_full_training(source)
        result = {
            "status": "FULL_TRAINING_COMPLETE",
            "classification": "ENGINEERING_FULL_2B_V11_COMPLETE_NO_SCIENTIFIC_CLAIM",
            "phase": PHASE,
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "full_training_seed": FULL_TRAINING_SEED,
            "result_root": RESULT_ROOT,
            "authoritative_v11_preflight_source_sha": V11_PREFLIGHT_SOURCE_SHA,
            "training": training,
            "resume_authorized": False,
            "scientific_claim_authorized": False,
        }
        _atomic_write(result_path, result)
        _atomic_write(
            root / "SUCCESS.json",
            {
                "status": "FULL_TRAINING_COMPLETE",
                "source_sha": source,
                "seed": FULL_TRAINING_SEED,
                "tokens_seen": training["tokens_seen"],
                "final_nll": training["final_eval"]["nll"],
                "resume_authorized": False,
                "scientific_claim_authorized": False,
            },
        )
        volume.commit()
        return json.dumps(result, sort_keys=True)
    except Exception as exc:
        failure = {
            "status": "FULL_TRAINING_ERROR",
            "classification": "ENGINEERING_FULL_2B_V11_ERROR_NO_RETRY",
            "phase": PHASE,
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "full_training_seed": FULL_TRAINING_SEED,
            "result_root": RESULT_ROOT,
            "error": f"{type(exc).__name__}: {exc}",
            "checkpoint_may_exist": (root / "latest.pt").exists(),
            "progress_may_exist": (root / "PROGRESS.json").exists(),
            "resume_authorized": False,
            "scientific_claim_authorized": False,
        }
        _atomic_write(result_path, failure)
        volume.commit()
        return json.dumps(failure, sort_keys=True)


@app.local_entrypoint()
def main(phase: str, source_sha: str, source_tree: str, harness_sha: str) -> None:
    if phase.strip().lower() != PHASE:
        raise RuntimeError(f"phase must be exactly {PHASE}")
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    zero_gpu = json.loads(verify_zero_gpu.remote(source, tree, harness))
    print(json.dumps({"zero_gpu": zero_gpu}, indent=2, sort_keys=True))
    reservation = json.loads(reserve_full_dispatch.remote(source, tree, harness))
    print(json.dumps({"dispatch_reservation": reservation}, indent=2, sort_keys=True))
    result = json.loads(h100_full_training.remote(source, tree, harness))
    print(json.dumps({"h100_full_training": result}, indent=2, sort_keys=True))
