from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Iterator

import modal

# This runner is engineering-preflight only. It deliberately exposes no full-2B
# function and cannot consume the paired historical-control seed 8100.
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"
TRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 5_000_000
TRAIN_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"

ENGINEERING_SEED = 2_026_091_002
MAX_PROJECTED_FULL_SECONDS = 8_500
HARD_FULL_TIMEOUT_SECONDS = 10_000
MAX_PREFLIGHT_VRAM_GIB = 70.0
CALIBRATION_STEPS = 40
CALIBRATION_MEASURED_TOKENS = 2_621_440
PROJECTION_OVERHEAD_MULTIPLIER = 1.10
EXPECTED_CORTEX_PARAMS = 101_778_112
PRODUCTION_MOE_BACKEND = "physical_padded_grouped_bf16"
BASE_GROUPED_SYSTEMS_VARIANT = "memory_lean_cast_before_gather_pad_v4"
INTEGRATED_SYSTEMS_VARIANT = "memory_lean_grouped_v4_triton_scan_v1"
PRODUCTION_SCAN_BACKEND = "affine_scan_triton_candidate"

CONSUMED_ENGINEERING_SEEDS = (
    910_001,
    2_026_090_901,
    2_026_090_902,
    2_026_090_903,
    2_026_090_904,
    2_026_090_905,
    2_026_090_906,
    2_026_090_907,
    2_026_091_001,
)
FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)

APP_NAME = "cortex-s-v0-100m-2b-preflight-v5-triton-scan"
VOLUME_NAME = "tam-research-data"
PREFLIGHT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v5-triton-scan"
PHASE = "preflight-v5-triton-scan"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
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


def _verify_frozen_data() -> dict[str, object]:
    root = Path(DATA_DIR)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    meta_path = root / "meta.json"
    missing = [str(path) for path in (train_path, val_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen 100M/2B corpus missing: {missing}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "assembly_version": 3,
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": VAL_TOKENS,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"data metadata mismatch for {key}: {meta.get(key)!r} != {expected!r}")
    if train_path.stat().st_size != TRAIN_TOKENS * 2:
        raise RuntimeError("train.bin is not exactly 2B uint16 tokens")
    if val_path.stat().st_size != VAL_TOKENS * 2:
        raise RuntimeError("val.bin is not exactly 5M uint16 tokens")

    train_hash = _sha256(train_path)
    val_hash = _sha256(val_path)
    meta_hash = _sha256(meta_path)
    if train_hash != TRAIN_SHA256:
        raise RuntimeError(f"train corpus fingerprint mismatch: {train_hash}")
    if val_hash != VAL_SHA256:
        raise RuntimeError(f"validation corpus fingerprint mismatch: {val_hash}")
    if meta_hash != META_SHA256:
        raise RuntimeError(f"metadata fingerprint mismatch: {meta_hash}")
    return {
        "meta": meta,
        "train_sha256": train_hash,
        "val_sha256": val_hash,
        "meta_sha256": meta_hash,
        "train_bytes": train_path.stat().st_size,
        "val_bytes": val_path.stat().st_size,
        "fingerprints_established_by": "issue-767-cpu-fingerprint-v2",
    }


def _validate_seed_contract() -> None:
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("v5 engineering seed is consumed or scientifically reserved")


@contextmanager
def _integrated_calibration_context() -> Iterator[object]:
    """Patch only the engineering calibration seed and production model builder.

    The frozen 100M/2B training/calibration implementation remains the timing source
    of truth. This context changes no optimizer, data, batch, compile, loss, or
    projection constants. It swaps only the already-merged production builder and
    the separately preregistered engineering calibration seed.
    """

    import architectures.cortex_s.experiments.scale100m_2b.protocol as protocol_module
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.production_scan_integration_v1 import (
        memory_lean_grouped_triton_scan_training_builder,
    )

    old_training_seed = training_module.CALIBRATION_SEED
    old_protocol_seed = protocol_module.CALIBRATION_SEED
    training_module.CALIBRATION_SEED = ENGINEERING_SEED
    protocol_module.CALIBRATION_SEED = ENGINEERING_SEED
    try:
        with memory_lean_grouped_triton_scan_training_builder():
            yield training_module
    finally:
        training_module.CALIBRATION_SEED = old_training_seed
        protocol_module.CALIBRATION_SEED = old_protocol_seed


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/vol": volume},
)
def verify_data_zero_gpu(source_sha: str) -> str:
    """Hash the frozen corpus and validate the exact integrated model with zero GPU."""

    from architectures.cortex_s.grouped_moe_v4 import (
        MemoryLeanPhysicalPaddedGroupedSparseMoE,
    )
    from architectures.cortex_s.language_model import parameter_count
    from architectures.cortex_s.production_scan_integration_v1 import (
        ProductionScanPersistentWorldState,
        build_memory_lean_grouped_triton_scan_cortex_100m,
        integration_status,
    )

    _validate_seed_contract()
    volume.reload()
    root = Path(PREFLIGHT_ROOT)
    data_path = root / "data.json"
    dispatch_path = root / "H100_PREFLIGHT_DISPATCHED.json"
    result_path = root / "h100.json"
    if data_path.exists() or dispatch_path.exists() or result_path.exists():
        raise RuntimeError("v5 Triton-scan preflight namespace is already consumed/touched")

    data = _verify_frozen_data()
    model = build_memory_lean_grouped_triton_scan_cortex_100m()
    actual = parameter_count(model)
    grouped_layers = sum(
        isinstance(block.moe, MemoryLeanPhysicalPaddedGroupedSparseMoE)
        for block in model.blocks
    )
    integrated_worlds = sum(
        isinstance(block.world, ProductionScanPersistentWorldState)
        for block in model.blocks
    )
    status = integration_status()

    if actual != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"integrated 100M parameter drift: {actual}")
    if grouped_layers != len(model.blocks):
        raise RuntimeError("not every MoE block is the memory-lean grouped-v4 backend")
    if integrated_worlds != len(model.blocks):
        raise RuntimeError("not every recurrent world uses the integrated scan class")
    if status.get("systems_variant") != INTEGRATED_SYSTEMS_VARIANT:
        raise RuntimeError("integrated systems-variant drift")
    if status.get("cuda_backend") != PRODUCTION_SCAN_BACKEND:
        raise RuntimeError("integrated scan backend drift")

    result = {
        "status": "PASS",
        "classification": "ZERO_GPU_DATA_AND_INTEGRATED_MODEL_GATE_V5",
        "source_sha": source_sha,
        "engineering_seed_reserved": ENGINEERING_SEED,
        "gpu_allocated": False,
        "data": data,
        "production_model": {
            "parameter_count": actual,
            "grouped_moe_layers": grouped_layers,
            "integrated_worlds": integrated_worlds,
            "production_moe_backend": PRODUCTION_MOE_BACKEND,
            "base_grouped_systems_variant": BASE_GROUPED_SYSTEMS_VARIANT,
            "integrated_systems_variant": INTEGRATED_SYSTEMS_VARIANT,
            "production_scan_backend": PRODUCTION_SCAN_BACKEND,
        },
        "budget_gate": {
            "calibration_steps": CALIBRATION_STEPS,
            "measured_tokens": CALIBRATION_MEASURED_TOKENS,
            "projection_overhead_multiplier": PROJECTION_OVERHEAD_MULTIPLIER,
            "max_projected_full_seconds": MAX_PROJECTED_FULL_SECONDS,
            "max_preflight_vram_gib": MAX_PREFLIGHT_VRAM_GIB,
            "hard_full_timeout_seconds": HARD_FULL_TIMEOUT_SECONDS,
        },
        "full_run_authorized": False,
        "scientific_claim_authorized": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    volume.commit()
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/vol": volume},
)
def h100_preflight(source_sha: str) -> str:
    """One bounded engineering preflight of the integrated production graph only."""

    import torch
    from architectures.cortex_s.affine_scan_triton_candidate import triton_available
    from architectures.cortex_s.production_scan_integration_v1 import _should_use_triton

    _validate_seed_contract()
    volume.reload()
    root = Path(PREFLIGHT_ROOT)
    data_path = root / "data.json"
    dispatch_path = root / "H100_PREFLIGHT_DISPATCHED.json"
    result_path = root / "h100.json"
    if not data_path.exists():
        raise RuntimeError("v5 zero-GPU gate is missing")
    zero_gpu = json.loads(data_path.read_text(encoding="utf-8"))
    if zero_gpu.get("status") != "PASS" or zero_gpu.get("source_sha") != source_sha:
        raise RuntimeError("zero-GPU gate is not valid for this source SHA")
    if dispatch_path.exists() or result_path.exists():
        raise RuntimeError("v5 H100 preflight namespace is already consumed")
    if not torch.cuda.is_available():
        raise RuntimeError("v5 production preflight requires CUDA")
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("v5 production preflight requires SM90/H100-class CUDA")
    if not triton_available():
        raise RuntimeError("Triton is unavailable on the H100 image")
    probe = torch.empty((1, 1, 1), device="cuda", dtype=torch.bfloat16)
    if not _should_use_triton(probe):
        raise RuntimeError("integrated production dispatcher would not select Triton")
    del probe

    dispatch = {
        "source_sha": source_sha,
        "engineering_seed": ENGINEERING_SEED,
        "phase": PHASE,
        "production_moe_backend": PRODUCTION_MOE_BACKEND,
        "integrated_systems_variant": INTEGRATED_SYSTEMS_VARIANT,
        "production_scan_backend": PRODUCTION_SCAN_BACKEND,
        "dispatched_unix": time.time(),
        "device_name": torch.cuda.get_device_name(),
        "cuda_capability": list(torch.cuda.get_device_capability()),
    }
    dispatch_path.write_text(json.dumps(dispatch, indent=2, sort_keys=True), encoding="utf-8")
    volume.commit()

    try:
        with _integrated_calibration_context() as training_module:
            result = training_module.run_h100_calibration(
                data_dir=DATA_DIR,
                output_path=str(result_path),
                source_sha=source_sha,
            )
        result = dict(result)

        if int(result.get("calibration_seed", -1)) != ENGINEERING_SEED:
            raise RuntimeError("calibration returned the wrong engineering seed")
        if int(result.get("measured_steps", -1)) != CALIBRATION_STEPS:
            raise RuntimeError("calibration step-count drift")
        if int(result.get("measured_tokens", -1)) != CALIBRATION_MEASURED_TOKENS:
            raise RuntimeError("calibration measured-token drift")
        if result.get("scientific_status") != "ENGINEERING_PREFLIGHT_ONLY":
            raise RuntimeError("preflight classification drift")
        if result.get("execution") == "compiled" and result.get("compile_mode") != "max-autotune-no-cudagraphs":
            raise RuntimeError("compiled preflight mode drift")

        measured_tps = float(result["training_tokens_per_second"])
        compile_seconds = float(result["compile_seconds"])
        expected_projection = TRAIN_TOKENS / measured_tps * PROJECTION_OVERHEAD_MULTIPLIER + compile_seconds
        if not math.isclose(
            float(result["projected_full_seconds"]),
            expected_projection,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise RuntimeError("full-envelope projection formula drift")
        gates = dict(result.get("gates", {}))
        if gates.get("peak_vram_le_70_gib") != (float(result["peak_vram_gb"]) <= MAX_PREFLIGHT_VRAM_GIB):
            raise RuntimeError("VRAM gate drift")
        if gates.get("projected_full_seconds_le_budget") != (expected_projection <= MAX_PROJECTED_FULL_SECONDS):
            raise RuntimeError("projected-full-time gate drift")

        budget_pass = result.get("status") == "PASS" and all(bool(v) for v in gates.values())
        result.update(
            {
                "classification": "ENGINEERING_INTEGRATED_PRODUCTION_PREFLIGHT_V5",
                "engineering_seed": ENGINEERING_SEED,
                "production_moe_backend": PRODUCTION_MOE_BACKEND,
                "base_grouped_systems_variant": BASE_GROUPED_SYSTEMS_VARIANT,
                "integrated_systems_variant": INTEGRATED_SYSTEMS_VARIANT,
                "production_scan_backend": PRODUCTION_SCAN_BACKEND,
                "triton_dispatch_probe_pass": True,
                "device_name": torch.cuda.get_device_name(),
                "cuda_capability": list(torch.cuda.get_device_capability()),
                "paired_run_eligible": budget_pass,
                # Deliberately false: PASS reaches a later explicit prereg/auth
                # boundary and cannot feed the legacy full_2b entrypoint directly.
                "full_run_authorized": False,
                "scientific_claim_authorized": False,
                "seed_8100_used": False,
            }
        )
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        volume.commit()
        return json.dumps(result, sort_keys=True)
    except Exception:
        # Preserve the dispatch marker and any result emitted by the frozen
        # calibration harness. This attempt is consumed even on infrastructure or
        # runtime failure and must never be retried in the same namespace.
        volume.commit()
        raise


@app.local_entrypoint()
def main(phase: str, source_sha: str) -> None:
    normalized = phase.strip().lower()
    source = source_sha.strip().lower()
    if normalized != PHASE:
        raise ValueError(f"phase must be exactly {PHASE}")
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("source_sha must be a full lowercase commit SHA")

    zero_gpu = json.loads(verify_data_zero_gpu.remote(source))
    print(json.dumps({"zero_gpu": zero_gpu}, indent=2, sort_keys=True), flush=True)
    if zero_gpu.get("status") != "PASS":
        raise RuntimeError("zero-GPU integrated model gate did not pass")

    result = json.loads(h100_preflight.remote(source))
    print(json.dumps({"h100": result}, indent=2, sort_keys=True), flush=True)
