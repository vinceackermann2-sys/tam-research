from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any

import modal

# ENGINEERING SYSTEMS PREFLIGHT ONLY.
# Success reaches only a later explicit full-run authorization boundary.
PHASE = "preflight-v5-triton-scan"
TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v5-triton-scan]"
ENGINEERING_SEED = 2_026_091_002
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v5-triton-scan"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

TRAIN_CORPUS_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_CORPUS_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_CORPUS_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"
TRAIN_IMPLEMENTATION_SHA256 = "6ded33f5981ab2bad6aad75b6fc238a4608960fd9ac839f1f3fa71924ad58844"

# Exact frozen issue #831 production-calibration contract.
EXPECTED_CORTEX_PARAMS = 101_778_112
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
GLOBAL_BATCH = 128
TOKENS_PER_STEP = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
CALIBRATION_WARMUP_STEPS = 5
MEASURED_STEPS = 40
MEASURED_TOKENS = MEASURED_STEPS * TOKENS_PER_STEP
TRAIN_TOKENS = 2_000_000_000
PROJECTION_OVERHEAD_MULTIPLIER = 1.10
MAX_PROJECTED_FULL_SECONDS = 8_500.0
MAX_PREFLIGHT_VRAM_GIB = 70.0
HARD_FULL_TIMEOUT_SECONDS = 10_000

EXPECTED_LAYERS = 24
EXPECTED_D_MODEL = 512
EXPECTED_STATE_DIM = 128
EXPECTED_NUM_EXPERTS = 8
EXPECTED_TOP_K = 2
EXPECTED_LOGICAL_HIDDEN = 338
EXPECTED_PHYSICAL_HIDDEN = 344
EXPECTED_ATTENTION_EVERY = 6
EXPECTED_ATTENTION_LAYERS = 4
EXPECTED_MAX_CONTEXT = 1024
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
        raise ValueError(f"{name} must be a full lowercase commit/tree SHA")
    return normalized


def _validate_payload(
    source_sha: str, source_tree: str, harness_sha: str
) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(
        FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS
    )
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("v5 engineering seed is consumed or scientifically reserved")
    if TOKENS_PER_STEP != 65_536:
        raise RuntimeError("tokens-per-step drift")
    if MEASURED_TOKENS != 2_621_440:
        raise RuntimeError("40-step measured-token contract drift")
    if GLOBAL_BATCH != MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS:
        raise RuntimeError("global-batch drift")
    return source, tree, harness


def _training_implementation_path() -> Path:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    source = inspect.getsourcefile(training_module)
    if not source:
        raise RuntimeError("cannot locate frozen training implementation")
    return Path(source)


def _verify_frozen_data_and_training_file() -> dict[str, Any]:
    root = Path(DATA_DIR)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    meta_path = root / "meta.json"
    missing = [
        str(path) for path in (train_path, val_path, meta_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"frozen 100M/2B corpus missing: {missing}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "assembly_version": 3,
        "train_tokens": 2_000_000_000,
        "val_tokens": 5_000_000,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"data metadata mismatch for {key}: {meta.get(key)!r} != {expected!r}"
            )
    if train_path.stat().st_size != 4_000_000_000:
        raise RuntimeError("train.bin is not exactly 2B uint16 tokens")
    if val_path.stat().st_size != 10_000_000:
        raise RuntimeError("val.bin is not exactly 5M uint16 tokens")

    train_hash = _sha256(train_path)
    val_hash = _sha256(val_path)
    meta_hash = _sha256(meta_path)
    implementation_path = _training_implementation_path()
    implementation_hash = _sha256(implementation_path)
    if train_hash != TRAIN_CORPUS_SHA256:
        raise RuntimeError(f"train corpus fingerprint mismatch: {train_hash}")
    if val_hash != VAL_CORPUS_SHA256:
        raise RuntimeError(f"validation corpus fingerprint mismatch: {val_hash}")
    if meta_hash != META_CORPUS_SHA256:
        raise RuntimeError(f"metadata fingerprint mismatch: {meta_hash}")
    if implementation_hash != TRAIN_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            f"training implementation fingerprint mismatch: {implementation_hash}"
        )
    return {
        "train_sha256": train_hash,
        "val_sha256": val_hash,
        "meta_sha256": meta_hash,
        "training_implementation_sha256": implementation_hash,
        "training_implementation_path": str(implementation_path),
        "metadata": meta,
    }


def _model_invariants(model: Any) -> dict[str, Any]:
    from architectures.cortex_s.grouped_moe_v4 import (
        MemoryLeanPhysicalPaddedGroupedSparseMoE,
    )
    from architectures.cortex_s.language_model import parameter_count
    from architectures.cortex_s.production_scan_integration_v1 import (
        ProductionScanPersistentWorldState,
        integration_status,
    )

    cfg = model.cfg
    blocks = list(model.blocks)
    actual_params = parameter_count(model)
    grouped_layers = sum(
        isinstance(block.moe, MemoryLeanPhysicalPaddedGroupedSparseMoE)
        for block in blocks
    )
    integrated_worlds = sum(
        isinstance(block.world, ProductionScanPersistentWorldState) for block in blocks
    )
    attention_layers = sum(bool(block.has_attention) for block in blocks)
    logical_hidden = {int(block.moe.hidden) for block in blocks}
    physical_hidden = {int(block.moe.physical_hidden) for block in blocks}
    top_k = {int(block.moe.top_k) for block in blocks}
    num_experts = {int(block.moe.num_experts) for block in blocks}
    status = integration_status()

    expected_cfg = {
        "d_model": EXPECTED_D_MODEL,
        "n_layers": EXPECTED_LAYERS,
        "n_heads": 16,
        "max_seq_len": EXPECTED_MAX_CONTEXT,
        "state_size": EXPECTED_STATE_DIM,
        "num_experts": EXPECTED_NUM_EXPERTS,
        "top_k": EXPECTED_TOP_K,
        "expert_hidden": EXPECTED_LOGICAL_HIDDEN,
        "attention_every": EXPECTED_ATTENTION_EVERY,
    }
    for key, expected in expected_cfg.items():
        if int(getattr(cfg, key)) != expected:
            raise RuntimeError(
                f"architecture drift for {key}: {getattr(cfg, key)!r} != {expected!r}"
            )
    if actual_params != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(
            f"parameter-count drift: {actual_params} != {EXPECTED_CORTEX_PARAMS}"
        )
    if (
        len(blocks) != EXPECTED_LAYERS
        or grouped_layers != EXPECTED_LAYERS
        or integrated_worlds != EXPECTED_LAYERS
    ):
        raise RuntimeError("production layer/backend integration drift")
    if attention_layers != EXPECTED_ATTENTION_LAYERS:
        raise RuntimeError("attention schedule drift")
    if logical_hidden != {EXPECTED_LOGICAL_HIDDEN} or physical_hidden != {
        EXPECTED_PHYSICAL_HIDDEN
    }:
        raise RuntimeError("logical/physical grouped expert width drift")
    if top_k != {EXPECTED_TOP_K} or num_experts != {EXPECTED_NUM_EXPERTS}:
        raise RuntimeError("MoE routing-width drift")
    if status.get("systems_variant") != INTEGRATED_SYSTEMS_VARIANT:
        raise RuntimeError("integrated systems variant drift")
    if status.get("cuda_backend") != PRODUCTION_SCAN_BACKEND:
        raise RuntimeError("production scan backend drift")
    return {
        "parameter_count": actual_params,
        "layers": len(blocks),
        "d_model": int(cfg.d_model),
        "state_dim": int(cfg.state_size),
        "num_experts": EXPECTED_NUM_EXPERTS,
        "top_k": EXPECTED_TOP_K,
        "logical_hidden": EXPECTED_LOGICAL_HIDDEN,
        "physical_hidden": EXPECTED_PHYSICAL_HIDDEN,
        "attention_every": EXPECTED_ATTENTION_EVERY,
        "attention_layers": attention_layers,
        "max_context": int(cfg.max_seq_len),
        "grouped_moe_layers": grouped_layers,
        "integrated_worlds": integrated_worlds,
    }


@app.function(
    image=base_image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/vol": volume},
)
def verify_zero_gpu(source_sha: str, source_tree: str, harness_sha: str) -> str:
    """Validate frozen inputs and the integrated model without allocating a GPU."""

    from architectures.cortex_s.production_scan_integration_v1 import (
        build_memory_lean_grouped_triton_scan_cortex_100m,
    )

    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("reserved v5 result namespace already exists; fail closed")

    frozen_inputs = _verify_frozen_data_and_training_file()
    model = build_memory_lean_grouped_triton_scan_cortex_100m()
    invariants = _model_invariants(model)
    payload = {
        "status": "PASS",
        "classification": "ZERO_GPU_SYSTEMS_GATE_V5_ISSUE831_CONTRACT",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed_reserved": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "gpu_allocated": False,
        "frozen_inputs": frozen_inputs,
        "architecture": invariants,
        "frozen_issue831_contract": {
            "warmup_steps": CALIBRATION_WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "measured_tokens": MEASURED_TOKENS,
            "max_vram_gib": MAX_PREFLIGHT_VRAM_GIB,
            "projection_formula": "2_000_000_000 / measured_tps * 1.10 + compile_seconds",
            "max_projected_full_seconds": MAX_PROJECTED_FULL_SECONDS,
        },
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.function(
    image=base_image,
    cpu=1,
    memory=512,
    timeout=5 * 60,
    volumes={"/vol": volume},
)
def reserve_h100_dispatch(
    source_sha: str, source_tree: str, harness_sha: str
) -> str:
    """Consume the single-use engineering attempt before any H100 allocation."""

    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not zero_path.exists():
        raise RuntimeError("zero-GPU systems gate is missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    expected = {
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "status": "PASS",
    }
    for key, value in expected.items():
        if zero.get(key) != value:
            raise RuntimeError(f"zero-GPU systems gate mismatch for {key}")
    if marker_path.exists() or result_path.exists():
        raise RuntimeError("v5 engineering attempt is already consumed")

    marker = {
        "status": "H100_DISPATCH_CONSUMED",
        "classification": "ENGINEERING_SINGLE_ATTEMPT_MARKER_V5_ISSUE831_CONTRACT",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "marked_unix": time.time(),
        "h100_allocation_started": False,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
    }
    _atomic_write(marker_path, marker)
    volume.commit()
    return json.dumps(marker, sort_keys=True)


def _prove_triton_dispatch(model: Any, train_data: Any, device: Any) -> dict[str, Any]:
    """Exercise all integrated worlds and make legacy CUDA fallback fatal."""

    import torch
    import architectures.cortex_s.production_scan_integration_v1 as integration

    original_candidate = integration.affine_scan_triton_candidate
    original_fallback = integration.affine_scan
    calls = {"candidate": 0, "fallback": 0}

    def counted_candidate(a, b, initial=None):
        calls["candidate"] += 1
        return original_candidate(a, b, initial)

    def forbidden_fallback(a, b, initial=None):
        calls["fallback"] += 1
        raise RuntimeError(
            "legacy affine-scan fallback selected during CUDA dispatcher proof"
        )

    integration.affine_scan_triton_candidate = counted_candidate
    integration.affine_scan = forbidden_fallback
    try:
        probe_generator = torch.Generator(device="cpu").manual_seed(
            ENGINEERING_SEED + 90_000
        )
        x, _ = train_data.batch(1, SEQ_LEN, probe_generator, device)
        model.eval()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize(device)
    finally:
        integration.affine_scan_triton_candidate = original_candidate
        integration.affine_scan = original_fallback

    if calls["candidate"] != EXPECTED_LAYERS or calls["fallback"] != 0:
        raise RuntimeError(f"dispatcher proof drift: {calls}")
    return {
        "candidate_calls": calls["candidate"],
        "legacy_fallback_calls": calls["fallback"],
        "expected_integrated_worlds": EXPECTED_LAYERS,
        "selected_triton_for_all_worlds": True,
    }


@app.function(
    image=base_image,
    gpu="H100!",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/vol": volume},
)
def h100_preflight(source_sha: str, source_tree: str, harness_sha: str) -> str:
    """Run the exact frozen #831 40-step production calibration once."""

    import torch
    from tam_research.data import TokenBin
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.affine_scan_triton_candidate import triton_available
    from architectures.cortex_s.production_scan_integration_v1 import (
        build_memory_lean_grouped_triton_scan_cortex_100m,
        memory_lean_grouped_triton_scan_training_builder,
    )

    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    marker_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not marker_path.exists():
        raise RuntimeError("durable H100 dispatch marker missing; refusing GPU work")
    if result_path.exists():
        raise RuntimeError("v5 RESULT.json already exists; refusing rerun")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker = {
        "status": "H100_DISPATCH_CONSUMED",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "h100_allocation_started": False,
    }
    for key, value in expected_marker.items():
        if marker.get(key) != value:
            raise RuntimeError(f"dispatch marker mismatch for {key}")

    if not torch.cuda.is_available():
        raise RuntimeError("v5 systems preflight requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("v5 systems preflight requires H100/SM90-class CUDA")
    if not triton_available():
        raise RuntimeError("Triton is unavailable on the H100 image")
    if _sha256(_training_implementation_path()) != TRAIN_IMPLEMENTATION_SHA256:
        raise RuntimeError("frozen training implementation changed before H100 execution")

    # Prove the merged production dispatcher selects the already-validated Triton scan.
    training_module.seed_all(ENGINEERING_SEED)
    train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
    proof_model = build_memory_lean_grouped_triton_scan_cortex_100m().to(device)
    dispatcher_proof = _prove_triton_dispatch(proof_model, train_data, device)
    del proof_model
    torch.cuda.empty_cache()

    # Reuse the frozen production calibration implementation byte-for-byte.
    # Only the fresh engineering seed and integrated builder are temporarily bound.
    original_calibration_seed = training_module.CALIBRATION_SEED
    if original_calibration_seed != 2_026_090_906:
        raise RuntimeError("frozen v4 calibration-seed binding drifted")
    training_module.CALIBRATION_SEED = ENGINEERING_SEED
    try:
        with memory_lean_grouped_triton_scan_training_builder():
            raw = training_module.run_h100_calibration(
                data_dir=DATA_DIR,
                output_path="/tmp/cortex_s_v5_issue831_calibration.json",
                source_sha=source,
            )
    finally:
        training_module.CALIBRATION_SEED = original_calibration_seed

    if raw.get("calibration_seed") != ENGINEERING_SEED:
        raise RuntimeError("fresh engineering seed was not used")
    if raw.get("measured_steps") != MEASURED_STEPS:
        raise RuntimeError("measured-step contract drift")
    if raw.get("measured_tokens") != MEASURED_TOKENS:
        raise RuntimeError("measured-token contract drift")
    if raw.get("projection_overhead_multiplier") != PROJECTION_OVERHEAD_MULTIPLIER:
        raise RuntimeError("projection-overhead contract drift")

    tps = float(raw["training_tokens_per_second"])
    compile_seconds = float(raw["compile_seconds"])
    projected_expected = (
        TRAIN_TOKENS / tps * PROJECTION_OVERHEAD_MULTIPLIER + compile_seconds
    )
    projected_actual = float(raw["projected_full_seconds"])
    if not math.isclose(
        projected_actual, projected_expected, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise RuntimeError("frozen #831 projection formula drift")
    if float(raw["peak_vram_gb"]) > MAX_PREFLIGHT_VRAM_GIB and raw["gates"].get(
        "peak_vram_le_70_gib"
    ):
        raise RuntimeError("70-GiB gate reporting drift")
    expected_gate = projected_actual <= MAX_PROJECTED_FULL_SECONDS
    if bool(raw["gates"].get("projected_full_seconds_le_budget")) != expected_gate:
        raise RuntimeError("8500-second gate reporting drift")

    calibration_pass = raw.get("status") == "PASS"
    dispatcher_pass = (
        dispatcher_proof["candidate_calls"] == EXPECTED_LAYERS
        and dispatcher_proof["legacy_fallback_calls"] == 0
    )
    status = (
        "SYSTEMS_PREFLIGHT_PASS"
        if calibration_pass and dispatcher_pass
        else "SYSTEMS_PREFLIGHT_FAIL"
    )

    result = {
        "status": status,
        "classification": "ENGINEERING_SYSTEMS_PREFLIGHT_V5_ISSUE831_CONTRACT_ONLY",
        "phase": PHASE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "dispatcher_proof": dispatcher_proof,
        "calibration": {
            "execution": raw["execution"],
            "compile_mode": raw["compile_mode"],
            "compile_seconds": compile_seconds,
            "compile_error": raw["compile_error"],
            "warmup_steps": CALIBRATION_WARMUP_STEPS,
            "measured_steps": raw["measured_steps"],
            "measured_tokens": raw["measured_tokens"],
            "measured_seconds": raw["measured_seconds"],
            "training_tokens_per_second": tps,
            "projected_training_seconds": raw["projected_training_seconds"],
            "projection_overhead_multiplier": raw[
                "projection_overhead_multiplier"
            ],
            "projected_full_seconds": projected_actual,
            "peak_vram_gb": raw["peak_vram_gb"],
            "last_train_loss": raw["last_train_loss"],
            "gates": raw["gates"],
        },
        "frozen_issue831_contract": {
            "measured_steps": MEASURED_STEPS,
            "measured_tokens": MEASURED_TOKENS,
            "amp_behavior": "frozen_training_module_bfloat16_autocast",
            "max_vram_gib": MAX_PREFLIGHT_VRAM_GIB,
            "projection_formula": "2_000_000_000 / measured_tps * 1.10 + compile_seconds",
            "max_projected_full_seconds": MAX_PROJECTED_FULL_SECONDS,
            "hard_full_timeout_seconds": HARD_FULL_TIMEOUT_SECONDS,
        },
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
        "seed_8100_used": False,
        "reserved_scientific_seeds_used": False,
        "note": (
            "PASS reaches only a separate explicit full-run preregistration and "
            "authorization boundary. This preflight never launches or authorizes 2B training."
        ),
    }
    _atomic_write(result_path, result)
    volume.commit()
    return json.dumps(result, sort_keys=True)


@app.local_entrypoint()
def main(phase: str, source_sha: str, source_tree: str, harness_sha: str) -> None:
    if phase.strip().lower() != PHASE:
        raise ValueError(f"phase must be exactly {PHASE}")
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)

    zero = json.loads(verify_zero_gpu.remote(source, tree, harness))
    print(json.dumps({"zero_gpu": zero}, indent=2, sort_keys=True), flush=True)
    if zero.get("status") != "PASS":
        raise RuntimeError("zero-GPU systems gate did not pass")

    reservation = json.loads(reserve_h100_dispatch.remote(source, tree, harness))
    print(
        json.dumps({"dispatch_reservation": reservation}, indent=2, sort_keys=True),
        flush=True,
    )
    if reservation.get("status") != "H100_DISPATCH_CONSUMED":
        raise RuntimeError("H100 dispatch reservation was not durably committed")

    result = json.loads(h100_preflight.remote(source, tree, harness))
    print(json.dumps({"h100": result}, indent=2, sort_keys=True), flush=True)
