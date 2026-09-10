from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Any

import modal

PHASE = "host-sync-profile-v1"
TRIGGER_TITLE = "[modal-cortex-s-100m-host-sync-profile-v1]"
ENGINEERING_SEED = 2_026_091_005
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/host-sync-profile-v1"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

TRAIN_CORPUS_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_CORPUS_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_CORPUS_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"
TRAIN_IMPLEMENTATION_SHA256 = "47c9e327095f99eab921b8780bb968101296b11e62ef13dd22488f2974bfb10f"

EXPECTED_CORTEX_PARAMS = 101_778_112
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
EXPECTED_SEQ_LEN = 512
EXPECTED_MICRO_BATCH_SIZE = 64
EXPECTED_GRAD_ACCUM_STEPS = 2
EXPECTED_COMPILE_MODE = "max-autotune-no-cudagraphs"
EXPECTED_GROUPED_MM_CALLS_PER_OPTIMIZER_STEP = 96

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
)
FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)

APP_NAME = "cortex-s-v0-100m-host-sync-profile-v1"
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


def _validate_payload(source_sha: str, source_tree: str, harness_sha: str) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    forbidden = set(RETIRED_OR_CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("host-sync engineering seed is retired, consumed, or scientifically reserved")
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
    missing = [str(path) for path in (train_path, val_path, meta_path) if not path.exists()]
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
            raise RuntimeError(f"data metadata mismatch for {key}: {meta.get(key)!r} != {expected!r}")
    if train_path.stat().st_size != 4_000_000_000:
        raise RuntimeError("train.bin is not exactly 2B uint16 tokens")
    if val_path.stat().st_size != 10_000_000:
        raise RuntimeError("val.bin is not exactly 5M uint16 tokens")

    observed = {
        "train": _sha256(train_path),
        "val": _sha256(val_path),
        "meta": _sha256(meta_path),
        "training": _sha256(_training_implementation_path()),
    }
    expected = {
        "train": TRAIN_CORPUS_SHA256,
        "val": VAL_CORPUS_SHA256,
        "meta": META_CORPUS_SHA256,
        "training": TRAIN_IMPLEMENTATION_SHA256,
    }
    for name, frozen in expected.items():
        if observed[name] != frozen:
            raise RuntimeError(f"{name} fingerprint mismatch: {observed[name]} != {frozen}")
    return {
        "train_sha256": observed["train"],
        "val_sha256": observed["val"],
        "meta_sha256": observed["meta"],
        "training_implementation_sha256": observed["training"],
        "metadata": meta,
    }


def _model_invariants(model: Any) -> dict[str, Any]:
    from architectures.cortex_s.grouped_moe_v4 import MemoryLeanPhysicalPaddedGroupedSparseMoE
    from architectures.cortex_s.language_model import parameter_count
    from architectures.cortex_s.production_scan_integration_v1 import ProductionScanPersistentWorldState

    cfg = model.cfg
    blocks = list(model.blocks)
    actual_params = parameter_count(model)
    grouped_layers = sum(isinstance(block.moe, MemoryLeanPhysicalPaddedGroupedSparseMoE) for block in blocks)
    integrated_worlds = sum(isinstance(block.world, ProductionScanPersistentWorldState) for block in blocks)
    attention_layers = sum(bool(block.has_attention) for block in blocks)
    logical_hidden = {int(block.moe.hidden) for block in blocks}
    physical_hidden = {int(block.moe.physical_hidden) for block in blocks}
    top_k = {int(block.moe.top_k) for block in blocks}
    num_experts = {int(block.moe.num_experts) for block in blocks}

    expected_cfg = {
        "d_model": EXPECTED_D_MODEL,
        "n_layers": EXPECTED_LAYERS,
        "max_seq_len": EXPECTED_MAX_CONTEXT,
        "state_size": EXPECTED_STATE_DIM,
        "num_experts": EXPECTED_NUM_EXPERTS,
        "top_k": EXPECTED_TOP_K,
        "expert_hidden": EXPECTED_LOGICAL_HIDDEN,
        "attention_every": EXPECTED_ATTENTION_EVERY,
    }
    for key, expected in expected_cfg.items():
        if int(getattr(cfg, key)) != expected:
            raise RuntimeError(f"architecture drift for {key}")
    if actual_params != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"parameter-count drift: {actual_params} != {EXPECTED_CORTEX_PARAMS}")
    if len(blocks) != EXPECTED_LAYERS or grouped_layers != EXPECTED_LAYERS or integrated_worlds != EXPECTED_LAYERS:
        raise RuntimeError("production layer/backend integration drift")
    if attention_layers != EXPECTED_ATTENTION_LAYERS:
        raise RuntimeError("attention schedule drift")
    if logical_hidden != {EXPECTED_LOGICAL_HIDDEN} or physical_hidden != {EXPECTED_PHYSICAL_HIDDEN}:
        raise RuntimeError("grouped expert width drift")
    if top_k != {EXPECTED_TOP_K} or num_experts != {EXPECTED_NUM_EXPERTS}:
        raise RuntimeError("MoE routing-width drift")

    grouped_mm_calls_per_step = grouped_layers * 2 * EXPECTED_GRAD_ACCUM_STEPS
    if grouped_mm_calls_per_step != EXPECTED_GROUPED_MM_CALLS_PER_OPTIMIZER_STEP:
        raise RuntimeError("frozen grouped-mm call arithmetic drift")

    return {
        "parameter_count": actual_params,
        "layers": len(blocks),
        "d_model": int(cfg.d_model),
        "state_dim": int(cfg.state_size),
        "grouped_moe_layers": grouped_layers,
        "integrated_worlds": integrated_worlds,
        "attention_layers": attention_layers,
        "logical_hidden": EXPECTED_LOGICAL_HIDDEN,
        "physical_hidden": EXPECTED_PHYSICAL_HIDDEN,
        "num_experts": EXPECTED_NUM_EXPERTS,
        "top_k": EXPECTED_TOP_K,
        "max_context": int(cfg.max_seq_len),
        "expected_grouped_mm_calls_per_optimizer_step": grouped_mm_calls_per_step,
    }


@app.function(image=base_image, cpu=4, memory=8192, timeout=30 * 60, volumes={"/vol": volume})
def verify_zero_gpu(source_sha: str, source_tree: str, harness_sha: str) -> str:
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.production_scan_integration_v1 import build_memory_lean_grouped_triton_scan_cortex_100m

    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("reserved host-sync attribution namespace already exists; fail closed")

    frozen_inputs = _verify_frozen_data_and_training_file()
    model = build_memory_lean_grouped_triton_scan_cortex_100m()
    invariants = _model_invariants(model)
    if training_module.COMPILE_MODE != EXPECTED_COMPILE_MODE:
        raise RuntimeError("compile mode drift")
    if int(training_module.SEQ_LEN) != EXPECTED_SEQ_LEN:
        raise RuntimeError("sequence length drift")
    if int(training_module.MICRO_BATCH_SIZE) != EXPECTED_MICRO_BATCH_SIZE:
        raise RuntimeError("micro batch drift")
    if int(training_module.GRAD_ACCUM_STEPS) != EXPECTED_GRAD_ACCUM_STEPS:
        raise RuntimeError("grad accumulation drift")

    payload = {
        "status": "PASS",
        "classification": "ZERO_GPU_HOST_SYNC_ATTRIBUTION_V1_GATE",
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
        "training_contract": {
            "compile_mode": training_module.COMPILE_MODE,
            "seq_len": int(training_module.SEQ_LEN),
            "micro_batch_size": int(training_module.MICRO_BATCH_SIZE),
            "grad_accum_steps": int(training_module.GRAD_ACCUM_STEPS),
        },
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.function(image=base_image, cpu=1, memory=512, timeout=5 * 60, volumes={"/vol": volume})
def reserve_h100_dispatch(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not zero_path.exists():
        raise RuntimeError("zero-GPU attribution gate is missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    for key, value in {"source_sha": source, "source_tree": tree, "harness_sha": harness, "status": "PASS"}.items():
        if zero.get(key) != value:
            raise RuntimeError(f"zero-GPU attribution gate mismatch for {key}")
    if marker_path.exists() or result_path.exists():
        raise RuntimeError("host-sync attribution engineering attempt is already consumed")

    marker = {
        "status": "H100_DISPATCH_CONSUMED",
        "classification": "ENGINEERING_SINGLE_ATTEMPT_MARKER_HOST_SYNC_ATTRIBUTION_V1",
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
    import torch
    import architectures.cortex_s.production_scan_integration_v1 as integration

    original_candidate = integration.affine_scan_triton_candidate
    original_fallback = integration.affine_scan
    calls = {"candidate": 0, "fallback": 0}
    was_training = bool(model.training)

    def counted_candidate(a, b, initial=None):
        calls["candidate"] += 1
        return original_candidate(a, b, initial)

    def forbidden_fallback(a, b, initial=None):
        calls["fallback"] += 1
        raise RuntimeError("legacy affine-scan fallback selected during CUDA dispatcher proof")

    integration.affine_scan_triton_candidate = counted_candidate
    integration.affine_scan = forbidden_fallback
    try:
        probe_generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 90_000)
        x, _ = train_data.batch(1, EXPECTED_SEQ_LEN, probe_generator, device)
        model.eval()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize(device)
    finally:
        integration.affine_scan_triton_candidate = original_candidate
        integration.affine_scan = original_fallback
        model.train(was_training)

    if calls["candidate"] != EXPECTED_LAYERS or calls["fallback"] != 0:
        raise RuntimeError(f"dispatcher proof drift: {calls}")
    return {
        "candidate_calls": calls["candidate"],
        "legacy_fallback_calls": calls["fallback"],
        "selected_triton_for_all_worlds": True,
    }


@app.function(image=base_image, gpu="H100!", cpu=8, memory=32768, timeout=30 * 60, volumes={"/vol": volume})
def h100_profile(source_sha: str, source_tree: str, harness_sha: str) -> str:
    import torch
    from tam_research.data import TokenBin
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.affine_scan_triton_candidate import triton_available
    from architectures.cortex_s.host_sync_profile_v1 import run_host_sync_profile
    from architectures.cortex_s.production_scan_integration_v1 import build_memory_lean_grouped_triton_scan_cortex_100m

    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    marker_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not marker_path.exists():
        raise RuntimeError("H100 dispatch marker missing; refusing GPU work")
    if result_path.exists():
        raise RuntimeError("host-sync attribution result already exists; refusing rerun")
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
        raise RuntimeError("H100 attribution profile requires CUDA")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper() or torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError(f"expected H100/SM90, got {gpu_name}")
    if not triton_available():
        raise RuntimeError("Triton is unavailable on the H100 attribution image")
    if _sha256(_training_implementation_path()) != TRAIN_IMPLEMENTATION_SHA256:
        raise RuntimeError("frozen training implementation changed before H100 execution")

    marker["h100_allocation_started"] = True
    marker["gpu_name"] = gpu_name
    marker["allocation_started_unix"] = time.time()
    _atomic_write(marker_path, marker)
    volume.commit()

    torch.set_float32_matmul_precision("high")
    training_module.seed_all(ENGINEERING_SEED)
    train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
    model = build_memory_lean_grouped_triton_scan_cortex_100m().to(device)
    invariants = _model_invariants(model)
    dispatcher = _prove_triton_dispatch(model, train_data, device)
    optimizer = training_module._make_optimizer(model)
    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 10_000)

    compile_start = time.perf_counter()
    compiled_runner = training_module._compile_model(model)
    training_module._one_optimizer_step(
        model=model,
        runner=compiled_runner,
        optimizer=optimizer,
        train_data=train_data,
        generator=generator,
        device=device,
        lr=training_module.LEARNING_RATE,
    )
    torch.cuda.synchronize(device)
    compile_seconds = max(time.perf_counter() - compile_start, 0.0)

    profile = run_host_sync_profile(
        training_module=training_module,
        model=model,
        runner=compiled_runner,
        optimizer=optimizer,
        train_data=train_data,
        generator=generator,
        device=device,
        learning_rate=training_module.LEARNING_RATE,
    )
    attribution = profile["profiler"]["attribution"]

    payload = {
        "status": "PROFILE_COMPLETE",
        "classification": attribution["classification"],
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "gpu_name": gpu_name,
        "compile_mode": training_module.COMPILE_MODE,
        "compile_seconds": compile_seconds,
        "architecture": invariants,
        "dispatcher_proof": dispatcher,
        "profile": profile,
        "seed_8100_used": False,
        "reserved_scientific_seeds_used": False,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
        "note": (
            "Engineering source attribution only. This result cannot authorize a production "
            "optimization, full 2B run, or scientific/breakthrough claim by itself."
        ),
    }
    _atomic_write(result_path, payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.local_entrypoint()
def main(phase: str, source_sha: str, source_tree: str, harness_sha: str) -> None:
    if phase != PHASE:
        raise ValueError(f"phase must be exactly {PHASE}")
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    print(json.dumps({"zero_gpu": json.loads(verify_zero_gpu.remote(source, tree, harness))}, indent=2))
    print(json.dumps({"dispatch_reservation": json.loads(reserve_h100_dispatch.remote(source, tree, harness))}, indent=2))
    print(json.dumps({"h100": json.loads(h100_profile.remote(source, tree, harness))}, indent=2))
