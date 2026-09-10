from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any

import modal

# ENGINEERING SYSTEMS PREFLIGHT ONLY. This file exposes no full-2B entrypoint.
# A future issue-triggered invocation is single-use. The engineering seed becomes
# consumed only after reserve_h100_dispatch durably commits the marker below;
# H100 allocation happens strictly afterwards in the local entrypoint.
PHASE = "preflight-v5-triton-scan"
TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v5-triton-scan]"
ENGINEERING_SEED = 2_026_091_002
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v5-triton-scan"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

# Frozen corpus identities from the established CPU fingerprint evidence.
TRAIN_CORPUS_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_CORPUS_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_CORPUS_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"
TRAIN_IMPLEMENTATION_SHA256 = "6ded33f5981ab2bad6aad75b6fc238a4608960fd9ac839f1f3fa71924ad58844"

# Frozen workload / fairness contract from preregistration issue #831.
EXPECTED_CORTEX_PARAMS = 101_778_112
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
GLOBAL_BATCH = 128
TOKENS_PER_STEP = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
TOTAL_OPTIMIZER_STEPS = 30_518
FULL_BATCH_TOKEN_EXPOSURES = 2_000_027_648
MEASURED_STEPS = 80
FIRST_WINDOW_STEPS = (1, 40)
STEADY_WINDOW_STEPS = (41, 80)
STEADY_START_INDEX = 40
STEADY_STEPS = 40
STEADY_TOKENS = STEADY_STEPS * TOKENS_PER_STEP
STARTUP_ALLOWANCE_SECONDS = 1_144.16
MIN_STEADY_TOKENS_PER_SECOND = 259_750.0
MAX_FULL_ENVELOPE_SECONDS = 8_500.0
MAX_ALLOCATED_VRAM_GIB = 80.0

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
PRODUCTION_MOE_BACKEND = "physical_padded_grouped_bf16"
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


def _validate_payload(source_sha: str, source_tree: str, harness_sha: str) -> tuple[str, str, str]:
    source = _full_sha(source_sha, "source_sha")
    tree = _full_sha(source_tree, "source_tree")
    harness = _full_sha(harness_sha, "harness_sha")
    forbidden = set(CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("v5 engineering seed is consumed or scientifically reserved")
    if TOKENS_PER_STEP != 65_536:
        raise RuntimeError("tokens-per-step drift")
    if GLOBAL_BATCH != MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS:
        raise RuntimeError("global-batch drift")
    if TOTAL_OPTIMIZER_STEPS * TOKENS_PER_STEP != FULL_BATCH_TOKEN_EXPOSURES:
        raise RuntimeError("literal exposure drift")
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
        raise RuntimeError(f"training implementation fingerprint mismatch: {implementation_hash}")
    return {
        "train_sha256": train_hash,
        "val_sha256": val_hash,
        "meta_sha256": meta_hash,
        "training_implementation_sha256": implementation_hash,
        "training_implementation_path": str(implementation_path),
        "metadata": meta,
    }


def _model_invariants(model: Any) -> dict[str, Any]:
    from architectures.cortex_s.grouped_moe_v4 import MemoryLeanPhysicalPaddedGroupedSparseMoE
    from architectures.cortex_s.language_model import parameter_count
    from architectures.cortex_s.production_scan_integration_v1 import (
        ProductionScanPersistentWorldState,
        integration_status,
    )

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
            raise RuntimeError(f"architecture drift for {key}: {getattr(cfg, key)!r} != {expected!r}")
    if actual_params != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(f"parameter-count drift: {actual_params} != {EXPECTED_CORTEX_PARAMS}")
    if len(blocks) != EXPECTED_LAYERS or grouped_layers != EXPECTED_LAYERS or integrated_worlds != EXPECTED_LAYERS:
        raise RuntimeError("production layer/backend integration drift")
    if attention_layers != EXPECTED_ATTENTION_LAYERS:
        raise RuntimeError("attention schedule drift")
    if logical_hidden != {EXPECTED_LOGICAL_HIDDEN} or physical_hidden != {EXPECTED_PHYSICAL_HIDDEN}:
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


@app.function(image=base_image, cpu=4, memory=8192, timeout=30 * 60, volumes={"/vol": volume})
def verify_zero_gpu(source_sha: str, source_tree: str, harness_sha: str) -> str:
    """Validate source-independent frozen inputs and model invariants without a GPU."""

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
        "classification": "ZERO_GPU_SYSTEMS_GATE_V5",
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
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
    }
    _atomic_write(root / "ZERO_GPU_GATE.json", payload)
    volume.commit()
    return json.dumps(payload, sort_keys=True)


@app.function(image=base_image, cpu=1, memory=512, timeout=5 * 60, volumes={"/vol": volume})
def reserve_h100_dispatch(source_sha: str, source_tree: str, harness_sha: str) -> str:
    """Consume the single-use engineering attempt durably before H100 allocation."""

    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not zero_path.exists():
        raise RuntimeError("zero-GPU systems gate is missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    expected = {"source_sha": source, "source_tree": tree, "harness_sha": harness, "status": "PASS"}
    for key, value in expected.items():
        if zero.get(key) != value:
            raise RuntimeError(f"zero-GPU systems gate mismatch for {key}")
    if marker_path.exists() or result_path.exists():
        raise RuntimeError("v5 engineering attempt is already consumed")

    marker = {
        "status": "H100_DISPATCH_CONSUMED",
        "classification": "ENGINEERING_SINGLE_ATTEMPT_MARKER_V5",
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


def _one_optimizer_step_no_amp(
    *,
    model: Any,
    runner: Any,
    optimizer: Any,
    train_data: Any,
    generator: Any,
    device: Any,
    lr: float,
) -> float:
    import torch
    import torch.nn.functional as F

    model.train()
    runner.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for _ in range(GRAD_ACCUM_STEPS):
        x, y = train_data.batch(MICRO_BATCH_SIZE, SEQ_LEN, generator, device)
        logits = runner(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)) / GRAD_ACCUM_STEPS
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CORTEX-S training loss")
        loss.backward()
        running += float(loss.detach())
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite CORTEX-S gradient norm")
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    return running


def _prove_triton_dispatch(model: Any, train_data: Any, device: Any) -> dict[str, Any]:
    """Exercise all 24 integrated worlds while making legacy fallback fatal."""

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
        raise RuntimeError("legacy affine-scan fallback selected during CUDA dispatcher proof")

    integration.affine_scan_triton_candidate = counted_candidate
    integration.affine_scan = forbidden_fallback
    try:
        probe_generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 90_000)
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


@app.function(image=base_image, gpu="H100!", cpu=8, memory=32768, timeout=30 * 60, volumes={"/vol": volume})
def h100_preflight(source_sha: str, source_tree: str, harness_sha: str) -> str:
    """Run exactly one bounded 80-step systems timing audit after durable reservation."""

    import torch
    from tam_research.data import TokenBin
    from tam_research.train import cosine_lr
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.affine_scan_triton_candidate import triton_available
    from architectures.cortex_s.production_scan_integration_v1 import (
        build_memory_lean_grouped_triton_scan_cortex_100m,
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

    torch.set_float32_matmul_precision("high")
    training_module.seed_all(ENGINEERING_SEED)
    train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))

    # Verify the actual guarded dispatcher on the integrated production graph.
    proof_model = build_memory_lean_grouped_triton_scan_cortex_100m().to(device)
    dispatcher_proof = _prove_triton_dispatch(proof_model, train_data, device)
    del proof_model
    torch.cuda.empty_cache()

    # Compile/warm a disposable copy without advancing the measured generator or
    # mutating the measured model. No autocast/AMP is used anywhere in this audit.
    startup_started = time.perf_counter()
    training_module.seed_all(ENGINEERING_SEED)
    warm_model = build_memory_lean_grouped_triton_scan_cortex_100m().to(device)
    warm_optimizer = training_module._make_optimizer(warm_model)
    warm_runner = training_module._compile_model(warm_model)
    warm_generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 99_999)
    _one_optimizer_step_no_amp(
        model=warm_model,
        runner=warm_runner,
        optimizer=warm_optimizer,
        train_data=train_data,
        generator=warm_generator,
        device=device,
        lr=training_module.LEARNING_RATE,
    )
    torch.cuda.synchronize(device)
    compile_and_warm_seconds = max(time.perf_counter() - startup_started, 0.0)
    del warm_runner, warm_optimizer, warm_model
    torch.cuda.empty_cache()

    # Rebuild deterministic measured state after compilation warmup.
    measured_setup_started = time.perf_counter()
    training_module.seed_all(ENGINEERING_SEED)
    model = build_memory_lean_grouped_triton_scan_cortex_100m().to(device)
    invariants = _model_invariants(model)
    optimizer = training_module._make_optimizer(model)
    runner = training_module._compile_model(model)
    generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 10_000)
    torch.cuda.synchronize(device)
    measured_setup_seconds = max(time.perf_counter() - measured_setup_started, 0.0)
    startup_seconds_before_step_1 = compile_and_warm_seconds + measured_setup_seconds

    total_steps = TOTAL_OPTIMIZER_STEPS
    warmup_steps = max(1, int(total_steps * training_module.WARMUP_RATIO))
    step_seconds: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)

    for step_index in range(MEASURED_STEPS):
        lr = cosine_lr(step_index, total_steps, warmup_steps, training_module.LEARNING_RATE)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss = _one_optimizer_step_no_amp(
            model=model,
            runner=runner,
            optimizer=optimizer,
            train_data=train_data,
            generator=generator,
            device=device,
            lr=lr,
        )
        torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - started, 1e-9)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at measured optimizer step {step_index + 1}")
        step_seconds.append(elapsed)
        losses.append(loss)

    if len(step_seconds) != MEASURED_STEPS or len(losses) != MEASURED_STEPS:
        raise RuntimeError("measured step-count drift")
    first_seconds = sum(step_seconds[:STEADY_START_INDEX])
    steady_seconds = sum(step_seconds[STEADY_START_INDEX:])
    first_tps = STEADY_TOKENS / max(first_seconds, 1e-9)
    steady_tps = STEADY_TOKENS / max(steady_seconds, 1e-9)
    projected_training_seconds = FULL_BATCH_TOKEN_EXPOSURES / steady_tps
    projected_full_seconds = projected_training_seconds + STARTUP_ALLOWANCE_SECONDS
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)

    gates = {
        "parameter_count_exact": invariants["parameter_count"] == EXPECTED_CORTEX_PARAMS,
        "architecture_invariants_unchanged": True,
        "workload_invariants_unchanged": (
            TOKENS_PER_STEP == 65_536
            and TOTAL_OPTIMIZER_STEPS == 30_518
            and FULL_BATCH_TOKEN_EXPOSURES == 2_000_027_648
            and MEASURED_STEPS == 80
        ),
        "triton_dispatcher_selected_no_legacy_fallback": (
            dispatcher_proof["candidate_calls"] == EXPECTED_LAYERS
            and dispatcher_proof["legacy_fallback_calls"] == 0
        ),
        "finite_loss_all_80_steps": all(math.isfinite(value) for value in losses),
        "peak_allocated_vram_lt_80_gib": peak_allocated_gib < MAX_ALLOCATED_VRAM_GIB,
        "steady_state_tps_ge_259750": steady_tps >= MIN_STEADY_TOKENS_PER_SECOND,
        "projected_full_envelope_le_8500_seconds": projected_full_seconds <= MAX_FULL_ENVELOPE_SECONDS,
    }
    status = "SYSTEMS_PREFLIGHT_PASS" if all(gates.values()) else "SYSTEMS_PREFLIGHT_FAIL"
    result = {
        "status": status,
        "classification": "ENGINEERING_SYSTEMS_PREFLIGHT_V5_ONLY",
        "phase": PHASE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "software": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "triton_available": triton_available(),
        },
        "architecture": invariants,
        "workload": {
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "global_batch": GLOBAL_BATCH,
            "tokens_per_optimizer_step": TOKENS_PER_STEP,
            "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
            "literal_exposures": FULL_BATCH_TOKEN_EXPOSURES,
            "measured_steps": MEASURED_STEPS,
            "first_window_steps": list(FIRST_WINDOW_STEPS),
            "steady_window_steps": list(STEADY_WINDOW_STEPS),
            "dropout": 0.0,
            "amp_enabled": False,
        },
        "dispatcher_proof": dispatcher_proof,
        "startup": {
            "compile_and_warm_seconds": compile_and_warm_seconds,
            "measured_setup_seconds": measured_setup_seconds,
            "startup_seconds_before_step_1": startup_seconds_before_step_1,
            "frozen_projection_allowance_seconds": STARTUP_ALLOWANCE_SECONDS,
        },
        "timing": {
            "step_seconds_1_to_80": step_seconds,
            "first_window_seconds": first_seconds,
            "first_window_tokens_per_second": first_tps,
            "steady_window_seconds": steady_seconds,
            "steady_window_tokens": STEADY_TOKENS,
            "steady_state_tokens_per_second": steady_tps,
            "projected_30518_step_training_seconds": projected_training_seconds,
            "projected_full_envelope_seconds": projected_full_seconds,
        },
        "memory": {
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
        },
        "loss": {
            "all_finite": all(math.isfinite(value) for value in losses),
            "step_losses_1_to_80": losses,
            "last": losses[-1],
        },
        "gates": gates,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
        "seed_8100_used": False,
        "reserved_scientific_seeds_used": False,
        "note": "PASS reaches only a separate explicit authorization boundary; it does not launch or authorize full 2B training.",
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

    # This call is deliberately separate and MUST complete before the first H100
    # function invocation. From this point onward the seed/trigger/root are consumed.
    reservation = json.loads(reserve_h100_dispatch.remote(source, tree, harness))
    print(json.dumps({"dispatch_reservation": reservation}, indent=2, sort_keys=True), flush=True)
    if reservation.get("status") != "H100_DISPATCH_CONSUMED":
        raise RuntimeError("H100 dispatch reservation was not durably committed")

    result = json.loads(h100_preflight.remote(source, tree, harness))
    print(json.dumps({"h100": result}, indent=2, sort_keys=True), flush=True)
