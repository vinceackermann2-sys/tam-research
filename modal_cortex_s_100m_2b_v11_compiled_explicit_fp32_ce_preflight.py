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
# Inert until the exact owner-only issue trigger is merged and separately authorized/opened.
PHASE = "preflight-v11-compiled-explicit-fp32-ce"
TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v11-compiled-explicit-fp32-ce]"
ENGINEERING_SEED = 2_026_091_013
RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v11-compiled-explicit-fp32-ce"
DATA_DIR = "/vol/data/tam100m-2b-curated-v1"

SYSTEMS_VARIANT = "compiled_explicit_fp32_ce_v1"
LOSS_COMPILE_MODE = "max-autotune-no-cudagraphs"
LOSS_FULLGRAPH = True
ACCEPTED_CANDIDATE_BLOB = "970a0dcf130348662f4c9a3ed11d8e6ad250781a"
ACCEPTED_CANDIDATE_TEST_BLOB = "f61abfefbc1959502cdb3a9f9e3330774db2914d"
TRAIN_CORPUS_SHA256 = "93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b"
VAL_CORPUS_SHA256 = "ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760"
META_CORPUS_SHA256 = "14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3"
TRAIN_IMPLEMENTATION_SHA256 = "d0d91de09bba8b7631913b3e9c0407ab2bb0114f7bba133c123ee5598d490461"

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
EXPECTED_ATTENTION_LAYERS = 4
EXPECTED_PHYSICAL_HIDDEN = 344
INTEGRATED_SYSTEMS_VARIANT = "memory_lean_grouped_v4_triton_scan_v1"
PRODUCTION_SCAN_BACKEND = "affine_scan_triton_candidate"

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
)
FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)

APP_NAME = "cortex-s-v0-100m-2b-preflight-v11-compiled-explicit-fp32-ce"
VOLUME_NAME = "tam-research-data"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.10,<2.11",
        "numpy>=2.0,<3",
    )
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
    forbidden = set(RETIRED_OR_CONSUMED_ENGINEERING_SEEDS) | set(FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS)
    if ENGINEERING_SEED in forbidden:
        raise RuntimeError("v11 engineering seed is retired, consumed, or scientifically reserved")
    if TOKENS_PER_STEP != 65_536 or MEASURED_TOKENS != 2_621_440:
        raise RuntimeError("frozen calibration token contract drift")
    if GLOBAL_BATCH != MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS:
        raise RuntimeError("global-batch drift")
    return source, tree, harness


def _training_module_and_path():
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    source = inspect.getsourcefile(training_module)
    if not source:
        raise RuntimeError("cannot locate production training implementation")
    return training_module, Path(source)


def _candidate_module_and_path():
    import architectures.cortex_s.compiled_explicit_fp32_ce_v1 as candidate

    source = inspect.getsourcefile(candidate)
    if not source:
        raise RuntimeError("cannot locate accepted v11 candidate implementation")
    return candidate, Path(source)


def _verify_frozen_inputs() -> dict[str, Any]:
    candidate, candidate_path = _candidate_module_and_path()
    training_module, implementation_path = _training_module_and_path()
    root = Path(DATA_DIR)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    meta_path = root / "meta.json"
    missing = [str(path) for path in (train_path, val_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen corpus missing: {missing}")

    hashes = {
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
        "meta_sha256": _sha256(meta_path),
        "training_implementation_sha256": _sha256(implementation_path),
        "candidate_source_sha256": _sha256(candidate_path),
    }
    expected = {
        "train_sha256": TRAIN_CORPUS_SHA256,
        "val_sha256": VAL_CORPUS_SHA256,
        "meta_sha256": META_CORPUS_SHA256,
        "training_implementation_sha256": TRAIN_IMPLEMENTATION_SHA256,
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise RuntimeError(f"frozen fingerprint mismatch for {key}: {hashes[key]}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, value in {
        "assembly_version": 3,
        "train_tokens": 2_000_000_000,
        "val_tokens": 5_000_000,
        "seed": 8100,
        "tokenizer": "gpt2",
        "dtype": "uint16",
    }.items():
        if meta.get(key) != value:
            raise RuntimeError(f"data metadata mismatch for {key}")
    if train_path.stat().st_size != 4_000_000_000 or val_path.stat().st_size != 10_000_000:
        raise RuntimeError("frozen corpus byte-size drift")

    status = candidate.integration_status()
    required_status = {
        "systems_variant": SYSTEMS_VARIANT,
        "classification": "ZERO_CREDIT_SYSTEMS_OPTIMIZATION_ONLY",
        "loss_compile_mode": LOSS_COMPILE_MODE,
        "loss_fullgraph": LOSS_FULLGRAPH,
        "explicit_full_logits_fp32_cast_preserved": True,
        "cross_entropy_semantics_preserved": True,
        "production_model_compile_unchanged": True,
        "default_model_forward_unchanged": True,
        "default_training_module_unchanged": True,
        "single_host_readback_preserved": True,
        "liger_used": False,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
        "gpu_dispatch_authorized": False,
    }
    for key, value in required_status.items():
        if status.get(key) != value:
            raise RuntimeError(f"v11 candidate contract drift for {key}: {status.get(key)!r}")

    module_source = candidate_path.read_text(encoding="utf-8")
    loss_source = inspect.getsource(candidate.explicit_fp32_cross_entropy)
    step_source = inspect.getsource(candidate._one_optimizer_step_compiled_explicit_fp32_ce)
    if "logits.float().reshape(-1, logits.size(-1))" not in loss_source:
        raise RuntimeError("v11 explicit-FP32 logits cast drift")
    if "targets.reshape(-1)" not in loss_source or "F.cross_entropy(" not in loss_source:
        raise RuntimeError("v11 explicit-FP32 CE formula drift")
    if "compiled_explicit_fp32_cross_entropy = torch.compile(" not in module_source:
        raise RuntimeError("v11 loss-only torch.compile seam missing")
    if "mode=LOSS_COMPILE_MODE" not in module_source or "fullgraph=LOSS_FULLGRAPH" not in module_source:
        raise RuntimeError("v11 loss-only compiler contract drift")
    if "with training_module._autocast(device):" not in step_source:
        raise RuntimeError("v11 loss left the production autocast region")
    if "compiled_explicit_fp32_cross_entropy(logits, y)" not in step_source:
        raise RuntimeError("v11 optimizer step is not using compiled explicit-FP32 CE")
    if step_source.count('.to(device="cpu")') != 1:
        raise RuntimeError("v11 optimizer-step host readback count drift")
    if "liger" in module_source.lower():
        raise RuntimeError("v11 unexpectedly references Liger")

    trainer_source = implementation_path.read_text(encoding="utf-8")
    if "return torch.compile(model, mode=COMPILE_MODE, fullgraph=False)" not in trainer_source:
        raise RuntimeError("frozen production model compile changed")
    if "logits.float().reshape(-1, logits.size(-1))" not in trainer_source:
        raise RuntimeError("frozen default production trainer baseline changed")
    if int(training_module.GRAD_ACCUM_STEPS) != GRAD_ACCUM_STEPS:
        raise RuntimeError("production grad-accum contract drift")
    if int(training_module.MICRO_BATCH_SIZE) != MICRO_BATCH_SIZE or int(training_module.SEQ_LEN) != SEQ_LEN:
        raise RuntimeError("production batch/sequence contract drift")

    hashes["metadata"] = meta
    hashes["candidate_status"] = status
    hashes["model_compile_mode"] = training_module.COMPILE_MODE
    hashes["accepted_candidate_blob"] = ACCEPTED_CANDIDATE_BLOB
    hashes["accepted_candidate_test_blob"] = ACCEPTED_CANDIDATE_TEST_BLOB
    return hashes


def _model_invariants(model: Any) -> dict[str, Any]:
    from architectures.cortex_s.grouped_moe_v4 import MemoryLeanPhysicalPaddedGroupedSparseMoE
    from architectures.cortex_s.language_model import parameter_count
    from architectures.cortex_s.production_scan_integration_v1 import ProductionScanPersistentWorldState, integration_status

    blocks = list(model.blocks)
    params = parameter_count(model)
    grouped_layers = sum(isinstance(block.moe, MemoryLeanPhysicalPaddedGroupedSparseMoE) for block in blocks)
    worlds = sum(isinstance(block.world, ProductionScanPersistentWorldState) for block in blocks)
    attention = sum(bool(block.has_attention) for block in blocks)
    if params != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError("parameter-count drift")
    if len(blocks) != EXPECTED_LAYERS or grouped_layers != EXPECTED_LAYERS or worlds != EXPECTED_LAYERS:
        raise RuntimeError("24-layer grouped-v4/Triton-world contract drift")
    if attention != EXPECTED_ATTENTION_LAYERS:
        raise RuntimeError("attention schedule drift")
    if {int(block.moe.physical_hidden) for block in blocks} != {EXPECTED_PHYSICAL_HIDDEN}:
        raise RuntimeError("physical expert width drift")
    status = integration_status()
    if status.get("systems_variant") != INTEGRATED_SYSTEMS_VARIANT or status.get("cuda_backend") != PRODUCTION_SCAN_BACKEND:
        raise RuntimeError("integrated production systems backend drift")
    return {
        "parameter_count": params,
        "layers": len(blocks),
        "grouped_moe_layers": grouped_layers,
        "integrated_worlds": worlds,
        "attention_layers": attention,
        "systems_variant": status.get("systems_variant"),
        "cuda_backend": status.get("cuda_backend"),
    }


@app.function(image=base_image, cpu=4, memory=8192, timeout=30 * 60, volumes={"/vol": volume})
def verify_zero_gpu(source_sha: str, source_tree: str, harness_sha: str) -> str:
    from architectures.cortex_s.production_scan_integration_v1 import build_memory_lean_grouped_triton_scan_cortex_100m

    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    if root.exists():
        raise RuntimeError("reserved v11 result namespace already exists; fail closed")
    frozen_inputs = _verify_frozen_inputs()
    model = build_memory_lean_grouped_triton_scan_cortex_100m()
    architecture = _model_invariants(model)
    payload = {
        "status": "PASS",
        "classification": "ZERO_GPU_COMPILED_EXPLICIT_FP32_CE_PREFLIGHT_V11",
        "phase": PHASE,
        "trigger_title": TRIGGER_TITLE,
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed_reserved": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "gpu_allocated": False,
        "frozen_inputs": frozen_inputs,
        "architecture": architecture,
        "calibration_contract": {
            "warmup_steps": CALIBRATION_WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "measured_tokens": MEASURED_TOKENS,
            "tokens_per_optimizer_step": TOKENS_PER_STEP,
            "bf16_autocast": True,
            "model_compile_mode": "max-autotune-no-cudagraphs",
            "loss_compile_mode": LOSS_COMPILE_MODE,
            "loss_fullgraph": LOSS_FULLGRAPH,
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


@app.function(image=base_image, cpu=1, memory=512, timeout=5 * 60, volumes={"/vol": volume})
def reserve_h100_dispatch(source_sha: str, source_tree: str, harness_sha: str) -> str:
    source, tree, harness = _validate_payload(source_sha, source_tree, harness_sha)
    volume.reload()
    root = Path(RESULT_ROOT)
    zero_path = root / "ZERO_GPU_GATE.json"
    marker_path = root / "H100_DISPATCH_CONSUMED.json"
    result_path = root / "RESULT.json"
    if not zero_path.exists():
        raise RuntimeError("zero-GPU systems gate is missing")
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    for key, expected in {"source_sha": source, "source_tree": tree, "harness_sha": harness, "status": "PASS"}.items():
        if zero.get(key) != expected:
            raise RuntimeError(f"zero-GPU gate mismatch for {key}")
    if marker_path.exists() or result_path.exists():
        raise RuntimeError("v11 engineering attempt is already consumed")
    marker = {
        "status": "H100_DISPATCH_CONSUMED",
        "classification": "ENGINEERING_SINGLE_ATTEMPT_MARKER_V11_COMPILED_EXPLICIT_FP32_CE",
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

    def counted_candidate(a, b, initial=None):
        calls["candidate"] += 1
        return original_candidate(a, b, initial)

    def forbidden_fallback(a, b, initial=None):
        calls["fallback"] += 1
        raise RuntimeError("legacy affine-scan fallback selected during CUDA dispatcher proof")

    integration.affine_scan_triton_candidate = counted_candidate
    integration.affine_scan = forbidden_fallback
    try:
        generator = torch.Generator(device="cpu").manual_seed(ENGINEERING_SEED + 90_000)
        x, _ = train_data.batch(1, SEQ_LEN, generator, device)
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


def _prove_compiled_explicit_fp32_ce_cuda(device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    import architectures.cortex_s.compiled_explicit_fp32_ce_v1 as candidate

    generator = torch.Generator(device="cuda").manual_seed(ENGINEERING_SEED + 91_000)
    base = torch.randn((37, 113), device=device, dtype=torch.bfloat16, generator=generator)
    targets = torch.randint(0, 113, (37,), device=device, generator=generator)
    reference_logits = base.detach().clone().requires_grad_(True)
    candidate_logits = base.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_loss = F.cross_entropy(
            reference_logits.float().reshape(-1, reference_logits.size(-1)),
            targets.reshape(-1),
        )
        candidate_loss = candidate.compiled_explicit_fp32_cross_entropy(candidate_logits, targets)
    reference_loss.backward()
    candidate_loss.backward()
    torch.cuda.synchronize(device)

    if candidate_loss.dtype != torch.float32 or reference_loss.dtype != torch.float32:
        raise RuntimeError(
            f"v11 explicit-FP32 loss dtype drift: candidate={candidate_loss.dtype}, reference={reference_loss.dtype}"
        )
    if not torch.allclose(candidate_loss, reference_loss, rtol=1e-6, atol=1e-6):
        raise RuntimeError("compiled explicit-FP32 CE loss no longer matches reference on CUDA")
    if reference_logits.grad is None or candidate_logits.grad is None:
        raise RuntimeError("compiled explicit-FP32 CE CUDA gradient proof produced missing gradients")
    if not torch.allclose(candidate_logits.grad, reference_logits.grad, rtol=2e-3, atol=2e-3):
        raise RuntimeError("compiled explicit-FP32 CE gradients no longer match reference on CUDA")

    max_grad_abs_diff = float((candidate_logits.grad - reference_logits.grad).abs().max().item())
    return {
        "input_logits_dtype": str(base.dtype),
        "candidate_loss_dtype": str(candidate_loss.dtype),
        "reference_loss_dtype": str(reference_loss.dtype),
        "loss_matches_explicit_fp32_reference": True,
        "gradient_matches_explicit_fp32_reference": True,
        "gradient_rtol": 2e-3,
        "gradient_atol": 2e-3,
        "max_gradient_abs_diff": max_grad_abs_diff,
        "proof_runs_after_timed_calibration": True,
    }


def _raw_calibration_summary(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    keys = (
        "status",
        "execution",
        "compile_mode",
        "compile_seconds",
        "compile_error",
        "measured_steps",
        "measured_tokens",
        "measured_seconds",
        "training_tokens_per_second",
        "projected_training_seconds",
        "projection_overhead_multiplier",
        "projected_full_seconds",
        "peak_vram_gb",
        "last_train_loss",
        "gates",
    )
    return {key: raw.get(key) for key in keys}


@app.function(image=base_image, gpu="H100!", cpu=8, memory=32768, timeout=30 * 60, volumes={"/vol": volume})
def h100_preflight(source_sha: str, source_tree: str, harness_sha: str) -> str:
    import torch
    from tam_research.data import TokenBin
    import architectures.cortex_s.experiments.scale100m_2b.train as training_module
    from architectures.cortex_s.affine_scan_triton_candidate import triton_available
    from architectures.cortex_s.compiled_explicit_fp32_ce_v1 import compiled_explicit_fp32_ce_training_builder
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
        raise RuntimeError("v11 RESULT.json already exists; refusing rerun")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for key, expected in {
        "status": "H100_DISPATCH_CONSUMED",
        "source_sha": source,
        "source_tree": tree,
        "harness_sha": harness,
        "engineering_seed": ENGINEERING_SEED,
        "result_root": RESULT_ROOT,
        "h100_allocation_started": False,
    }.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"dispatch marker mismatch for {key}")

    if not torch.cuda.is_available():
        raise RuntimeError("v11 systems preflight requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("v11 systems preflight requires H100/SM90-class CUDA")
    if not triton_available():
        raise RuntimeError("Triton is unavailable on the H100 image")
    if _sha256(_training_module_and_path()[1]) != TRAIN_IMPLEMENTATION_SHA256:
        raise RuntimeError("production training implementation changed before H100 execution")
    _verify_frozen_inputs()
    if int(training_module.CALIBRATION_WARMUP_STEPS) != CALIBRATION_WARMUP_STEPS:
        raise RuntimeError("warmup-step contract drift")
    if int(training_module.CALIBRATION_STEPS) != MEASURED_STEPS:
        raise RuntimeError("measured-step contract drift")

    marker["h100_allocation_started"] = True
    marker["h100_function_started_unix"] = time.time()
    _atomic_write(marker_path, marker)
    volume.commit()

    dispatcher_proof: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    compiled_ce_cuda_proof: dict[str, Any] | None = None
    try:
        training_module.seed_all(ENGINEERING_SEED)
        train_data = TokenBin(str(Path(DATA_DIR) / "train.bin"))
        proof_model = build_memory_lean_grouped_triton_scan_cortex_100m().to(device)
        dispatcher_proof = _prove_triton_dispatch(proof_model, train_data, device)
        del proof_model
        torch.cuda.empty_cache()

        original_seed = training_module.CALIBRATION_SEED
        training_module.CALIBRATION_SEED = ENGINEERING_SEED
        try:
            with memory_lean_grouped_triton_scan_training_builder():
                with compiled_explicit_fp32_ce_training_builder():
                    raw = training_module.run_h100_calibration(
                        data_dir=DATA_DIR,
                        output_path="/tmp/cortex_s_v11_compiled_explicit_fp32_ce_calibration.json",
                        source_sha=source,
                    )
        finally:
            training_module.CALIBRATION_SEED = original_seed

        if raw.get("execution") != "compiled":
            raise RuntimeError(f"v11 candidate did not remain compiled: {raw.get('compile_error')}")
        if raw.get("compile_error") not in (None, ""):
            raise RuntimeError(f"v11 compiled calibration reported compile error: {raw.get('compile_error')}")
        if raw.get("calibration_seed") != ENGINEERING_SEED:
            raise RuntimeError("fresh engineering seed was not used")
        if raw.get("measured_steps") != MEASURED_STEPS or raw.get("measured_tokens") != MEASURED_TOKENS:
            raise RuntimeError("measured calibration contract drift")
        if raw.get("projection_overhead_multiplier") != PROJECTION_OVERHEAD_MULTIPLIER:
            raise RuntimeError("projection-overhead contract drift")

        tps = float(raw["training_tokens_per_second"])
        compile_seconds = float(raw["compile_seconds"])
        if not math.isfinite(tps) or tps <= 0.0:
            raise RuntimeError("non-positive or non-finite production throughput")
        projected_expected = TRAIN_TOKENS / tps * PROJECTION_OVERHEAD_MULTIPLIER + compile_seconds
        projected_actual = float(raw["projected_full_seconds"])
        if not math.isclose(projected_actual, projected_expected, rel_tol=1e-12, abs_tol=1e-9):
            raise RuntimeError("projection formula drift")
        expected_gate = projected_actual <= MAX_PROJECTED_FULL_SECONDS
        if bool(raw["gates"].get("projected_full_seconds_le_budget")) != expected_gate:
            raise RuntimeError("8500-second gate reporting drift")
        if float(raw["peak_vram_gb"]) > MAX_PREFLIGHT_VRAM_GIB and raw["gates"].get("peak_vram_le_70_gib"):
            raise RuntimeError("70-GiB gate reporting drift")

        # Run independent CUDA semantic/gradient proof only after timed calibration so it cannot warm
        # the loss compiler before compile_seconds and steady-state throughput are measured.
        compiled_ce_cuda_proof = _prove_compiled_explicit_fp32_ce_cuda(device)

        calibration_pass = raw.get("status") == "PASS"
        dispatcher_pass = (
            dispatcher_proof["candidate_calls"] == EXPECTED_LAYERS
            and dispatcher_proof["legacy_fallback_calls"] == 0
        )
        semantics_pass = bool(compiled_ce_cuda_proof.get("loss_matches_explicit_fp32_reference")) and bool(
            compiled_ce_cuda_proof.get("gradient_matches_explicit_fp32_reference")
        )
        status = "SYSTEMS_PREFLIGHT_PASS" if calibration_pass and dispatcher_pass and semantics_pass else "SYSTEMS_PREFLIGHT_FAIL"
        result = {
            "status": status,
            "classification": "ENGINEERING_SYSTEMS_PREFLIGHT_V11_COMPILED_EXPLICIT_FP32_CE_ONLY",
            "phase": PHASE,
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "engineering_seed": ENGINEERING_SEED,
            "result_root": RESULT_ROOT,
            "dispatcher_proof": dispatcher_proof,
            "compiled_explicit_fp32_ce_cuda_proof": compiled_ce_cuda_proof,
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
                "projection_overhead_multiplier": raw["projection_overhead_multiplier"],
                "projected_full_seconds": projected_actual,
                "peak_vram_gb": raw["peak_vram_gb"],
                "last_train_loss": raw["last_train_loss"],
                "gates": raw["gates"],
            },
            "contract": {
                "measured_steps": MEASURED_STEPS,
                "measured_tokens": MEASURED_TOKENS,
                "tokens_per_optimizer_step": TOKENS_PER_STEP,
                "amp_behavior": "production_training_module_bfloat16_autocast",
                "systems_variant": SYSTEMS_VARIANT,
                "model_compile_mode": raw["compile_mode"],
                "loss_compile_mode": LOSS_COMPILE_MODE,
                "loss_fullgraph": LOSS_FULLGRAPH,
                "semantic_proof_after_timed_calibration": True,
                "profiling_inside_timed_calibration": False,
                "max_vram_gib": MAX_PREFLIGHT_VRAM_GIB,
                "projection_formula": "2_000_000_000 / measured_tps * 1.10 + compile_seconds",
                "max_projected_full_seconds": MAX_PROJECTED_FULL_SECONDS,
                "hard_full_timeout_seconds": HARD_FULL_TIMEOUT_SECONDS,
            },
            "full_training_authorized": False,
            "scientific_claim_authorized": False,
            "seed_8100_used": False,
            "reserved_scientific_seeds_used": False,
        }
        _atomic_write(result_path, result)
        volume.commit()
        return json.dumps(result, sort_keys=True)
    except Exception as exc:
        failure = {
            "status": "SYSTEMS_PREFLIGHT_ERROR",
            "classification": "ENGINEERING_SYSTEMS_PREFLIGHT_V11_COMPILED_EXPLICIT_FP32_CE_ERROR",
            "phase": PHASE,
            "source_sha": source,
            "source_tree": tree,
            "harness_sha": harness,
            "engineering_seed": ENGINEERING_SEED,
            "result_root": RESULT_ROOT,
            "error": f"{type(exc).__name__}: {exc}",
            "dispatcher_proof": dispatcher_proof,
            "partial_calibration": _raw_calibration_summary(raw),
            "compiled_explicit_fp32_ce_cuda_proof": compiled_ce_cuda_proof,
            "full_training_authorized": False,
            "scientific_claim_authorized": False,
            "seed_8100_used": False,
            "reserved_scientific_seeds_used": False,
        }
        _atomic_write(result_path, failure)
        volume.commit()
        raise


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
    print(json.dumps({"dispatch_reservation": reservation}, indent=2, sort_keys=True), flush=True)
    if reservation.get("status") != "H100_DISPATCH_CONSUMED":
        raise RuntimeError("H100 dispatch reservation was not durably committed")
    result = json.loads(h100_preflight.remote(source, tree, harness))
    print(json.dumps({"h100": result}, indent=2, sort_keys=True), flush=True)
