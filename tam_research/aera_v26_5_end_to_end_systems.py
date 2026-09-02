from __future__ import annotations

"""AERA-v26.5 CPU-first integrated systems comparison frozen by issue #501.

This module changes no learned/scientific architecture. It compares two identical
v26 models loaded from the existing seed8471 checkpoint: the default exact Torch
FICEM execution backend versus the already-proven v26.4 Triton READ+WRITE backend.
The matched seed8471 Transformer remains the deployment-throughput comparator.

No GPU run is authorized by this module or by issue #501 itself. A later separate
gate must authorize exactly one L4 attempt after this module is CPU-green/merged.
"""

import hashlib
import math
from pathlib import Path
from typing import Any, Callable

import torch

from . import aera_real_language_v12 as v12
from . import aera_real_language_v25 as v25
from . import aera_v25_post8471_triage as triage
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from .aera_hardware_core_v26 import (
    CoalescedFICEMMemory,
    HardwareAwareAERATextLMV26,
    TorchFICEMReferenceBackend,
    coalesced_runtime_v26_protocol,
)
from .aera_hardware_core_v26_4_ficem_write_triton import (
    TritonFICEMReadWriteBackend,
    fused_ficem_read_write_v26_4_protocol,
)

RESEARCH_ISSUE = 501
SOURCE_MAIN = "148bde16c4995877798a874154f0f18363c406f4"
SOURCE_CHECKPOINT_SEED = 8471
CHECKPOINT_RELATIVE_DIR = "/vol/aera-real-language/v25-dev-seed8471"

V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
READ_PASS_RUN = 33618950619
READ_PASS_JOB = 100211244996
READ_PASS_RESULT_SHA256 = "a3b5a85b1de26a3e76d1908753860c7f6105195f63a2faa18a9bdd62db549dac"
WRITE_PASS_RUN = 33651216734
WRITE_PASS_JOB = 100318422299
WRITE_PASS_RESULT_SHA256 = "64105bb08a65f7d3d55528ed35c5b8e77edc55c2c2ec818765896a6a4d16ea8b"

SYSTEM_BATCH_SIZES: tuple[int, ...] = (8, 64)
SYSTEM_WARMUP_CALLS = 3
SYSTEM_TIMED_CALLS_PER_ROUND = 20
SYSTEM_ROUNDS = 5
BATCH8_MIN_FULL_SPEED_RATIO = 0.25
BATCH64_MIN_FULL_SPEED_RATIO = 1.25
INTEGRATED_ATOL = 1e-2
INTEGRATED_RTOL = 1e-2
EXPECTED_STATE_BYTES = 77_760
EXPECTED_CANDIDATES = 255
EXPECTED_SELECTED_WRITES = 16
EXPECTED_VECTOR_UPDATES = 1
MAX_GPU_SECONDS = 600


def systems_protocol() -> dict[str, Any]:
    return {
        "version": "aera-v26.5-issue501-physically-real-sparse-e2e-systems",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_checkpoint_seed": SOURCE_CHECKPOINT_SEED,
        "checkpoint_use": "read-only routing/system geometry only",
        "checkpoint_relative_dir": CHECKPOINT_RELATIVE_DIR,
        "v26_interface_blob": V26_INTERFACE_BLOB,
        "read_backend_blob": READ_BACKEND_BLOB,
        "write_backend_blob": WRITE_BACKEND_BLOB,
        "stable_reference_blob": STABLE_REFERENCE_BLOB,
        "read_pass_run": READ_PASS_RUN,
        "read_pass_job": READ_PASS_JOB,
        "read_pass_result_sha256": READ_PASS_RESULT_SHA256,
        "write_pass_run": WRITE_PASS_RUN,
        "write_pass_job": WRITE_PASS_JOB,
        "write_pass_result_sha256": WRITE_PASS_RESULT_SHA256,
        "batch_sizes": list(SYSTEM_BATCH_SIZES),
        "random_token_seed_rule": "138471 + 10000 + batch_size",
        "warmup_calls": SYSTEM_WARMUP_CALLS,
        "timed_calls_per_round": SYSTEM_TIMED_CALLS_PER_ROUND,
        "rounds": SYSTEM_ROUNDS,
        "timing_order": "rotated interleaved conditions per issue381",
        "timing_clock": "CUDA events with synchronize before/after",
        "batch8_min_full_speed_ratio": BATCH8_MIN_FULL_SPEED_RATIO,
        "batch64_min_full_speed_ratio": BATCH64_MIN_FULL_SPEED_RATIO,
        "integrated_atol": INTEGRATED_ATOL,
        "integrated_rtol": INTEGRATED_RTOL,
        "persistent_state_bytes_per_session": EXPECTED_STATE_BYTES,
        "production_write_geometry": [
            EXPECTED_SELECTED_WRITES,
            EXPECTED_CANDIDATES,
            EXPECTED_VECTOR_UPDATES,
        ],
        "reference_backend": TorchFICEMReferenceBackend.name,
        "candidate_backend": TritonFICEMReadWriteBackend.name,
        "hard": True,
        "route_mode": "hard_sparse",
        "physically_real_sparse_required": True,
        "dense_masked_sparse_credit": False,
        "gpu": "one NVIDIA L4 only in separately preregistered successor gate",
        "max_gpu_seconds": MAX_GPU_SECONDS,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_write_authorized": False,
        "scientific_seed_consumed": False,
        "independent_replication_credit": False,
        "end_to_end_systems_authorized_by_issue501": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def cpu_contract_preflight() -> dict[str, Any]:
    protocol = systems_protocol()
    if SYSTEM_BATCH_SIZES != (8, 64):
        raise RuntimeError("issue501 batch sizes drifted")
    if (SYSTEM_WARMUP_CALLS, SYSTEM_TIMED_CALLS_PER_ROUND, SYSTEM_ROUNDS) != (3, 20, 5):
        raise RuntimeError("issue501 timing contract drifted from issue381")
    if (BATCH8_MIN_FULL_SPEED_RATIO, BATCH64_MIN_FULL_SPEED_RATIO) != (0.25, 1.25):
        raise RuntimeError("issue501 deployment thresholds drifted from issue381")
    if (INTEGRATED_ATOL, INTEGRATED_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue501 integrated BF16 tolerance drifted")
    if EXPECTED_STATE_BYTES != 77_760:
        raise RuntimeError("issue501 state-byte contract drifted")
    if (EXPECTED_SELECTED_WRITES, EXPECTED_CANDIDATES, EXPECTED_VECTOR_UPDATES) != (16, 255, 1):
        raise RuntimeError("issue501 write geometry drifted")

    v26_protocol = coalesced_runtime_v26_protocol()
    if not v26_protocol["coalesced_optional_state"]:
        raise RuntimeError("issue501 requires coalesced optional-state execution")
    if v26_protocol["persistent_state_bytes_real_language_four_stage_memory_dim50"] != EXPECTED_STATE_BYTES:
        raise RuntimeError("issue501 inherited v26 state bytes drifted")
    if v26_protocol["real_language_selected_writes"] != EXPECTED_SELECTED_WRITES:
        raise RuntimeError("issue501 inherited v26 write budget drifted")

    rw_protocol = fused_ficem_read_write_v26_4_protocol()
    if rw_protocol["backend_name"] != TritonFICEMReadWriteBackend.name:
        raise RuntimeError("issue501 repaired backend identity drifted")
    if rw_protocol["repair5_read_backend_blob"] != READ_BACKEND_BLOB:
        raise RuntimeError("issue501 repair5 READ dependency drifted")
    if rw_protocol["write_threshold_input_dtype_visibility_repair1"] is not True:
        raise RuntimeError("issue501 requires repaired WRITE threshold visibility")
    if rw_protocol["write_tail_triton_launches_target"] != 2:
        raise RuntimeError("issue501 repaired WRITE kernel topology drifted")

    return {
        "protocol": protocol,
        "v26_protocol": v26_protocol,
        "read_write_protocol": rw_protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "checkpoint_loaded": False,
        "training_performed": False,
        "scientific_seed_consumed": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(run_dir: str) -> dict[str, str]:
    root = Path(run_dir)
    paths = {"aera": root / "aera.pt", "transformer": root / "transformer.pt"}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"required seed8471 checkpoint missing: {path}")
    return {name: _sha256(path) for name, path in paths.items()}


def _parameter_versions(model: torch.nn.Module) -> tuple[int, ...]:
    return tuple(int(parameter._version) for parameter in model.parameters())


def _parameter_schema(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(parameter.shape)) for name, parameter in model.named_parameters())


def _build_v26(payload: dict[str, Any], device: torch.device) -> HardwareAwareAERATextLMV26:
    torch.manual_seed(SOURCE_CHECKPOINT_SEED)
    model = HardwareAwareAERATextLMV26(v12.aera_v12_config()).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    model.set_memory_pretraining_mode(False)
    return model


def _install_candidate_backend(model: HardwareAwareAERATextLMV26) -> None:
    for stage in model.stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue501 candidate stage lacks CoalescedFICEMMemory")
        memory._execution_backend = TritonFICEMReadWriteBackend()
        if memory.execution_backend_name != TritonFICEMReadWriteBackend.name:
            raise RuntimeError("issue501 candidate backend installation failed")


def load_models(*, run_dir: str, device: torch.device):
    root = Path(run_dir)
    aera_payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    transformer_payload = torch.load(
        root / "transformer.pt", map_location="cpu", weights_only=False
    )
    if aera_payload.get("seed") != SOURCE_CHECKPOINT_SEED:
        raise RuntimeError("issue501 AERA checkpoint seed mismatch")
    if transformer_payload.get("seed") != SOURCE_CHECKPOINT_SEED:
        raise RuntimeError("issue501 Transformer checkpoint seed mismatch")

    reference = _build_v26(aera_payload, device)
    candidate = _build_v26(aera_payload, device)
    _install_candidate_backend(candidate)

    torch.manual_seed(SOURCE_CHECKPOINT_SEED)
    transformer = v25.build_transformer(device).eval()
    transformer.load_state_dict(transformer_payload["model"], strict=True)

    if _parameter_schema(reference) != _parameter_schema(candidate):
        raise RuntimeError("issue501 reference/candidate learned parameter schema differs")
    if reference.state_dict().keys() != candidate.state_dict().keys():
        raise RuntimeError("issue501 reference/candidate state-dict schema differs")
    for stage in reference.stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue501 reference stage lacks CoalescedFICEMMemory")
        if memory.execution_backend_name != TorchFICEMReferenceBackend.name:
            raise RuntimeError("issue501 reference backend is not exact Torch reference")
    return reference, candidate, transformer


def _model_call(model, tokens: torch.Tensor, *, update_memory: bool = True):
    return triage._model_forward(model, tokens, update_memory=update_memory)


def _transformer_call(transformer, tokens: torch.Tensor):
    with triage._autocast(tokens.device):
        return transformer(tokens)


def _route_signature(output: dict[str, object]) -> tuple[torch.Tensor, ...]:
    routes = output.get("stage_routes")
    if not isinstance(routes, list):
        raise RuntimeError("issue501 output missing stage routes")
    signature: list[torch.Tensor] = []
    for chunk in routes:
        if not isinstance(chunk, list):
            raise RuntimeError("issue501 malformed stage-route chunk")
        for row in chunk:
            if not isinstance(row, dict):
                raise RuntimeError("issue501 malformed stage-route row")
            gate = row.get("stage_route_gate")
            if not isinstance(gate, torch.Tensor):
                raise RuntimeError("issue501 route row missing gate tensor")
            signature.append(gate.detach().ge(0.5).cpu())
    return tuple(signature)


def _optional_executed_fractions(output: dict[str, object]) -> list[float]:
    routes = output.get("stage_routes")
    if not isinstance(routes, list):
        raise RuntimeError("issue501 output missing stage routes")
    fractions: list[float] = []
    for chunk in routes:
        if not isinstance(chunk, list):
            raise RuntimeError("issue501 malformed stage-route chunk")
        for stage_index, row in enumerate(chunk):
            if stage_index == 0:
                continue
            if not isinstance(row, dict):
                raise RuntimeError("issue501 malformed optional-stage route row")
            fraction = row.get("executed_fraction")
            if not isinstance(fraction, (float, int)):
                raise RuntimeError("issue501 optional-stage route missing executed_fraction")
            fractions.append(float(fraction))
    return fractions


def _finite_output(output: dict[str, object]) -> bool:
    logits = output.get("logits")
    state = output.get("state")
    if not isinstance(logits, torch.Tensor) or not isinstance(state, HardwareAERAState):
        return False
    if not bool(torch.isfinite(logits).all()):
        return False
    for stage_state in state.stages:
        if not bool(torch.isfinite(stage_state.stream).all()):
            return False
        memory = stage_state.memory
        if not isinstance(memory, ContextualEpisodicMemoryState):
            return False
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (memory.keys, memory.values, memory.strengths)
        ):
            return False
    return True


def _state_equivalence(reference: HardwareAERAState, candidate: HardwareAERAState) -> dict[str, Any]:
    if len(reference.stages) != len(candidate.stages):
        return {"pass": False, "reason": "stage_count_mismatch"}
    maxima = {"stream": 0.0, "keys": 0.0, "values": 0.0, "strengths": 0.0}
    validity_exact = True
    dtype_device_shape_exact = True
    for ref_stage, cand_stage in zip(reference.stages, candidate.stages):
        ref_memory = ref_stage.memory
        cand_memory = cand_stage.memory
        if not isinstance(ref_memory, ContextualEpisodicMemoryState) or not isinstance(
            cand_memory, ContextualEpisodicMemoryState
        ):
            return {"pass": False, "reason": "episodic_state_type_mismatch"}
        pairs = (
            ("stream", ref_stage.stream, cand_stage.stream),
            ("keys", ref_memory.keys, cand_memory.keys),
            ("values", ref_memory.values, cand_memory.values),
            ("strengths", ref_memory.strengths, cand_memory.strengths),
        )
        for name, ref_tensor, cand_tensor in pairs:
            dtype_device_shape_exact = dtype_device_shape_exact and (
                ref_tensor.dtype == cand_tensor.dtype
                and ref_tensor.device == cand_tensor.device
                and ref_tensor.shape == cand_tensor.shape
            )
            maxima[name] = max(
                maxima[name],
                float((ref_tensor.float() - cand_tensor.float()).abs().max()),
            )
        validity_exact = validity_exact and bool(torch.equal(ref_memory.valid, cand_memory.valid))
    maximum = max(maxima.values())
    return {
        "pass": bool(
            maximum <= INTEGRATED_ATOL
            and validity_exact
            and dtype_device_shape_exact
        ),
        "atol": INTEGRATED_ATOL,
        "rtol": INTEGRATED_RTOL,
        "max_stream_abs": maxima["stream"],
        "max_keys_abs": maxima["keys"],
        "max_values_abs": maxima["values"],
        "max_strengths_abs": maxima["strengths"],
        "max_continuous_abs": maximum,
        "validity_exact": validity_exact,
        "dtype_device_shape_exact": dtype_device_shape_exact,
    }


def _logit_equivalence(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    exact_meta = (
        reference.dtype == candidate.dtype
        and reference.device == candidate.device
        and reference.shape == candidate.shape
    )
    max_abs = float((reference.float() - candidate.float()).abs().max())
    close = bool(
        torch.allclose(
            reference.float(),
            candidate.float(),
            atol=INTEGRATED_ATOL,
            rtol=INTEGRATED_RTOL,
        )
    )
    return {
        "pass": bool(exact_meta and close),
        "allclose": close,
        "dtype_device_shape_exact": exact_meta,
        "max_abs": max_abs,
        "atol": INTEGRATED_ATOL,
        "rtol": INTEGRATED_RTOL,
    }


def _episodic_state_bytes_per_session(state: HardwareAERAState, batch_size: int) -> int:
    total = 0
    for stage_state in state.stages:
        memory = stage_state.memory
        if not isinstance(memory, ContextualEpisodicMemoryState):
            raise RuntimeError("issue501 expected contextual episodic state")
        for tensor in (memory.keys, memory.values, memory.strengths, memory.valid):
            total += tensor.numel() * tensor.element_size()
    if total % batch_size:
        raise RuntimeError("issue501 episodic state bytes do not divide batch")
    return total // batch_size


def _write_geometry(model: HardwareAwareAERATextLMV26) -> dict[str, Any]:
    rows: list[dict[str, int]] = []
    for stage_index, stage in enumerate(model.stages):
        rows.append(
            {
                "stage": stage_index,
                "candidates": int(stage.last_candidate_count),
                "selected_writes": int(stage.last_selected_count),
                "vectorized_updates": int(stage.last_vectorized_update_calls),
            }
        )
    nonzero = [row for row in rows if row["candidates"] > 0]
    exact = bool(
        nonzero
        and all(
            row["candidates"] == EXPECTED_CANDIDATES
            and row["selected_writes"] == EXPECTED_SELECTED_WRITES
            and row["vectorized_updates"] == EXPECTED_VECTOR_UPDATES
            for row in nonzero
        )
    )
    return {"pass": exact, "stage_rows": rows, "nonzero_stage_rows": len(nonzero)}


def _reset_execution_counters(model: HardwareAwareAERATextLMV26) -> None:
    for name in (
        "coalesced_float_state_select_calls",
        "coalesced_valid_select_calls",
        "coalesced_float_state_merge_calls",
        "coalesced_valid_merge_calls",
        "coalesced_pack_calls",
        "legacy_float_component_selects_avoided",
        "legacy_float_component_merges_avoided",
    ):
        setattr(model, name, 0)
    for stage in model.stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue501 counter reset expected CoalescedFICEMMemory")
        memory.backend_read_calls = 0
        memory.backend_update_calls = 0
        memory.backend_projected_update_calls = 0


def _physical_sparse_proof(model: HardwareAwareAERATextLMV26, output: dict[str, object]) -> dict[str, Any]:
    fractions = _optional_executed_fractions(output)
    sparse_route_exercised = bool(fractions and any(fraction < 1.0 for fraction in fractions))
    backend_names: list[str] = []
    read_calls = 0
    update_calls = 0
    projected_update_calls = 0
    for stage in model.stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue501 sparse proof expected CoalescedFICEMMemory")
        backend_names.append(memory.execution_backend_name)
        read_calls += int(memory.backend_read_calls)
        update_calls += int(memory.backend_update_calls)
        projected_update_calls += int(memory.backend_projected_update_calls)
    backend_exact = bool(
        backend_names
        and all(name == TritonFICEMReadWriteBackend.name for name in backend_names)
    )
    coalesced_select_merge = bool(
        model.coalesced_float_state_select_calls > 0
        and model.coalesced_valid_select_calls > 0
        and model.coalesced_float_state_merge_calls > 0
        and model.coalesced_valid_merge_calls > 0
    )
    backend_activity = bool(read_calls > 0 and (update_calls + projected_update_calls) > 0)
    return {
        "pass": bool(sparse_route_exercised and coalesced_select_merge and backend_activity and backend_exact),
        "optional_executed_fractions": fractions,
        "sparse_route_exercised": sparse_route_exercised,
        "coalesced_select_merge_positive": coalesced_select_merge,
        "backend_activity_positive": backend_activity,
        "backend_names_exact": backend_exact,
        "backend_names": backend_names,
        "backend_read_calls": read_calls,
        "backend_update_calls": update_calls,
        "backend_projected_update_calls": projected_update_calls,
        "coalesced_float_state_select_calls": int(model.coalesced_float_state_select_calls),
        "coalesced_valid_select_calls": int(model.coalesced_valid_select_calls),
        "coalesced_float_state_merge_calls": int(model.coalesced_float_state_merge_calls),
        "coalesced_valid_merge_calls": int(model.coalesced_valid_merge_calls),
        "dense_masked_sparse_credit": False,
    }


def _timed_summaries(calls: dict[str, Callable[[], object]], *, batch_size: int) -> dict[str, dict[str, float]]:
    for call in calls.values():
        for _ in range(SYSTEM_WARMUP_CALLS):
            output = call()
            del output
    samples: dict[str, list[float]] = {name: [] for name in calls}
    names = list(calls)
    for round_index in range(SYSTEM_ROUNDS):
        rotated = names[round_index:] + names[:round_index]
        for name in rotated:
            for _ in range(SYSTEM_TIMED_CALLS_PER_ROUND):
                samples[name].append(triage._cuda_timed_call(calls[name]))
    return {
        name: triage._timing_stats(values, batch_size=batch_size)
        for name, values in samples.items()
    }


def _peak_vram_mb(call: Callable[[], object]) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = call()
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    del output
    return {"peak_allocated_mb": float(allocated), "peak_reserved_mb": float(reserved)}


def _profile_candidate(call: Callable[[], object]) -> dict[str, Any]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        output = call()
        del output
    torch.cuda.synchronize()
    fragmentation = {token: 0 for token in ("copy", "index", "gather", "scatter", "topk", "cat")}
    rows: list[dict[str, Any]] = []
    for item in profile.key_averages():
        key = str(item.key)
        lower = key.lower()
        for token in fragmentation:
            if token in lower:
                fragmentation[token] += int(item.count)
        self_cuda_us = float(
            getattr(item, "self_cuda_time_total", getattr(item, "self_device_time_total", 0.0))
        )
        rows.append({"operator": key, "self_cuda_time_us": self_cuda_us, "calls": int(item.count)})
    rows.sort(key=lambda row: row["self_cuda_time_us"], reverse=True)
    return {"top_cuda_operators": rows[:20], "fragmentation_operator_calls": fragmentation}


def _threshold_for_batch(batch_size: int) -> float:
    if batch_size == 8:
        return BATCH8_MIN_FULL_SPEED_RATIO
    if batch_size == 64:
        return BATCH64_MIN_FULL_SPEED_RATIO
    raise ValueError(f"unregistered issue501 systems batch size: {batch_size}")


@torch.inference_mode()
def run_end_to_end_systems(*, run_dir: str = CHECKPOINT_RELATIVE_DIR) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue501 integrated systems comparison requires one NVIDIA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    hashes_before = checkpoint_hashes(run_dir)
    reference, candidate, transformer = load_models(run_dir=run_dir, device=device)
    reference_versions_before = _parameter_versions(reference)
    candidate_versions_before = _parameter_versions(candidate)
    transformer_versions_before = _parameter_versions(transformer)

    rows: dict[str, Any] = {}
    for batch_size in SYSTEM_BATCH_SIZES:
        generator = torch.Generator(device="cpu").manual_seed(
            triage.DIAGNOSTIC_SEED + 10_000 + batch_size
        )
        tokens = torch.randint(
            0,
            triage.VOCAB_SIZE,
            (batch_size, triage.SEQ_LEN),
            generator=generator,
        ).to(device)

        transformer_call = lambda: _transformer_call(transformer, tokens)
        reference_full_call = lambda: _model_call(reference, tokens, update_memory=True)
        candidate_full_call = lambda: _model_call(candidate, tokens, update_memory=True)
        calls: dict[str, Callable[[], object]] = {
            "transformer": transformer_call,
            "v26_torch_reference_full_ficem": reference_full_call,
            "v26_4_triton_full_ficem": candidate_full_call,
        }
        summaries = _timed_summaries(calls, batch_size=batch_size)

        _reset_execution_counters(reference)
        reference_output = reference_full_call()
        if not isinstance(reference_output, dict):
            raise RuntimeError("issue501 reference full call did not return mapping")
        _reset_execution_counters(candidate)
        candidate_output = candidate_full_call()
        if not isinstance(candidate_output, dict):
            raise RuntimeError("issue501 candidate full call did not return mapping")

        reference_logits = reference_output.get("logits")
        candidate_logits = candidate_output.get("logits")
        reference_state = reference_output.get("state")
        candidate_state = candidate_output.get("state")
        if not isinstance(reference_logits, torch.Tensor) or not isinstance(candidate_logits, torch.Tensor):
            raise RuntimeError("issue501 full call missing logits")
        if not isinstance(reference_state, HardwareAERAState) or not isinstance(candidate_state, HardwareAERAState):
            raise RuntimeError("issue501 full call missing HardwareAERAState")

        reference_signature = _route_signature(reference_output)
        candidate_signature = _route_signature(candidate_output)
        routing_exact = bool(
            len(reference_signature) == len(candidate_signature)
            and all(torch.equal(a, b) for a, b in zip(reference_signature, candidate_signature))
            and triage._routing_accounting(reference_output, batch_size)
            == triage._routing_accounting(candidate_output, batch_size)
        )
        logit_equivalence = _logit_equivalence(reference_logits, candidate_logits)
        state_equivalence = _state_equivalence(reference_state, candidate_state)
        physical_sparse = _physical_sparse_proof(candidate, candidate_output)
        write_geometry = _write_geometry(candidate)
        finite = _finite_output(reference_output) and _finite_output(candidate_output)
        actual_state_bytes = _episodic_state_bytes_per_session(candidate_state, batch_size)

        transformer_tps = summaries["transformer"]["tokens_per_second_from_median"]
        candidate_tps = summaries["v26_4_triton_full_ficem"]["tokens_per_second_from_median"]
        reference_ms = summaries["v26_torch_reference_full_ficem"]["median_ms"]
        candidate_ms = summaries["v26_4_triton_full_ficem"]["median_ms"]
        full_speed_ratio = candidate_tps / transformer_tps
        required_speed_ratio = _threshold_for_batch(batch_size)
        no_reference_latency_regression = candidate_ms <= reference_ms

        rows[str(batch_size)] = {
            "timings": summaries,
            "routing_reference": triage._routing_accounting(reference_output, batch_size),
            "routing_candidate": triage._routing_accounting(candidate_output, batch_size),
            "routing_exact": routing_exact,
            "logit_equivalence": logit_equivalence,
            "state_equivalence": state_equivalence,
            "physical_sparse": physical_sparse,
            "write_geometry": write_geometry,
            "finite": finite,
            "persistent_state_bytes_per_session_actual": actual_state_bytes,
            "persistent_state_bytes_pass": actual_state_bytes == EXPECTED_STATE_BYTES,
            "candidate_full_vs_transformer_speed_ratio": full_speed_ratio,
            "required_full_speed_ratio": required_speed_ratio,
            "throughput_pass": full_speed_ratio >= required_speed_ratio,
            "reference_full_latency_ms": reference_ms,
            "candidate_full_latency_ms": candidate_ms,
            "candidate_vs_reference_latency_ratio": candidate_ms / reference_ms,
            "no_reference_full_latency_regression": no_reference_latency_regression,
            "peak_vram": {
                "transformer": _peak_vram_mb(transformer_call),
                "v26_torch_reference_full": _peak_vram_mb(reference_full_call),
                "v26_4_triton_full": _peak_vram_mb(candidate_full_call),
            },
            "profiler_candidate_full": _profile_candidate(candidate_full_call),
        }
        del reference_output, candidate_output, reference_state, candidate_state, tokens
        torch.cuda.empty_cache()

    versions_unchanged = bool(
        reference_versions_before == _parameter_versions(reference)
        and candidate_versions_before == _parameter_versions(candidate)
        and transformer_versions_before == _parameter_versions(transformer)
    )
    hashes_after = checkpoint_hashes(run_dir)
    checkpoint_hashes_unchanged = hashes_before == hashes_after

    per_batch_pass = {
        batch: bool(
            row["routing_exact"]
            and row["logit_equivalence"]["pass"]
            and row["state_equivalence"]["pass"]
            and row["physical_sparse"]["pass"]
            and row["write_geometry"]["pass"]
            and row["finite"]
            and row["persistent_state_bytes_pass"]
            and row["throughput_pass"]
            and row["no_reference_full_latency_regression"]
        )
        for batch, row in rows.items()
    }
    overall_pass = bool(
        all(per_batch_pass.values())
        and versions_unchanged
        and checkpoint_hashes_unchanged
    )
    return {
        "scope": "aera_v26_5_issue501_physically_real_sparse_end_to_end_systems",
        "protocol": systems_protocol(),
        "device": torch.cuda.get_device_name(device),
        "rows": rows,
        "per_batch_pass": per_batch_pass,
        "parameter_versions_unchanged": versions_unchanged,
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": hashes_after,
        "checkpoint_hashes_unchanged": checkpoint_hashes_unchanged,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_written": False,
        "scientific_seed_consumed": False,
        "independent_replication_credit": False,
        "overall_pass": overall_pass,
        "decision": "PASS_E2E_SYSTEMS" if overall_pass else "FAIL_FROZEN_E2E_SYSTEMS_GATE",
        "claims": {
            "systems_prerequisite_satisfied": overall_pass,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
