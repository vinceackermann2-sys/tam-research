from __future__ import annotations

"""Single-attempt AERA-v25.1 systems comparison frozen by issue #381.

This module is inference-only. It reuses the seed8471 checkpoint read-only to
compare original v25 and the state-dict-compatible v25.1 execution candidate on
identical fixed random tokens. It never opens a corpus, creates an optimizer,
performs backward, updates parameters, or writes a model checkpoint.
"""

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from . import aera_real_language_v12 as v12
from . import aera_real_language_v25 as v25
from . import aera_v25_post8471_triage as triage
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v23 import sparse_write_budget
from .aera_hardware_core_v24 import (
    ContextualEpisodicMemoryState,
    episodic_state_bytes_per_session,
)
from .aera_hardware_core_v25_1 import HardwareAwareAERATextLMV251

RESEARCH_ISSUE = 381
SOURCE_IMPLEMENTATION_PR = 382
SOURCE_IMPLEMENTATION_MERGE = "d6327653498da0c693c24d31b9090743b7e2a0d9"
SOURCE_SEED = 8471
CPU_EQUIVALENCE_ATOL = 1e-6
CPU_EQUIVALENCE_RTOL = 1e-6
CPU_EQUIVALENCE_FULL_REPO_TESTS = 421
SYSTEM_BATCH_SIZES: tuple[int, ...] = (8, 64)
SYSTEM_WARMUP_CALLS = triage.SYSTEM_WARMUP_CALLS
SYSTEM_TIMED_CALLS_PER_ROUND = triage.SYSTEM_TIMED_CALLS_PER_ROUND
SYSTEM_ROUNDS = triage.SYSTEM_ROUNDS
BATCH8_MIN_FULL_SPEED_RATIO = 0.25
BATCH64_MIN_FULL_SPEED_RATIO = 1.25
EXPECTED_STATE_BYTES = 77_760
EXPECTED_CANDIDATES = 255
EXPECTED_SELECTED_WRITES = 16
EXPECTED_VECTORIZED_UPDATES = 1


def systems_protocol() -> dict[str, Any]:
    return {
        "version": "aera-v25.1-issue381-systems-only",
        "research_issue": RESEARCH_ISSUE,
        "source_implementation_pr": SOURCE_IMPLEMENTATION_PR,
        "source_implementation_merge": SOURCE_IMPLEMENTATION_MERGE,
        "source_checkpoint_seed": SOURCE_SEED,
        "source_checkpoint_use": "read-only inference/system geometry only",
        "cpu_equivalence_atol": CPU_EQUIVALENCE_ATOL,
        "cpu_equivalence_rtol": CPU_EQUIVALENCE_RTOL,
        "cpu_equivalence_full_repo_tests": CPU_EQUIVALENCE_FULL_REPO_TESTS,
        "batch_sizes": list(SYSTEM_BATCH_SIZES),
        "random_token_seed_rule": "138471 + 10000 + batch_size (identical to issue379)",
        "warmup_calls": SYSTEM_WARMUP_CALLS,
        "timed_calls_per_round": SYSTEM_TIMED_CALLS_PER_ROUND,
        "rounds": SYSTEM_ROUNDS,
        "timing_order": "rotated interleaved conditions per issue379",
        "timing_clock": "CUDA events with synchronize before/after",
        "batch8_min_full_speed_ratio": BATCH8_MIN_FULL_SPEED_RATIO,
        "batch64_min_full_speed_ratio": BATCH64_MIN_FULL_SPEED_RATIO,
        "no_full_latency_regression_vs_original_v25": True,
        "production_write_geometry": [
            EXPECTED_SELECTED_WRITES,
            EXPECTED_CANDIDATES,
            EXPECTED_VECTORIZED_UPDATES,
        ],
        "persistent_state_bytes_per_session": EXPECTED_STATE_BYTES,
        "gpu": "one NVIDIA L4 only",
        "max_gpu_seconds": 600,
        "actions_attempt": 1,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_write_authorized": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
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


def load_models(*, run_dir: str, device: torch.device):
    root = Path(run_dir)
    aera_payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    transformer_payload = torch.load(
        root / "transformer.pt", map_location="cpu", weights_only=False
    )
    if (
        aera_payload.get("seed") != SOURCE_SEED
        or transformer_payload.get("seed") != SOURCE_SEED
    ):
        raise RuntimeError("issue381 source checkpoint seed mismatch")

    torch.manual_seed(SOURCE_SEED)
    original = v25.build_aera(device).eval()
    torch.manual_seed(SOURCE_SEED)
    candidate = HardwareAwareAERATextLMV251(v12.aera_v12_config()).to(device).eval()
    torch.manual_seed(SOURCE_SEED)
    transformer = v25.build_transformer(device).eval()

    original.load_state_dict(aera_payload["model"], strict=True)
    candidate.load_state_dict(aera_payload["model"], strict=True)
    transformer.load_state_dict(transformer_payload["model"], strict=True)
    original.set_memory_pretraining_mode(False)
    candidate.set_memory_pretraining_mode(False)

    if original.state_dict().keys() != candidate.state_dict().keys():
        raise RuntimeError("v25.1 state-dict schema differs from v25")
    if sum(p.numel() for p in original.parameters()) != sum(
        p.numel() for p in candidate.parameters()
    ):
        raise RuntimeError("v25.1 learned parameter count differs from v25")
    return original, candidate, transformer


def cpu_contract_preflight() -> dict[str, Any]:
    if SYSTEM_BATCH_SIZES != (8, 64):
        raise RuntimeError("issue381 systems batch sizes drifted")
    if (
        SYSTEM_WARMUP_CALLS != 3
        or SYSTEM_TIMED_CALLS_PER_ROUND != 20
        or SYSTEM_ROUNDS != 5
    ):
        raise RuntimeError("issue381 timing contract drifted from issue379")
    if sparse_write_budget(EXPECTED_CANDIDATES) != EXPECTED_SELECTED_WRITES:
        raise RuntimeError("issue381 production sparse-write budget drifted")
    state_bytes = episodic_state_bytes_per_session(n_stages=4, memory_dim=50)
    if state_bytes != EXPECTED_STATE_BYTES:
        raise RuntimeError("issue381 persistent FICEM state accounting drifted")
    if BATCH8_MIN_FULL_SPEED_RATIO != 0.25 or BATCH64_MIN_FULL_SPEED_RATIO != 1.25:
        raise RuntimeError("issue381 throughput thresholds drifted")
    return {
        "protocol": systems_protocol(),
        "state_bytes": state_bytes,
        "selected_writes": sparse_write_budget(EXPECTED_CANDIDATES),
        "gpu_authorized_by_cpu_preflight": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_written": False,
    }


def _model_call(model, tokens: torch.Tensor, *, update_memory: bool):
    return triage._model_forward(model, tokens, update_memory=update_memory)


def _core_call(model, tokens: torch.Tensor):
    with triage.bypass_ficem_reads(model):
        return _model_call(model, tokens, update_memory=False)


def _writes_only_call(model, tokens: torch.Tensor):
    with triage.bypass_ficem_reads(model):
        return _model_call(model, tokens, update_memory=True)


def _route_signature(output: dict[str, object]) -> tuple[torch.Tensor, ...]:
    routes = output.get("stage_routes")
    if not isinstance(routes, list):
        raise RuntimeError("AERA output missing stage routes")
    signature: list[torch.Tensor] = []
    for chunk in routes:
        if not isinstance(chunk, list):
            raise RuntimeError("invalid stage-route chunk")
        for row in chunk:
            if not isinstance(row, dict):
                raise RuntimeError("invalid stage-route row")
            gate = row.get("stage_route_gate")
            if not isinstance(gate, torch.Tensor):
                raise RuntimeError("stage route missing gate tensor")
            signature.append(gate.detach().ge(0.5).cpu())
    return tuple(signature)


def _state_equivalence(
    original: HardwareAERAState,
    candidate: HardwareAERAState,
) -> dict[str, Any]:
    if len(original.stages) != len(candidate.stages):
        return {"pass": False, "reason": "stage_count_mismatch"}
    max_stream = 0.0
    max_keys = 0.0
    max_values = 0.0
    max_strengths = 0.0
    valid_exact = True
    for old_stage, new_stage in zip(original.stages, candidate.stages):
        if not isinstance(old_stage.memory, ContextualEpisodicMemoryState) or not isinstance(
            new_stage.memory, ContextualEpisodicMemoryState
        ):
            return {"pass": False, "reason": "episodic_state_type_mismatch"}
        max_stream = max(
            max_stream,
            float((old_stage.stream.float() - new_stage.stream.float()).abs().max()),
        )
        max_keys = max(
            max_keys,
            float((old_stage.memory.keys.float() - new_stage.memory.keys.float()).abs().max()),
        )
        max_values = max(
            max_values,
            float((old_stage.memory.values.float() - new_stage.memory.values.float()).abs().max()),
        )
        max_strengths = max(
            max_strengths,
            float(
                (
                    old_stage.memory.strengths.float()
                    - new_stage.memory.strengths.float()
                ).abs().max()
            ),
        )
        valid_exact = valid_exact and bool(
            torch.equal(old_stage.memory.valid, new_stage.memory.valid)
        )
    maximum = max(max_stream, max_keys, max_values, max_strengths)
    return {
        "pass": bool(maximum <= CPU_EQUIVALENCE_ATOL and valid_exact),
        "atol": CPU_EQUIVALENCE_ATOL,
        "max_stream_abs": max_stream,
        "max_keys_abs": max_keys,
        "max_values_abs": max_values,
        "max_strengths_abs": max_strengths,
        "max_continuous_abs": maximum,
        "valid_exact": valid_exact,
    }


def _episodic_state_bytes_per_session(state: HardwareAERAState, batch_size: int) -> int:
    total = 0
    for stage_state in state.stages:
        memory = stage_state.memory
        if not isinstance(memory, ContextualEpisodicMemoryState):
            raise RuntimeError("issue381 expected contextual episodic state")
        for tensor in (memory.keys, memory.values, memory.strengths, memory.valid):
            total += tensor.numel() * tensor.element_size()
    if total % batch_size:
        raise RuntimeError("episodic state bytes do not divide batch")
    return total // batch_size


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


def _write_geometry(candidate) -> dict[str, Any]:
    rows: list[dict[str, int]] = []
    for stage_index, stage in enumerate(candidate.stages):
        candidates = int(stage.last_candidate_count)
        selected = int(stage.last_selected_count)
        updates = int(stage.last_vectorized_update_calls)
        rows.append(
            {
                "stage": stage_index,
                "candidates": candidates,
                "selected_writes": selected,
                "vectorized_updates": updates,
            }
        )
    stage0 = rows[0]
    nonzero_rows = [row for row in rows if row["candidates"] > 0]
    exact = (
        stage0["candidates"] == EXPECTED_CANDIDATES
        and stage0["selected_writes"] == EXPECTED_SELECTED_WRITES
        and stage0["vectorized_updates"] == EXPECTED_VECTORIZED_UPDATES
        and all(
            row["candidates"] == EXPECTED_CANDIDATES
            and row["selected_writes"] == EXPECTED_SELECTED_WRITES
            and row["vectorized_updates"] == EXPECTED_VECTORIZED_UPDATES
            for row in nonzero_rows
        )
    )
    return {"pass": bool(exact), "stage_rows": rows}


def _peak_vram_mb(call: Callable[[], object]) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = call()
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    del output
    return {"peak_allocated_mb": float(allocated), "peak_reserved_mb": float(reserved)}


def _profile_full_candidate(call: Callable[[], object]) -> dict[str, Any]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        output = call()
        del output
    torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    fragmentation = {
        "copy": 0,
        "index": 0,
        "gather": 0,
        "scatter": 0,
        "cat": 0,
        "topk": 0,
    }
    for item in profile.key_averages():
        key = str(item.key)
        lower = key.lower()
        for token in fragmentation:
            if token in lower:
                fragmentation[token] += int(item.count)
        self_cuda_us = float(
            getattr(
                item,
                "self_cuda_time_total",
                getattr(item, "self_device_time_total", 0.0),
            )
        )
        rows.append(
            {
                "operator": key,
                "self_cuda_time_us": self_cuda_us,
                "cuda_time_us": float(
                    getattr(
                        item,
                        "cuda_time_total",
                        getattr(item, "device_time_total", 0.0),
                    )
                ),
                "calls": int(item.count),
            }
        )
    rows.sort(key=lambda row: row["self_cuda_time_us"], reverse=True)
    return {"top_cuda_operators": rows[:20], "fragmentation_operator_calls": fragmentation}


def _timed_summaries(
    calls: dict[str, Callable[[], object]],
    *,
    batch_size: int,
) -> dict[str, dict[str, float]]:
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


def _threshold_for_batch(batch_size: int) -> float:
    if batch_size == 8:
        return BATCH8_MIN_FULL_SPEED_RATIO
    if batch_size == 64:
        return BATCH64_MIN_FULL_SPEED_RATIO
    raise ValueError(f"unregistered systems batch size: {batch_size}")


@torch.inference_mode()
def run_systems_comparison(*, run_dir: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue381 systems comparison requires one NVIDIA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    hashes_before = checkpoint_hashes(run_dir)
    original, candidate, transformer = load_models(run_dir=run_dir, device=device)
    original_versions_before = _parameter_versions(original)
    candidate_versions_before = _parameter_versions(candidate)

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

        def transformer_call():
            with triage._autocast(device):
                return transformer(tokens)

        calls: dict[str, Callable[[], object]] = {
            "transformer": transformer_call,
            "original_v25_core_memory_bypassed": lambda: _core_call(original, tokens),
            "v25_1_core_memory_bypassed": lambda: _core_call(candidate, tokens),
            "original_v25_full_ficem": lambda: _model_call(
                original, tokens, update_memory=True
            ),
            "v25_1_full_ficem": lambda: _model_call(
                candidate, tokens, update_memory=True
            ),
            "v25_1_read_empty": lambda: _model_call(
                candidate, tokens, update_memory=False
            ),
            "v25_1_writes_only_reads_bypassed": lambda: _writes_only_call(
                candidate, tokens
            ),
        }
        summaries = _timed_summaries(calls, batch_size=batch_size)

        old_output = calls["original_v25_full_ficem"]()
        if not isinstance(old_output, dict):
            raise RuntimeError("original v25 full call did not return mapping")
        old_signature = _route_signature(old_output)
        old_routing = triage._routing_accounting(old_output, batch_size)
        old_state = old_output.get("state")
        if not isinstance(old_state, HardwareAERAState):
            raise RuntimeError("original v25 full call missing state")
        del old_output

        candidate_output = calls["v25_1_full_ficem"]()
        if not isinstance(candidate_output, dict):
            raise RuntimeError("v25.1 full call did not return mapping")
        candidate_signature = _route_signature(candidate_output)
        candidate_routing = triage._routing_accounting(candidate_output, batch_size)
        candidate_state = candidate_output.get("state")
        if not isinstance(candidate_state, HardwareAERAState):
            raise RuntimeError("v25.1 full call missing state")
        finite = _finite_output(candidate_output)
        actual_state_bytes = _episodic_state_bytes_per_session(
            candidate_state, batch_size
        )
        state_equivalence = _state_equivalence(old_state, candidate_state)
        routing_exact = (
            len(old_signature) == len(candidate_signature)
            and all(
                torch.equal(old_gate, new_gate)
                for old_gate, new_gate in zip(old_signature, candidate_signature)
            )
            and old_routing == candidate_routing
        )
        write_geometry = _write_geometry(candidate)
        del candidate_output, old_state, candidate_state

        transformer_tps = summaries["transformer"]["tokens_per_second_from_median"]
        old_core_tps = summaries["original_v25_core_memory_bypassed"][
            "tokens_per_second_from_median"
        ]
        new_core_tps = summaries["v25_1_core_memory_bypassed"][
            "tokens_per_second_from_median"
        ]
        old_full_tps = summaries["original_v25_full_ficem"][
            "tokens_per_second_from_median"
        ]
        new_full_tps = summaries["v25_1_full_ficem"][
            "tokens_per_second_from_median"
        ]
        new_core_ms = summaries["v25_1_core_memory_bypassed"]["median_ms"]
        new_read_empty_ms = summaries["v25_1_read_empty"]["median_ms"]
        new_writes_only_ms = summaries["v25_1_writes_only_reads_bypassed"][
            "median_ms"
        ]
        old_full_ms = summaries["original_v25_full_ficem"]["median_ms"]
        new_full_ms = summaries["v25_1_full_ficem"]["median_ms"]
        speed_ratio = new_full_tps / transformer_tps
        threshold = _threshold_for_batch(batch_size)
        no_latency_regression = new_full_ms <= old_full_ms

        rows[str(batch_size)] = {
            "timings": summaries,
            "routing_original_v25": old_routing,
            "routing_v25_1": candidate_routing,
            "routing_exact": routing_exact,
            "state_equivalence": state_equivalence,
            "write_geometry": write_geometry,
            "candidate_output_state_finite": finite,
            "persistent_state_bytes_per_session_actual": actual_state_bytes,
            "persistent_state_bytes_pass": actual_state_bytes == EXPECTED_STATE_BYTES,
            "original_core_vs_transformer_speed_ratio": old_core_tps / transformer_tps,
            "v25_1_core_vs_transformer_speed_ratio": new_core_tps / transformer_tps,
            "original_full_vs_transformer_speed_ratio": old_full_tps / transformer_tps,
            "v25_1_full_vs_transformer_speed_ratio": speed_ratio,
            "required_full_speed_ratio": threshold,
            "throughput_pass": speed_ratio >= threshold,
            "full_latency_original_v25_ms": old_full_ms,
            "full_latency_v25_1_ms": new_full_ms,
            "no_full_latency_regression": no_latency_regression,
            "v25_1_core_latency_reduction_vs_original_ms": (
                summaries["original_v25_core_memory_bypassed"]["median_ms"]
                - new_core_ms
            ),
            "v25_1_read_empty_incremental_over_core_ms": new_read_empty_ms - new_core_ms,
            "v25_1_writes_only_incremental_over_core_ms": (
                new_writes_only_ms - new_core_ms
            ),
            "v25_1_full_incremental_memory_over_core_ms": new_full_ms - new_core_ms,
            "peak_vram": {
                "transformer": _peak_vram_mb(calls["transformer"]),
                "original_v25_full": _peak_vram_mb(calls["original_v25_full_ficem"]),
                "v25_1_full": _peak_vram_mb(calls["v25_1_full_ficem"]),
            },
            "profiler_v25_1_full": _profile_full_candidate(
                calls["v25_1_full_ficem"]
            ),
        }
        del tokens
        torch.cuda.empty_cache()

    original_versions_after = _parameter_versions(original)
    candidate_versions_after = _parameter_versions(candidate)
    hashes_after = checkpoint_hashes(run_dir)
    parameter_versions_unchanged = (
        original_versions_before == original_versions_after
        and candidate_versions_before == candidate_versions_after
    )
    checkpoint_hashes_unchanged = hashes_before == hashes_after

    per_batch_pass = {
        batch: bool(
            row["routing_exact"]
            and row["state_equivalence"]["pass"]
            and row["write_geometry"]["pass"]
            and row["candidate_output_state_finite"]
            and row["persistent_state_bytes_pass"]
            and row["throughput_pass"]
            and row["no_full_latency_regression"]
        )
        for batch, row in rows.items()
    }
    overall_pass = bool(
        all(per_batch_pass.values())
        and parameter_versions_unchanged
        and checkpoint_hashes_unchanged
    )
    return {
        "scope": "aera_v25_1_issue381_single_l4_systems_only",
        "protocol": systems_protocol(),
        "gpu": torch.cuda.get_device_name(device),
        "timing_method": (
            "CUDA events; 3 warmups; 20 calls/condition x5 rotated interleaved rounds; "
            "same fixed random-token seed rule as issue379"
        ),
        "rows": rows,
        "per_batch_pass": per_batch_pass,
        "parameter_versions_unchanged": parameter_versions_unchanged,
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": hashes_after,
        "checkpoint_hashes_unchanged": checkpoint_hashes_unchanged,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "new_model_checkpoint_written": False,
        "scientific_seed_consumed": False,
        "overall_pass": overall_pass,
        "decision": (
            "PASS_SYSTEMS_EQUIVALENT_CANDIDATE"
            if overall_pass
            else "FAIL_FROZEN_SYSTEMS_GATE"
        ),
        "claims": {
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
