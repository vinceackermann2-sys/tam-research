from __future__ import annotations

"""Guarded AERA-v25 FICEM real-language development harness frozen by #366.

Seed8471 is fresh development-only evidence. The harness deliberately reuses the
v18 real-language trainer/evaluators and changes only the bound AERA model,
preregistered v25 payload teaching term, and representation-specific episodic
state audit. It cannot count toward later post-freeze replication.
"""

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import aera_real_language_v18_gpu as base
from . import aera_real_language_v25 as v25
from .aera import AERAState
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v24 import (
    EPISODIC_CAPACITY,
    ContextualEpisodicMemoryState,
    episodic_state_bytes_per_session,
)
from .aera_hardware_core_v25 import FactorizedIdentityContextEpisodicMemoryStage
from .aera_real_language import GRAD_ACCUM, SEQ_LEN, TOKEN_BUDGET, TOTAL_STEPS
from .data import TokenBin

SEED = 8471
EVAL_SEED = 98_471
MEMORY_EVAL_SEED = 108_471
SYSTEMS_EVAL_SEED = 118_471
EXPECTED_CANDIDATES = 255
EXPECTED_SELECTED_WRITES = 16
EXPECTED_VECTORIZED_UPDATES = 1
EXPECTED_STATE_BYTES = 77_760

QUALITY_GAP_MAX_NLL = base.QUALITY_GAP_MAX_NLL
MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL = base.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL
MEMORY_OVERALL_MIN_ADVANTAGE_NLL = base.MEMORY_OVERALL_MIN_ADVANTAGE_NLL
WRITE_MEAN_MIN = base.WRITE_MEAN_MIN
WRITE_MEAN_MAX = base.WRITE_MEAN_MAX
WRITE_SPREAD_MIN = base.WRITE_SPREAD_MIN
OPTIONAL_STAGE_TARGET_MAE_MAX = base.OPTIONAL_STAGE_TARGET_MAE_MAX
OPTIONAL_STAGE_MIN_RUN_FRACTION = base.OPTIONAL_STAGE_MIN_RUN_FRACTION
TOTAL_STAGE_EXEC_MIN = base.TOTAL_STAGE_EXEC_MIN
TOTAL_STAGE_EXEC_MAX = base.TOTAL_STAGE_EXEC_MAX
BATCH8_MIN_SPEED_RATIO = base.BATCH8_MIN_SPEED_RATIO
BATCH64_MIN_SPEED_RATIO = base.BATCH64_MIN_SPEED_RATIO
MEMORY_EVAL_BATCHES = base.MEMORY_EVAL_BATCHES
MEMORY_EVAL_BATCH_SIZE = base.MEMORY_EVAL_BATCH_SIZE

_ORIGINAL_VALIDATE_PROTOCOL = base.validate_protocol


def _install_v25_binding() -> None:
    base.v18 = v25
    base.SEED = SEED
    base.EVAL_SEED = EVAL_SEED
    base.MEMORY_EVAL_SEED = MEMORY_EVAL_SEED
    base.SYSTEMS_EVAL_SEED = SYSTEMS_EVAL_SEED


def _decorate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("development_seed") != SEED:
        raise RuntimeError("v25 base protocol did not pick up frozen seed8471")
    if v25.CHUNK_SIZE != 256 or SEQ_LEN != 512:
        raise RuntimeError("v25 frozen real-language chunk/sequence geometry changed")
    if v25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH != 256 or GRAD_ACCUM != 4:
        raise RuntimeError("v25 frozen payload-teaching budget changed")
    if v25.PAYLOAD_TOKEN_WEIGHT != 1.0 or v25.ADDRESS_AUXILIARY_WEIGHT != 0.0:
        raise RuntimeError("v25 frozen memory teaching weights changed")
    if episodic_state_bytes_per_session(n_stages=4, memory_dim=50) != EXPECTED_STATE_BYTES:
        raise RuntimeError("v25 episodic state accounting changed")

    protocol["version"] = "aera-v25-ficem-real-language-development"
    protocol["research_issue"] = 366
    protocol["gpu_authorization_scope"] = (
        "one guarded AERA-v25 development seed8471 L4 run only"
    )
    protocol["counts_toward_independent_replication"] = False
    protocol["architecture"] = {
        "source": "exact merged AERA-v25 controlled + systems PASS",
        "memory_dim": 50,
        "identity_dim": 25,
        "context_dim": 25,
        "episodic_capacity_per_stage": EPISODIC_CAPACITY,
        "state_bytes_per_session": EXPECTED_STATE_BYTES,
        "event_pair_candidates_per_chunk": EXPECTED_CANDIDATES,
        "physically_selected_writes_per_chunk": EXPECTED_SELECTED_WRITES,
        "vectorized_updates_per_completed_executed_stage_chunk": EXPECTED_VECTORIZED_UPDATES,
        "selector": "pair gate; hard top-k forward + straight-through soft k-hot backward",
        "selected_write_order": "original chronological order before newest-first insertion",
        "read_top_k": 4,
        "read_temperature": 0.10,
        "factorized_identity_context_address": True,
        "strictly_causal_context": True,
        "newest_wins_duplicate_replacement": True,
        "sequential_dual_delta_recurrence": False,
        "inverse_covariance_state": False,
        "routing_changed_from_inherited_real_language": False,
        "predictive_stream_changed_from_inherited_real_language": False,
    }
    protocol["memory_training_objective"] = {
        "payload_token_weight": v25.PAYLOAD_TOKEN_WEIGHT,
        "address_auxiliary_weight": v25.ADDRESS_AUXILIARY_WEIGHT,
        "max_sampled_adjacent_payload_events_per_microbatch": v25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH,
        "gradient_accumulation_microbatches": GRAD_ACCUM,
        "max_sampled_adjacent_payload_events_per_optimizer_step": (
            v25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH * GRAD_ACCUM
        ),
        "sampling": "deterministic stratified plus step-dependent cyclic offset",
        "payload_representation": "v25 strictly-causal contextual stage0 representation",
        "event_representations_detached": True,
        "decoder_weights_detached": True,
        "synthetic_address_labels": False,
    }
    protocol["thresholds_inherited_unchanged_from_v18_issue324"] = {
        "quality_gap_max_nll": QUALITY_GAP_MAX_NLL,
        "memory_second_chunk_min_advantage_nll": MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL,
        "memory_overall_min_advantage_nll": MEMORY_OVERALL_MIN_ADVANTAGE_NLL,
        "write_mean_range": [WRITE_MEAN_MIN, WRITE_MEAN_MAX],
        "write_spread_min": WRITE_SPREAD_MIN,
        "optional_stage_target_mae_max": OPTIONAL_STAGE_TARGET_MAE_MAX,
        "optional_stage_min_run_fraction": OPTIONAL_STAGE_MIN_RUN_FRACTION,
        "total_stage_execution_range": [TOTAL_STAGE_EXEC_MIN, TOTAL_STAGE_EXEC_MAX],
        "batch8_min_speed_ratio": BATCH8_MIN_SPEED_RATIO,
        "batch64_min_speed_ratio": BATCH64_MIN_SPEED_RATIO,
    }
    protocol["v25_specific_safety"] = {
        "episodic_state_bytes_exact": EXPECTED_STATE_BYTES,
        "episodic_capacity": EPISODIC_CAPACITY,
        "keys_values_strengths_finite_required": True,
        "validity_boolean_required": True,
        "session_isolation_required": True,
        "exact_sparse_geometry_required": [
            EXPECTED_SELECTED_WRITES,
            EXPECTED_CANDIDATES,
            EXPECTED_VECTORIZED_UPDATES,
        ],
        "deployment_base_parameter_nonmutation_required": True,
        "hidden_memory_state_allowed": False,
    }
    return protocol


def validate_protocol(data_dir: str) -> dict[str, Any]:
    _install_v25_binding()
    protocol = _ORIGINAL_VALIDATE_PROTOCOL(data_dir)
    if protocol.get("token_budget_per_model") != TOKEN_BUDGET or TOKEN_BUDGET != 8_388_608:
        raise RuntimeError("v25 real-language token budget changed")
    return _decorate_protocol(protocol)


def _session_isolation_probe(aera, device: torch.device) -> bool:
    g = torch.Generator(device="cpu").manual_seed(MEMORY_EVAL_SEED + 77)
    tokens = torch.randint(
        0,
        aera.cfg.vocab_size,
        (1, v25.CHUNK_SIZE),
        generator=g,
    ).to(device)
    first = aera.empty_state(tokens)
    second = aera.empty_state(tokens)
    for left, right in zip(first.stages, second.stages):
        if not isinstance(left.memory, ContextualEpisodicMemoryState):
            return False
        if not isinstance(right.memory, ContextualEpisodicMemoryState):
            return False
        for name in ("keys", "values", "strengths", "valid"):
            if getattr(left.memory, name).data_ptr() == getattr(right.memory, name).data_ptr():
                return False
    with torch.no_grad():
        first.stages[0].memory.keys.add_(1.0)
        first.stages[0].memory.values.add_(1.0)
        first.stages[0].memory.strengths.fill_(0.5)
        first.stages[0].memory.valid.fill_(True)
    untouched = second.stages[0].memory
    return bool(
        untouched.keys.eq(0).all()
        and untouched.values.eq(0).all()
        and untouched.strengths.eq(0).all()
        and (~untouched.valid).all()
    )


def _tensor_stats(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0, "count": 0.0}
    flat = torch.cat([value.detach().float().reshape(-1).cpu() for value in values])
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "p10": float(torch.quantile(flat, 0.10)),
        "p90": float(torch.quantile(flat, 0.90)),
        "count": float(flat.numel()),
    }


def _second_chunk_nll(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(
        F.cross_entropy(
            logits[:, v25.CHUNK_SIZE :].float().reshape(-1, logits.size(-1)),
            y[:, v25.CHUNK_SIZE :].reshape(-1),
        )
    )


def _capture_sparse_write_stats(
    aera,
    *,
    batch_index: int,
    rows: list[dict[str, Any]],
    pair_gate_values: list[torch.Tensor],
    selected_strength_values: list[torch.Tensor],
) -> None:
    """Snapshot diagnostics immediately after a memory-enabled forward."""
    for stage_index, stage in enumerate(aera.stages):
        if not isinstance(stage, FactorizedIdentityContextEpisodicMemoryStage):
            raise RuntimeError("v25 memory suite lost FICEM stage")
        if stage.last_candidate_count:
            rows.append(
                {
                    "batch": batch_index,
                    "stage": stage_index,
                    "candidates": int(stage.last_candidate_count),
                    "selected_writes": int(stage.last_selected_count),
                    "vectorized_updates": int(stage.last_vectorized_update_calls),
                }
            )
            if stage.last_pair_gate is not None:
                pair_gate_values.append(stage.last_pair_gate.detach().cpu())
            if stage.last_pair_strength is not None:
                selected_strength_values.append(stage.last_pair_strength.detach().cpu())


@torch.no_grad()
def _v25_memory_suite(*, data_dir: str, run_dir: str, seed: int) -> dict[str, Any]:
    device = torch.device("cuda")
    root = Path(run_dir)
    a_payload = torch.load(root / "aera.pt", map_location="cpu", weights_only=False)
    t_payload = torch.load(root / "transformer.pt", map_location="cpu", weights_only=False)
    if a_payload.get("seed") != seed or t_payload.get("seed") != seed:
        raise RuntimeError("v25 memory-suite checkpoint seed mismatch")

    torch.manual_seed(seed)
    aera = v25.build_aera(device).eval()
    torch.manual_seed(seed)
    transformer = v25.build_transformer(device).eval()
    aera.load_state_dict(a_payload["model"], strict=True)
    transformer.load_state_dict(t_payload["model"], strict=True)
    aera.set_memory_pretraining_mode(False)

    parameter_versions_before = [p._version for p in aera.parameters()]
    val = TokenBin(str(Path(data_dir) / "val.bin"))
    g = torch.Generator(device="cpu").manual_seed(MEMORY_EVAL_SEED)

    t_losses: list[float] = []
    memory_losses: list[float] = []
    stream_losses: list[float] = []
    reset_losses: list[float] = []
    memory_second: list[float] = []
    stream_second: list[float] = []
    reset_second: list[float] = []
    memory_only_second: list[float] = []
    read_values: list[torch.Tensor] = []
    write_values: list[torch.Tensor] = []
    episodic_norms: list[torch.Tensor] = []
    slot_utilization: list[torch.Tensor] = []
    pair_gate_values: list[torch.Tensor] = []
    selected_strength_values: list[torch.Tensor] = []
    sparse_rows: list[dict[str, Any]] = []
    state_bytes_per_session: int | None = None
    all_state_finite = True
    capacity_bounded = True
    validity_boolean = True

    for batch_index in range(MEMORY_EVAL_BATCHES):
        x, y = val.batch(MEMORY_EVAL_BATCH_SIZE, SEQ_LEN, g, device)
        with base._autocast(device):
            t_logits = transformer(x)
            mem_out = aera(
                x,
                hard=True,
                route_mode="hard_sparse",
                update_memory=True,
            )
        _capture_sparse_write_stats(
            aera,
            batch_index=batch_index,
            rows=sparse_rows,
            pair_gate_values=pair_gate_values,
            selected_strength_values=selected_strength_values,
        )
        with base._autocast(device):
            stream_out = aera(
                x,
                hard=True,
                route_mode="hard_sparse",
                update_memory=False,
            )

        mem_logits = mem_out["logits"]
        stream_logits = stream_out["logits"]
        assert isinstance(mem_logits, torch.Tensor) and isinstance(stream_logits, torch.Tensor)
        t_losses.append(float(F.cross_entropy(t_logits.float().reshape(-1, t_logits.size(-1)), y.reshape(-1))))
        memory_losses.append(float(F.cross_entropy(mem_logits.float().reshape(-1, mem_logits.size(-1)), y.reshape(-1))))
        stream_losses.append(float(F.cross_entropy(stream_logits.float().reshape(-1, stream_logits.size(-1)), y.reshape(-1))))
        memory_second.append(_second_chunk_nll(mem_logits, y))
        stream_second.append(_second_chunk_nll(stream_logits, y))

        reset_logits_parts: list[torch.Tensor] = []
        for start in range(0, SEQ_LEN, v25.CHUNK_SIZE):
            chunk = x[:, start : start + v25.CHUNK_SIZE]
            with base._autocast(device):
                reset_out = aera(
                    chunk,
                    state=None,
                    hard=True,
                    route_mode="hard_sparse",
                    update_memory=False,
                )
            chunk_logits = reset_out["logits"]
            assert isinstance(chunk_logits, torch.Tensor)
            reset_logits_parts.append(chunk_logits)
        reset_logits = torch.cat(reset_logits_parts, dim=1)
        reset_losses.append(float(F.cross_entropy(reset_logits.float().reshape(-1, reset_logits.size(-1)), y.reshape(-1))))
        reset_second.append(_second_chunk_nll(reset_logits, y))

        first = x[:, : v25.CHUNK_SIZE]
        second = x[:, v25.CHUNK_SIZE :]
        with base._autocast(device):
            first_out = aera(
                first,
                state=None,
                hard=True,
                route_mode="hard_sparse",
                update_memory=True,
            )
        first_state = first_out.get("state")
        if not isinstance(first_state, HardwareAERAState):
            raise RuntimeError("v25 first-chunk output missing HardwareAERAState")
        memory_only_state = HardwareAERAState(
            [
                AERAState(stream=torch.zeros_like(stage_state.stream), memory=stage_state.memory)
                for stage_state in first_state.stages
            ]
        )
        with base._autocast(device):
            memory_only_out = aera(
                second,
                state=memory_only_state,
                hard=True,
                route_mode="hard_sparse",
                update_memory=False,
            )
        memory_only_logits = memory_only_out["logits"]
        assert isinstance(memory_only_logits, torch.Tensor)
        memory_only_second.append(
            float(
                F.cross_entropy(
                    memory_only_logits.float().reshape(-1, memory_only_logits.size(-1)),
                    y[:, v25.CHUNK_SIZE :].reshape(-1),
                )
            )
        )

        reads, writes = base._collect_memory_gates(mem_out)
        read_values.extend(reads)
        write_values.extend(writes)
        final_state = mem_out.get("state")
        if not isinstance(final_state, HardwareAERAState):
            raise RuntimeError("v25 memory output missing HardwareAERAState")
        bytes_this_batch = 0
        for stage_state in final_state.stages:
            memory = stage_state.memory
            if not isinstance(memory, ContextualEpisodicMemoryState):
                raise RuntimeError("v25 memory suite lost episodic state")
            if memory.keys.shape != memory.values.shape:
                raise RuntimeError("v25 episodic key/value shape mismatch")
            if memory.keys.size(1) != EPISODIC_CAPACITY:
                capacity_bounded = False
            if memory.valid.dtype != torch.bool:
                validity_boolean = False
            all_state_finite = (
                all_state_finite
                and bool(torch.isfinite(memory.keys).all())
                and bool(torch.isfinite(memory.values).all())
                and bool(torch.isfinite(memory.strengths).all())
            )
            key_norm = torch.linalg.vector_norm(
                memory.keys.float().reshape(memory.keys.size(0), -1), dim=1
            )
            value_norm = torch.linalg.vector_norm(
                memory.values.float().reshape(memory.values.size(0), -1), dim=1
            )
            episodic_norms.append(torch.sqrt(key_norm.square() + value_norm.square()).cpu())
            slot_utilization.append(memory.valid.float().mean(dim=1).cpu())
            bytes_this_batch += memory.keys.numel() * memory.keys.element_size()
            bytes_this_batch += memory.values.numel() * memory.values.element_size()
            bytes_this_batch += memory.strengths.numel() * memory.strengths.element_size()
            bytes_this_batch += memory.valid.numel() * memory.valid.element_size()

        per_session = bytes_this_batch // MEMORY_EVAL_BATCH_SIZE
        if state_bytes_per_session is None:
            state_bytes_per_session = per_session
        elif state_bytes_per_session != per_session:
            raise RuntimeError("v25 episodic state bytes/session changed across batches")

    parameter_versions_after = [p._version for p in aera.parameters()]
    if not sparse_rows:
        raise RuntimeError("v25 held-out memory suite observed no completed memory write")
    sparse_exact = all(
        row["candidates"] == EXPECTED_CANDIDATES
        and row["selected_writes"] == EXPECTED_SELECTED_WRITES
        and row["vectorized_updates"] == EXPECTED_VECTORIZED_UPDATES
        for row in sparse_rows
    )

    aux_g = torch.Generator(device="cpu").manual_seed(MEMORY_EVAL_SEED + 91)
    aux_x, _ = val.batch(MEMORY_EVAL_BATCH_SIZE, SEQ_LEN, aux_g, device)
    with base._autocast(device):
        payload_aux = v25.payload_teaching_terms(
            aera,
            aux_x,
            step=TOTAL_STEPS - 1,
            max_events=v25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH,
        )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    t_nll = mean(t_losses)
    mem_nll = mean(memory_losses)
    stream_nll = mean(stream_losses)
    reset_nll = mean(reset_losses)
    mem_second = mean(memory_second)
    stream_second_nll = mean(stream_second)
    reset_second_nll = mean(reset_second)
    memory_only_second_nll = mean(memory_only_second)
    write_stats = base._gate_stats(write_values)
    read_stats = base._gate_stats(read_values)
    episodic_norm = float(torch.cat(episodic_norms).mean()) if episodic_norms else 0.0
    utilization = _tensor_stats(slot_utilization)
    isolation = _session_isolation_probe(aera, device)

    return {
        "seed": seed,
        "eval_seed": MEMORY_EVAL_SEED,
        "examples": MEMORY_EVAL_BATCHES * MEMORY_EVAL_BATCH_SIZE,
        "transformer_nll": t_nll,
        "stream_plus_memory_nll": mem_nll,
        "stream_only_nll": stream_nll,
        "reset_state_and_memory_nll": reset_nll,
        "quality_gap_nll": mem_nll - t_nll,
        "memory_overall_advantage_nll": stream_nll - mem_nll,
        "stream_over_reset_advantage_nll": reset_nll - stream_nll,
        "second_chunk": {
            "stream_plus_memory_nll": mem_second,
            "stream_only_nll": stream_second_nll,
            "memory_only_nll": memory_only_second_nll,
            "reset_state_and_memory_nll": reset_second_nll,
            "memory_advantage_nll": stream_second_nll - mem_second,
            "stream_advantage_nll": reset_second_nll - stream_second_nll,
            "memory_only_advantage_over_reset_nll": reset_second_nll - memory_only_second_nll,
        },
        "executed_stage_memory_read_gate": read_stats,
        "executed_stage_effective_write_strength": write_stats,
        "mean_final_memory_frobenius_norm": episodic_norm,
        "memory_state_bytes_per_session": int(state_bytes_per_session or 0),
        "episodic_state": {
            "capacity_per_stage": EPISODIC_CAPACITY,
            "all_keys_values_strengths_finite": all_state_finite,
            "capacity_bounded_exactly_48": capacity_bounded,
            "validity_dtype_boolean": validity_boolean,
            "valid_slot_fraction": utilization,
            "session_isolation_exact": isolation,
        },
        "sparse_write_execution": {
            "rows": sparse_rows,
            "all_completed_measured_stages_exact_16_of_255_one_update": sparse_exact,
            "pair_gate": _tensor_stats(pair_gate_values),
            "selected_effective_strength": _tensor_stats(selected_strength_values),
        },
        "heldout_payload_auxiliary": {
            key: float(value.detach().float().cpu()) for key, value in payload_aux.items()
        },
        "deployment_base_parameter_versions_unchanged": (
            parameter_versions_before == parameter_versions_after
        ),
        "training_performed": False,
        "checkpoint_mutated": False,
    }


def _remap_and_finalize(result: dict[str, Any], run_dir: str) -> dict[str, Any]:
    for old, new in (
        ("v18_memory_eval", "v25_memory_eval"),
        ("v18_heldout_adaptivity", "v25_heldout_adaptivity"),
        ("v18_systems_eval", "v25_systems_eval"),
        ("v18_development_checks", "v25_development_checks"),
        ("v18_development_pass", "v25_inherited_development_pass"),
    ):
        if old not in result:
            raise RuntimeError(f"v25 expected inherited result key {old!r}")
        result[new] = result.pop(old)

    protocol = result.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("v25 inherited result missing protocol")
    result["protocol"] = _decorate_protocol(protocol)

    memory = result["v25_memory_eval"]
    checks = result["v25_development_checks"]
    episodic = memory["episodic_state"]
    sparse = memory["sparse_write_execution"]
    payload = memory["heldout_payload_auxiliary"]
    checks["episodic_state_bytes_exact_77760"] = (
        int(memory["memory_state_bytes_per_session"]) == EXPECTED_STATE_BYTES
    )
    checks["episodic_state_all_finite"] = bool(
        episodic["all_keys_values_strengths_finite"]
    )
    checks["episodic_capacity_exact_48"] = bool(
        episodic["capacity_bounded_exactly_48"]
    )
    checks["episodic_validity_boolean"] = bool(episodic["validity_dtype_boolean"])
    checks["session_isolation_exact"] = bool(episodic["session_isolation_exact"])
    checks["heldout_sparse_write_execution_exact_16_of_255_one_update"] = bool(
        sparse["all_completed_measured_stages_exact_16_of_255_one_update"]
    )
    checks["heldout_payload_sample_count_exact_256"] = (
        int(round(payload["sampled_payload_events"])) == 256
    )
    result["v25_development_pass"] = all(bool(value) for value in checks.values())
    result["claims"] = {
        "development_seed_only": True,
        "counts_toward_independent_replication": False,
        "real_language_memory_advantage_proven_in_development": bool(
            result["v25_development_pass"]
        ),
        "architecture_freeze_boundary_may_be_preregistered": bool(
            result["v25_development_pass"]
        ),
        "architecture_frozen": False,
        "s2_authorized": False,
        "independent_replication_complete": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    Path(run_dir, "result.json").write_text(json.dumps(result, indent=2))
    return result


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"v25 development run is frozen to fresh seed {SEED}")
    _install_v25_binding()

    original_validate = base.validate_protocol
    original_memory = base._memory_suite
    base.validate_protocol = validate_protocol
    base._memory_suite = _v25_memory_suite
    try:
        result = base.train_matched_pair(
            data_dir=data_dir,
            run_dir=run_dir,
            seed=seed,
        )
    finally:
        base.validate_protocol = original_validate
        base._memory_suite = original_memory
    gc.collect()
    torch.cuda.empty_cache()
    return _remap_and_finalize(result, run_dir)


def frozen_protocol_summary() -> dict[str, Any]:
    return {
        "research_issue": 366,
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "memory_eval_seed": MEMORY_EVAL_SEED,
        "systems_eval_seed": SYSTEMS_EVAL_SEED,
        "development_only": True,
        "token_budget_per_model": TOKEN_BUDGET,
        "memory_dim": 50,
        "chunk_size": v25.CHUNK_SIZE,
        "episodic_capacity_per_stage": EPISODIC_CAPACITY,
        "state_bytes_per_session": EXPECTED_STATE_BYTES,
        "candidates_per_chunk": EXPECTED_CANDIDATES,
        "selected_writes_per_chunk": EXPECTED_SELECTED_WRITES,
        "vectorized_updates_per_completed_stage_chunk": EXPECTED_VECTORIZED_UPDATES,
        "payload_token_weight": v25.PAYLOAD_TOKEN_WEIGHT,
        "address_auxiliary_weight": v25.ADDRESS_AUXILIARY_WEIGHT,
        "payload_events_per_microbatch": v25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH,
        "payload_events_per_optimizer_step": (
            v25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH * GRAD_ACCUM
        ),
        "thresholds_identical_to_issue324_v18": True,
        "gpu_authorized_by_module": False,
        "counts_toward_independent_replication": False,
        "100m_authorized": False,
    }
