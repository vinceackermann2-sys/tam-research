from __future__ import annotations

"""Guarded AERA-v22 real-language memory development harness.

The trainer/evaluators are inherited from the frozen v18/v19 ~25M protocol.  V22
changes the architecture binding and memory teaching objective only as frozen in
issue #324.  Seed8391 is development-only and can never count as replication.
"""

import gc
import json
from pathlib import Path
from typing import Any

import torch

from . import aera_real_language_v18_gpu as base
from . import aera_real_language_v22 as v22
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v22 import DualDeltaFastMemoryState
from .aera_real_language import SEQ_LEN, TOTAL_STEPS
from .data import TokenBin

SEED = 8391
EVAL_SEED = 98_391
MEMORY_EVAL_SEED = 108_391
SYSTEMS_EVAL_SEED = 118_391
P_SYMMETRY_MAX_ABS = 1e-4

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
SYSTEM_BATCH_SIZES = base.SYSTEM_BATCH_SIZES
MEMORY_EVAL_BATCHES = base.MEMORY_EVAL_BATCHES
MEMORY_EVAL_BATCH_SIZE = base.MEMORY_EVAL_BATCH_SIZE

_ORIGINAL_VALIDATE_PROTOCOL = base.validate_protocol
_ORIGINAL_MEMORY_SUITE = base._memory_suite


def _install_v22_binding() -> None:
    base.v18 = v22
    base.SEED = SEED
    base.EVAL_SEED = EVAL_SEED
    base.MEMORY_EVAL_SEED = MEMORY_EVAL_SEED
    base.SYSTEMS_EVAL_SEED = SYSTEMS_EVAL_SEED


def _decorate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("development_seed") != SEED:
        raise RuntimeError("v22 base protocol did not pick up frozen seed")
    protocol["version"] = "aera-v22-dual-delta-real-language-development"
    protocol["gpu_authorization_scope"] = (
        "one guarded AERA-v22 development seed8391 L4 run only"
    )
    protocol["counts_toward_independent_replication"] = False
    protocol["architecture"] = {
        "source": "merged AERA-v22 controlled PASS",
        "memory_dim": 50,
        "memory_dim_changed_from_v19_real_language": False,
        "event_pair_writes": True,
        "tokenwise_prior_state_reads": True,
        "inverse_key_covariance_state": True,
        "covariance_preconditioned_dual_delta": True,
        "blanket_matrix_decay": False,
        "routing_changed_from_v19": False,
        "predictive_stream_changed_from_v19": False,
    }
    protocol["memory_training_objective"] = {
        "address_contrastive_weight": v22.ADDRESS_CONTRASTIVE_WEIGHT,
        "address_temperature": v22.ADDRESS_TEMPERATURE,
        "payload_token_weight": v22.PAYLOAD_TOKEN_WEIGHT,
        "latent_payload_weight": v22.LATENT_PAYLOAD_WEIGHT,
        "max_sampled_adjacent_events_per_step": v22.MAX_MEMORY_AUX_EVENTS,
        "sampling": "deterministic stratified plus step-dependent cyclic offset",
        "address_identity": "observed GPT-2 token id + local chunk position",
        "event_representations_detached": True,
        "decoder_weights_detached": True,
    }
    protocol["thresholds_inherited_unchanged_from_v18_seed8351"] = {
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
    protocol["v22_specific_safety"] = {
        "inverse_key_covariance_finite_required": True,
        "inverse_key_covariance_symmetry_max_abs": P_SYMMETRY_MAX_ABS,
        "session_isolation_required": True,
        "memory_state_bytes_include_M_and_P": True,
    }
    return protocol


def validate_protocol(data_dir: str) -> dict[str, Any]:
    _install_v22_binding()
    return _decorate_protocol(_ORIGINAL_VALIDATE_PROTOCOL(data_dir))


def _session_isolation_probe(aera, device: torch.device) -> bool:
    g = torch.Generator(device="cpu").manual_seed(MEMORY_EVAL_SEED + 77)
    tokens = torch.randint(0, aera.cfg.vocab_size, (1, v22.CHUNK_SIZE), generator=g).to(device)
    first = aera.empty_state(tokens)
    second = aera.empty_state(tokens)
    for a, b in zip(first.stages, second.stages):
        if not isinstance(a.memory, DualDeltaFastMemoryState) or not isinstance(b.memory, DualDeltaFastMemoryState):
            return False
        if a.memory.matrix.data_ptr() == b.memory.matrix.data_ptr():
            return False
        if a.memory.inverse_key_covariance.data_ptr() == b.memory.inverse_key_covariance.data_ptr():
            return False
    # Mutating one detached session state must not affect the other or parameters.
    with torch.no_grad():
        first.stages[0].memory.matrix.add_(1.0)
        first.stages[0].memory.inverse_key_covariance.mul_(0.5)
    return bool(
        second.stages[0].memory.matrix.eq(0).all()
        and torch.equal(
            second.stages[0].memory.inverse_key_covariance,
            torch.eye(aera.cfg.memory_dim, device=device, dtype=second.stages[0].memory.inverse_key_covariance.dtype)
            .unsqueeze(0),
        )
    )


@torch.no_grad()
def _v22_state_and_aux_audit(*, data_dir: str, run_dir: str, seed: int) -> dict[str, Any]:
    device = torch.device("cuda")
    payload = torch.load(Path(run_dir) / "aera.pt", map_location="cpu", weights_only=False)
    if payload.get("seed") != seed:
        raise RuntimeError("v22 state audit checkpoint seed mismatch")
    torch.manual_seed(seed)
    aera = v22.build_aera(device).eval()
    aera.load_state_dict(payload["model"], strict=True)
    aera.set_memory_pretraining_mode(False)

    parameter_versions_before = [p._version for p in aera.parameters()]
    val = TokenBin(str(Path(data_dir) / "val.bin"))
    g = torch.Generator(device="cpu").manual_seed(MEMORY_EVAL_SEED + 91)
    x, _ = val.batch(MEMORY_EVAL_BATCH_SIZE, SEQ_LEN, g, device)
    with base._autocast(device):
        out = aera(
            x,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
        )
        aux = v22.memory_auxiliary_terms(
            aera,
            x,
            step=TOTAL_STEPS - 1,
            max_events=v22.MAX_MEMORY_AUX_EVENTS,
        )
    state = out.get("state")
    if not isinstance(state, HardwareAERAState):
        raise RuntimeError("v22 state audit missing HardwareAERAState")

    m_norms: list[torch.Tensor] = []
    p_norms: list[torch.Tensor] = []
    p_min_eigs: list[torch.Tensor] = []
    max_symmetry_error = 0.0
    all_finite = True
    matrix_bytes = 0
    covariance_bytes = 0
    for stage_state in state.stages:
        memory = stage_state.memory
        if not isinstance(memory, DualDeltaFastMemoryState):
            raise RuntimeError("v22 state audit lost DualDeltaFastMemoryState")
        m = memory.matrix.float()
        p = memory.inverse_key_covariance.float()
        all_finite = all_finite and bool(torch.isfinite(m).all()) and bool(torch.isfinite(p).all())
        m_norms.append(torch.linalg.vector_norm(m.reshape(m.size(0), -1), dim=1).cpu())
        p_norms.append(torch.linalg.vector_norm(p.reshape(p.size(0), -1), dim=1).cpu())
        symmetry = (p - p.transpose(-1, -2)).abs().amax()
        max_symmetry_error = max(max_symmetry_error, float(symmetry))
        p_min_eigs.append(torch.linalg.eigvalsh(0.5 * (p + p.transpose(-1, -2)))[:, 0].cpu())
        matrix_bytes += memory.matrix.numel() * memory.matrix.element_size()
        covariance_bytes += memory.inverse_key_covariance.numel() * memory.inverse_key_covariance.element_size()

    parameter_versions_after = [p._version for p in aera.parameters()]
    total_bytes = matrix_bytes + covariance_bytes
    per_session_matrix = matrix_bytes // MEMORY_EVAL_BATCH_SIZE
    per_session_covariance = covariance_bytes // MEMORY_EVAL_BATCH_SIZE
    per_session_total = total_bytes // MEMORY_EVAL_BATCH_SIZE
    isolation = _session_isolation_probe(aera, device)

    return {
        "mean_final_memory_frobenius_norm": float(torch.cat(m_norms).mean()),
        "mean_inverse_key_covariance_frobenius_norm": float(torch.cat(p_norms).mean()),
        "min_inverse_key_covariance_eigenvalue": float(torch.cat(p_min_eigs).min()),
        "inverse_key_covariance_max_symmetry_error": max_symmetry_error,
        "inverse_key_covariance_all_finite": all_finite,
        "matrix_state_bytes_per_session": int(per_session_matrix),
        "inverse_covariance_state_bytes_per_session": int(per_session_covariance),
        "total_memory_state_bytes_per_session": int(per_session_total),
        "session_isolation_exact": isolation,
        "deployment_base_parameter_versions_unchanged": parameter_versions_before == parameter_versions_after,
        "heldout_sampled_memory_auxiliary": {
            key: float(value.detach().float().cpu())
            for key, value in aux.items()
        },
    }


def _v22_memory_suite(*, data_dir: str, run_dir: str, seed: int) -> dict[str, Any]:
    inherited = _ORIGINAL_MEMORY_SUITE(data_dir=data_dir, run_dir=run_dir, seed=seed)
    supplemental = _v22_state_and_aux_audit(data_dir=data_dir, run_dir=run_dir, seed=seed)
    # The inherited v18 metric counted M only. Preserve that value explicitly, then
    # replace the public total with honest M+P state bytes for v22.
    inherited["matrix_state_bytes_per_session"] = int(
        supplemental["matrix_state_bytes_per_session"]
    )
    inherited["inverse_covariance_state_bytes_per_session"] = int(
        supplemental["inverse_covariance_state_bytes_per_session"]
    )
    inherited["memory_state_bytes_per_session"] = int(
        supplemental["total_memory_state_bytes_per_session"]
    )
    inherited["mean_final_memory_frobenius_norm"] = float(
        supplemental["mean_final_memory_frobenius_norm"]
    )
    inherited["inverse_key_covariance"] = {
        "mean_frobenius_norm": supplemental["mean_inverse_key_covariance_frobenius_norm"],
        "min_eigenvalue": supplemental["min_inverse_key_covariance_eigenvalue"],
        "max_symmetry_error": supplemental["inverse_key_covariance_max_symmetry_error"],
        "all_finite": supplemental["inverse_key_covariance_all_finite"],
    }
    inherited["session_isolation_exact"] = bool(supplemental["session_isolation_exact"])
    inherited["heldout_sampled_memory_auxiliary"] = supplemental[
        "heldout_sampled_memory_auxiliary"
    ]
    inherited["deployment_base_parameter_versions_unchanged"] = bool(
        inherited["deployment_base_parameter_versions_unchanged"]
        and supplemental["deployment_base_parameter_versions_unchanged"]
    )
    return inherited


def _remap_and_finalize(result: dict[str, Any], run_dir: str) -> dict[str, Any]:
    for old, new in (
        ("v18_memory_eval", "v22_memory_eval"),
        ("v18_heldout_adaptivity", "v22_heldout_adaptivity"),
        ("v18_systems_eval", "v22_systems_eval"),
        ("v18_development_checks", "v22_development_checks"),
        ("v18_development_pass", "v22_inherited_development_pass"),
    ):
        if old not in result:
            raise RuntimeError(f"v22 expected inherited result key {old!r}")
        result[new] = result.pop(old)

    protocol = result.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("v22 inherited result missing protocol")
    result["protocol"] = _decorate_protocol(protocol)

    memory = result["v22_memory_eval"]
    checks = result["v22_development_checks"]
    p = memory["inverse_key_covariance"]
    checks["inverse_key_covariance_all_finite"] = bool(p["all_finite"])
    checks["inverse_key_covariance_symmetric"] = (
        float(p["max_symmetry_error"]) <= P_SYMMETRY_MAX_ABS
    )
    checks["session_isolation_exact"] = bool(memory["session_isolation_exact"])
    checks["memory_state_bytes_include_M_and_P"] = (
        memory["memory_state_bytes_per_session"]
        == memory["matrix_state_bytes_per_session"]
        + memory["inverse_covariance_state_bytes_per_session"]
    )
    result["v22_development_pass"] = all(bool(v) for v in checks.values())
    result["claims"] = {
        "development_seed_only": True,
        "counts_toward_independent_replication": False,
        "real_language_memory_advantage_proven_in_development": bool(
            result["v22_development_pass"]
        ),
        "architecture_freeze_authorized": bool(result["v22_development_pass"]),
        "independent_replication_complete": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    Path(run_dir, "result.json").write_text(json.dumps(result, indent=2))
    return result


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"v22 development run is frozen to fresh seed {SEED}")
    _install_v22_binding()

    original_validate = base.validate_protocol
    original_memory = base._memory_suite
    base.validate_protocol = validate_protocol
    base._memory_suite = _v22_memory_suite
    try:
        result = base.train_matched_pair(data_dir=data_dir, run_dir=run_dir, seed=seed)
    finally:
        base.validate_protocol = original_validate
        base._memory_suite = original_memory
    gc.collect()
    torch.cuda.empty_cache()
    return _remap_and_finalize(result, run_dir)


def frozen_protocol_summary() -> dict[str, Any]:
    return {
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "memory_eval_seed": MEMORY_EVAL_SEED,
        "systems_eval_seed": SYSTEMS_EVAL_SEED,
        "development_only": True,
        "memory_dim": 50,
        "thresholds_identical_to_v18": True,
        "p_symmetry_max_abs": P_SYMMETRY_MAX_ABS,
        "gpu_authorized_by_module": False,
    }
