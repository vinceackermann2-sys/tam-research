from __future__ import annotations

"""CHM-v2 value-projected EIEM development protocol (#1112).

Implementation-only surface.  This module defines the preregistered architecture,
paired initialization, projected-value training/evaluation semantics, and pure
development-gate classification.  It contains no optimizer, corpus loader,
checkpoint loader, launcher, Modal path, or scientific execution authority.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from .chm_v1_100m_scale import (
    ADDRESS_DIM,
    D_MODEL,
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    PARAMETER_MISMATCH_LIMIT,
    CHMV1100MEIEMLM,
    CHMV1100MLocalLM,
)
from .chm_v1_100m_stage_c_execution import _exact_nearest_positions
from .chm_v1_small_lm import EpisodicState, LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden
from .chm_v1_small_lm_protocol import FLAT_TRAIN_TEMPERATURE
from .models import parameter_count

ISSUE = 1112
CLASSIFICATION = "CHM_V2_100M_VALUE_PROJECTED_EIEM_PREREGISTRATION_NO_EXECUTION_AUTHORITY"
RESERVED_DEVELOPMENT_SEED = 2_011_121
CONSUMED_CHM_V1_SEED = 977_001

VALUE_RANK = ADDRESS_DIM
EXPECTED_VALUE_PROJECTION_PARAMETERS = D_MODEL * VALUE_RANK + VALUE_RANK + VALUE_RANK * D_MODEL + D_MODEL
EXPECTED_VP_EIEM_PARAMETERS = EXPECTED_EIEM_PARAMETERS + EXPECTED_VALUE_PROJECTION_PARAMETERS
EXPECTED_VP_EXTRA_VS_LOCAL = EXPECTED_VP_EIEM_PARAMETERS - EXPECTED_LOCAL_PARAMETERS
EXPECTED_VP_DELTA_VS_LOCAL = EXPECTED_VP_EXTRA_VS_LOCAL / EXPECTED_LOCAL_PARAMETERS

TRAINING_TOKENS_PER_MODEL = 33_554_432
TOKENS_PER_OPTIMIZER_STEP = 16_384
OPTIMIZER_STEPS_PER_MODEL = 2_048
WARMUP_STEPS = 40

LONG_RANGE_FAMILIES = ("rare_fact", "overwrite", "two_hop")
POSITIVE_CLASSIFICATION = "CHM_V2_100M_VALUE_PROJECTED_EIEM_POSITIVE_DEVELOPMENT_SCREEN"
STOP_CLASSIFICATION = "CHM_V2_100M_VALUE_PROJECTED_EIEM_STOP"


class CHMV2100MValueProjectedEIEMLM(CHMV1100MEIEMLM):
    """100M EIEM with one identity-initialized rank-32 residual value map."""

    def __init__(self, *, value_rank: int = VALUE_RANK) -> None:
        if int(value_rank) != VALUE_RANK:
            raise RuntimeError(f"#1112 value rank is frozen at {VALUE_RANK}")
        super().__init__(address_dim=ADDRESS_DIM)
        self.value_rank = int(value_rank)
        self.value_down = nn.Linear(D_MODEL, self.value_rank, bias=True)
        self.value_up = nn.Linear(self.value_rank, D_MODEL, bias=True)
        # Freeze functional equivalence to RAW-EIEM at initialization.
        nn.init.zeros_(self.value_up.weight)
        nn.init.zeros_(self.value_up.bias)

    def value_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.value_up(self.value_down(hidden))


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "issue": ISSUE,
        "reserved_development_seed": RESERVED_DEVELOPMENT_SEED,
        "reserved_seed_authorized": False,
        "consumed_chm_v1_seed": CONSUMED_CHM_V1_SEED,
        "value_rank": VALUE_RANK,
        "value_transform": "h + up(down(h))",
        "value_nonlinearity": None,
        "up_zero_initialized": True,
        "parameter_mismatch_limit": PARAMETER_MISMATCH_LIMIT,
        "training_tokens_per_model": TRAINING_TOKENS_PER_MODEL,
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "optimizer_steps_per_model": OPTIMIZER_STEPS_PER_MODEL,
        "warmup_steps": WARMUP_STEPS,
        "soft_training_temperature": FLAT_TRAIN_TEMPERATURE,
        "gpu_authorized": False,
        "training_execution_authorized": False,
        "scientific_execution_authorized": False,
        "stage_d_authorized": False,
    }


def validate_engineering_seed(seed: int) -> int:
    value = int(seed)
    if value == RESERVED_DEVELOPMENT_SEED:
        raise RuntimeError("#1112 reserved development seed requires separate run-control authority")
    if value == CONSUMED_CHM_V1_SEED:
        raise RuntimeError("#1112 refuses permanently consumed CHM-v1 seed 977001")
    return value


def analytical_parameter_accounting() -> dict[str, int | float | bool]:
    projection = (
        D_MODEL * VALUE_RANK
        + VALUE_RANK
        + VALUE_RANK * D_MODEL
        + D_MODEL
    )
    vp = EXPECTED_EIEM_PARAMETERS + projection
    delta = (vp - EXPECTED_LOCAL_PARAMETERS) / EXPECTED_LOCAL_PARAMETERS
    return {
        "local_trainable_parameters": EXPECTED_LOCAL_PARAMETERS,
        "raw_eiem_trainable_parameters": EXPECTED_EIEM_PARAMETERS,
        "value_projection_parameters": projection,
        "vp_eiem_trainable_parameters": vp,
        "vp_extra_vs_local": vp - EXPECTED_LOCAL_PARAMETERS,
        "vp_delta_fraction_vs_local": delta,
        "within_point_one_percent": abs(delta) <= PARAMETER_MISMATCH_LIMIT,
    }


def instantiated_parameter_accounting() -> dict[str, int | float | bool]:
    # LOCAL/RAW counts are already exact-frozen by #977. Instantiate only the
    # new VP candidate here to keep CPU/CI peak memory bounded.
    vp = CHMV2100MValueProjectedEIEMLM()
    result = {
        "local_trainable_parameters": EXPECTED_LOCAL_PARAMETERS,
        "raw_eiem_trainable_parameters": EXPECTED_EIEM_PARAMETERS,
        "value_projection_parameters": (
            parameter_count(vp.value_down) + parameter_count(vp.value_up)
        ),
        "vp_eiem_trainable_parameters": parameter_count(vp),
    }
    result["vp_extra_vs_local"] = (
        int(result["vp_eiem_trainable_parameters"])
        - int(result["local_trainable_parameters"])
    )
    result["vp_delta_fraction_vs_local"] = (
        int(result["vp_extra_vs_local"])
        / int(result["local_trainable_parameters"])
    )
    result["within_point_one_percent"] = (
        abs(float(result["vp_delta_fraction_vs_local"])) <= PARAMETER_MISMATCH_LIMIT
    )
    return result


def validate_architecture_contract() -> dict[str, Any]:
    analytical = analytical_parameter_accounting()
    instantiated = instantiated_parameter_accounting()
    if analytical != instantiated:
        raise RuntimeError(
            f"#1112 analytical/instantiated parameter accounting mismatch: "
            f"{analytical} != {instantiated}"
        )
    if int(instantiated["value_projection_parameters"]) != EXPECTED_VALUE_PROJECTION_PARAMETERS:
        raise RuntimeError("#1112 value-projection parameter count drift")
    if int(instantiated["vp_eiem_trainable_parameters"]) != EXPECTED_VP_EIEM_PARAMETERS:
        raise RuntimeError("#1112 VP-EIEM parameter count drift")
    if not bool(instantiated["within_point_one_percent"]):
        raise RuntimeError("#1112 parameter fairness gate failed")
    if VALUE_RANK != ADDRESS_DIM or VALUE_RANK != 32:
        raise RuntimeError("#1112 value rank/address rank drift")
    if TRAINING_TOKENS_PER_MODEL != TOKENS_PER_OPTIMIZER_STEP * OPTIMIZER_STEPS_PER_MODEL:
        raise RuntimeError("#1112 training-token arithmetic drift")
    return {**protocol_manifest(), "parameter_accounting": instantiated}


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _assert_same_state(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if left.keys() != right.keys():
        raise RuntimeError(f"#1112 {label} state keys differ")
    for name in left:
        if not torch.equal(left[name], right[name]):
            raise RuntimeError(f"#1112 {label} initialization mismatch at {name}")


def build_engineering_triplet(
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CHMV1100MLocalLM, CHMV1100MEIEMLM, CHMV2100MValueProjectedEIEMLM]:
    """Build paired LOCAL/RAW/VP models without permitting the reserved seed."""
    seed = validate_engineering_seed(seed)
    _seed_torch(seed)
    local = CHMV1100MLocalLM().to(device)
    _seed_torch(seed)
    raw = CHMV1100MEIEMLM().to(device)
    _seed_torch(seed)
    vp = CHMV2100MValueProjectedEIEMLM().to(device)

    _assert_same_state(local.backbone.state_dict(), raw.backbone.state_dict(), label="LOCAL/RAW backbone")
    _assert_same_state(local.backbone.state_dict(), vp.backbone.state_dict(), label="LOCAL/VP backbone")
    _assert_same_state(raw.query_address.state_dict(), vp.query_address.state_dict(), label="RAW/VP query")
    _assert_same_state(raw.key_address.state_dict(), vp.key_address.state_dict(), label="RAW/VP key")
    if not torch.equal(raw.memory_gate_logit, vp.memory_gate_logit):
        raise RuntimeError("#1112 RAW/VP gate initialization mismatch")
    if torch.count_nonzero(vp.value_up.weight).item() != 0:
        raise RuntimeError("#1112 value_up.weight must start at exact zero")
    if torch.count_nonzero(vp.value_up.bias).item() != 0:
        raise RuntimeError("#1112 value_up.bias must start at exact zero")
    return local, raw, vp


def _chunked(tokens: torch.Tensor):
    if tokens.ndim != 2 or tokens.shape[1] % LOCAL_WINDOW != 0:
        raise ValueError("tokens must be [batch, n*LOCAL_WINDOW]")
    for start in range(0, tokens.shape[1], LOCAL_WINDOW):
        yield tokens[:, start : start + LOCAL_WINDOW]


def vp_eiem_flat_training_session_logits(
    model: CHMV2100MValueProjectedEIEMLM,
    tokens: torch.Tensor,
    *,
    temperature: float = FLAT_TRAIN_TEMPERATURE,
) -> torch.Tensor:
    """Frozen differentiable soft retrieval with projected prior-chunk values."""
    if temperature != FLAT_TRAIN_TEMPERATURE:
        raise RuntimeError(
            f"#1112 training temperature is frozen at {FLAT_TRAIN_TEMPERATURE}"
        )
    memory_keys: torch.Tensor | None = None
    memory_values: torch.Tensor | None = None
    logits: list[torch.Tensor] = []

    for chunk in _chunked(tokens):
        hidden = _hidden(model.backbone, chunk)
        keys = model.key_for(hidden)
        values = model.value_for(hidden)
        query_state = hidden
        if memory_keys is not None:
            assert memory_values is not None
            for _ in range(RETRIEVAL_HOPS):
                queries = model.query_for(query_state)
                scores = (
                    torch.einsum("btd,bsd->bts", queries, memory_keys)
                    / FLAT_TRAIN_TEMPERATURE
                )
                weights = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
                memory = torch.matmul(weights, memory_values)
                query_state = model._integrate(query_state, memory)
        logits.append(model.backbone.lm_head(query_state))
        # Write only after current-chunk logits: no current/future episodic access.
        memory_keys = keys if memory_keys is None else torch.cat((memory_keys, keys), dim=1)
        memory_values = (
            values if memory_values is None else torch.cat((memory_values, values), dim=1)
        )
    return torch.cat(logits, dim=1)


@torch.no_grad()
def vp_eiem_exact_flat_two_chunk_logits(
    model: CHMV2100MValueProjectedEIEMLM,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Exact two-chunk validation scorer with projected stored values."""
    if tokens.ndim != 2 or tokens.shape[1] != 2 * LOCAL_WINDOW:
        raise ValueError(f"expected [batch,{2 * LOCAL_WINDOW}] sessions")
    first = tokens[:, :LOCAL_WINDOW]
    second = tokens[:, LOCAL_WINDOW:]

    first_hidden = _hidden(model.backbone, first)
    memory_keys = model.key_for(first_hidden).float()
    memory_values = model.value_for(first_hidden).float()
    first_logits = model.backbone.lm_head(first_hidden)

    second_hidden = _hidden(model.backbone, second)
    query_state = second_hidden
    for _ in range(RETRIEVAL_HOPS):
        queries = model.query_for(query_state)
        positions = _exact_nearest_positions(memory_keys, queries)
        gather_index = positions.unsqueeze(-1).expand(
            positions.shape[0], positions.shape[1], memory_values.shape[-1]
        )
        memory = torch.gather(memory_values, dim=1, index=gather_index)
        query_state = model._integrate(query_state, memory.to(query_state.dtype))

    second_logits = model.backbone.lm_head(query_state)
    return torch.cat((first_logits, second_logits), dim=1)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def vp_eiem_flat_final_logits(
    model: CHMV2100MValueProjectedEIEMLM,
    prompt_ids: Sequence[int],
) -> torch.Tensor:
    """Aligned-v4 final-token hard-flat scorer with projected prior-chunk values."""
    ids = tuple(int(token) for token in prompt_ids)
    if not ids:
        raise ValueError("prompt_ids must be non-empty")
    device = _model_device(model)
    state = EpisodicState("chm-v2-1112-final-token")

    for start in range(0, len(ids), LOCAL_WINDOW):
        chunk_ids = ids[start : start + LOCAL_WINDOW]
        tokens = torch.tensor(chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        hidden = _hidden(model.backbone, tokens)
        is_final = start + len(chunk_ids) == len(ids)
        if not is_final:
            keys = model.key_for(hidden)
            values = model.value_for(hidden)
            state.write(keys[0], values[0])
            continue

        query_state = hidden[0, -1]
        if len(state):
            for _ in range(RETRIEVAL_HOPS):
                query = model.query_for(query_state)
                value, _, _, _, _, _ = state.retrieve(
                    query,
                    mode="flat",
                    verify_indexed_exactness=False,
                )
                query_state = model._integrate(query_state, value)

        fused = hidden.clone()
        fused[0, -1] = query_state
        return model.backbone.lm_head(fused)[0, -1]
    raise RuntimeError("#1112 final VP-EIEM chunk was not reached")


def projection_correction_ratio(
    model: CHMV2100MValueProjectedEIEMLM,
    hidden: torch.Tensor,
) -> torch.Tensor:
    projected = model.value_for(hidden)
    correction = torch.linalg.vector_norm(projected - hidden, dim=-1)
    base = torch.linalg.vector_norm(hidden, dim=-1).clamp_min(1e-12)
    return correction / base


def classify_development_gate(
    *,
    integrity: Mapping[str, bool],
    aggregate: Mapping[str, float],
    families: Mapping[str, Mapping[str, float]],
    ordinary_vp_minus_local_nll: float,
    local_negative_accuracy_gain: float,
    local_negative_nll_regression: float,
    local_overwrite_stale_rate: float,
    vp_overwrite_stale_rate: float,
) -> dict[str, Any]:
    """Pure implementation of the #1112 one-seed development gate."""
    failures: list[str] = []

    required_integrity = (
        "parameter_fairness",
        "paired_initialization",
        "byte_identical_training_stream",
        "exact_training_tokens",
        "matched_schedule",
        "finite_losses",
        "finite_parameters",
        "no_future_leakage",
        "no_cross_session_aliasing",
    )
    for key in required_integrity:
        if integrity.get(key) is not True:
            failures.append(f"integrity:{key}")

    if float(aggregate["vp_vs_local_accuracy_gain"]) < 0.05:
        failures.append("vp_vs_local_accuracy_gain")
    if float(aggregate["vp_vs_local_nll_benefit"]) < 0.05:
        failures.append("vp_vs_local_nll_benefit")

    accuracy_family_wins = 0
    positive_nll_families = 0
    vp_not_worse_raw_nll = 0
    for family in LONG_RANGE_FAMILIES:
        row = families[family]
        acc = float(row["vp_vs_local_accuracy_gain"])
        nll = float(row["vp_vs_local_nll_benefit"])
        raw_delta = float(row["vp_vs_raw_nll_benefit"])
        if acc >= 0.03:
            accuracy_family_wins += 1
        if acc < -0.01:
            failures.append(f"{family}:vp_accuracy_harm")
        if nll > 0.0:
            positive_nll_families += 1
        if raw_delta >= 0.0:
            vp_not_worse_raw_nll += 1

    if accuracy_family_wins < 2:
        failures.append("long_range_accuracy_family_wins")
    if positive_nll_families < 2:
        failures.append("long_range_positive_nll_families")

    if float(aggregate["vp_vs_raw_nll_benefit"]) < 0.05:
        failures.append("vp_vs_raw_nll_benefit")
    if float(aggregate["vp_vs_raw_accuracy_gain"]) <= 0.0:
        failures.append("vp_vs_raw_accuracy_gain")
    if vp_not_worse_raw_nll < 2:
        failures.append("vp_vs_raw_family_nll")

    if float(ordinary_vp_minus_local_nll) > 0.03:
        failures.append("ordinary_language_nll")
    if float(local_negative_accuracy_gain) < -0.02:
        failures.append("local_negative_accuracy")
    if float(local_negative_nll_regression) > 0.03:
        failures.append("local_negative_nll")

    local_stale = float(local_overwrite_stale_rate)
    vp_stale = float(vp_overwrite_stale_rate)
    stale_ok = (
        vp_stale <= local_stale + 0.01
        if local_stale < 0.01
        else vp_stale <= 0.80 * local_stale
    )
    if not stale_ok:
        failures.append("overwrite_stale_guard")

    passed = not failures
    return {
        "classification": POSITIVE_CLASSIFICATION if passed else STOP_CLASSIFICATION,
        "passed": passed,
        "stop_reasons": failures,
        "stage_d_authorized": False,
        "multi_seed_replication_authorized": False,
        "scale_up_authorized": False,
        "interpretation_ceiling": (
            "One-seed 100M development evidence only; no replication, scale, novelty, "
            "SOTA, AGI, RSI, or breakthrough claim."
        ),
    }
