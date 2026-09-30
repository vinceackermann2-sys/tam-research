from __future__ import annotations

"""CHM-v2 ~100M query-conditioned value adapter Stage A (#1147).

Implementation/preflight only. This module defines the frozen EIEM-QVA-r32
successor, exact parameter accounting, CPU-only engineering construction, and
training/evaluation helpers compatible with the inherited CHM-v1 protocol.

It contains no optimizer, corpus loader, training loop, Modal/GPU launcher,
scientific seed, paid-compute trigger, or Stage-B/Stage-C execution authority.
"""

from dataclasses import asdict
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chm_v1_100m_scale import (
    ADDRESS_DIM,
    D_MODEL,
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    FUTURE_SCREEN_SEED,
    PARAMETER_MISMATCH_LIMIT,
    CHMV1100MEIEMLM,
    CHMV1100MLocalLM,
    transformer_100m_config,
)
from .chm_v1_100m_stage_c_eval import eiem_flat_final_logits
from .chm_v1_small_lm import LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden
from .chm_v1_small_lm_protocol import FLAT_TRAIN_TEMPERATURE, SESSION_LEN
from .models import parameter_count

ISSUE = 1147
PREREGISTERED_MAIN_SHA = "593b24c10268e4c4c40416aa938a3d7de8f46243"
IMPLEMENTATION_BASE_MAIN_SHA = "db300b7748fa56500b2f53a9f85f2bf1b7bbc5c8"
IMPLEMENTATION_BASE_TREE_SHA = "4aaed1af001e4cf90cd3238910367532e89e584a"

QVA_RANK = 32
QVA_GATE_INIT = -4.0
ENGINEERING_SEED = 1_147_099

EXPECTED_QVA_ADDED_PARAMETERS = 49_185
EXPECTED_QVA_PARAMETERS = 101_885_985
EXPECTED_QVA_MINUS_LOCAL = 82_465

BLOCKED_SCIENTIFIC_SEEDS = (977_001, FUTURE_SCREEN_SEED)


class QueryConditionedValueAdapter(nn.Module):
    """Low-rank query-conditioned residual transform frozen by #1147.

    h' = h + sigmoid(g0 + w_g^T SiLU(W_q LN(h) + W_v LN(m)))
             * W_o SiLU(W_q LN(h) + W_v LN(m))

    W_o is exactly zero-initialized, making the adapter an exact no-op at
    construction even though the query/value/gate path is present.
    """

    def __init__(self, d_model: int, rank: int = QVA_RANK) -> None:
        super().__init__()
        d_model = int(d_model)
        rank = int(rank)
        if d_model <= 0 or rank <= 0:
            raise ValueError("d_model and rank must be positive")
        self.d_model = d_model
        self.rank = rank

        self.hidden_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.query_projection = nn.Linear(d_model, rank, bias=False)
        self.value_projection = nn.Linear(d_model, rank, bias=False)
        self.output_projection = nn.Linear(rank, d_model, bias=False)
        self.gate_projection = nn.Linear(rank, 1, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(float(QVA_GATE_INIT)))

        nn.init.zeros_(self.output_projection.weight)

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if hidden.shape != memory.shape:
            raise ValueError(
                f"QVA hidden/memory shape mismatch: {tuple(hidden.shape)} != {tuple(memory.shape)}"
            )
        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"QVA last dimension must be {self.d_model}, got {hidden.shape[-1]}"
            )

        q = self.query_projection(self.hidden_norm(hidden))
        v = self.value_projection(self.memory_norm(memory))
        z = F.silu(q + v)
        delta = self.output_projection(z)
        gate = torch.sigmoid(self.gate_bias + self.gate_projection(z))
        return hidden + gate.to(hidden.dtype) * delta.to(hidden.dtype)


class CHMV2100MEIEMQVA(CHMV1100MEIEMLM):
    """CHM-v1 EIEM with only the frozen #1147 integration rule replaced.

    The inherited CHM-v1 memory_gate_logit parameter is deliberately retained
    for exact provenance/parameter accounting but is inert under QVA.
    Addressing, backbone, memory writes, hop count, and retrieval semantics
    remain unchanged.
    """

    def __init__(self, *, address_dim: int = ADDRESS_DIM, rank: int = QVA_RANK) -> None:
        if int(rank) != QVA_RANK:
            raise RuntimeError(f"#1147 rank is frozen at {QVA_RANK}")
        super().__init__(address_dim=address_dim)
        self.qva = QueryConditionedValueAdapter(D_MODEL, QVA_RANK)

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.qva(hidden, memory)


def validate_engineering_seed(seed: int) -> int:
    seed = int(seed)
    if seed in BLOCKED_SCIENTIFIC_SEEDS:
        raise RuntimeError("CHM-v1 scientific seed is historical/reserved and unavailable to #1147")
    if seed != ENGINEERING_SEED:
        raise RuntimeError(f"#1147 Stage A accepts only engineering seed {ENGINEERING_SEED}")
    return seed


def qva_added_parameter_count(*, d_model: int = D_MODEL, rank: int = QVA_RANK) -> int:
    d_model = int(d_model)
    rank = int(rank)
    return 3 * d_model * rank + rank + 1


def analytical_parameter_accounting() -> dict[str, int | float | bool]:
    added = qva_added_parameter_count()
    qva = EXPECTED_EIEM_PARAMETERS + added
    delta = (qva - EXPECTED_LOCAL_PARAMETERS) / EXPECTED_LOCAL_PARAMETERS
    return {
        "local_trainable_parameters": EXPECTED_LOCAL_PARAMETERS,
        "chm_v1_eiem_trainable_parameters": EXPECTED_EIEM_PARAMETERS,
        "qva_added_parameters": added,
        "qva_trainable_parameters": qva,
        "qva_minus_local_parameters": qva - EXPECTED_LOCAL_PARAMETERS,
        "delta_fraction_vs_local": delta,
        "within_point_one_percent": abs(delta) <= PARAMETER_MISMATCH_LIMIT,
    }


def instantiated_parameter_accounting() -> dict[str, int | float | bool]:
    model = CHMV2100MEIEMQVA()
    qva = parameter_count(model)
    delta = (qva - EXPECTED_LOCAL_PARAMETERS) / EXPECTED_LOCAL_PARAMETERS
    result: dict[str, int | float | bool] = {
        "local_trainable_parameters": EXPECTED_LOCAL_PARAMETERS,
        "chm_v1_eiem_trainable_parameters": EXPECTED_EIEM_PARAMETERS,
        "qva_added_parameters": qva - EXPECTED_EIEM_PARAMETERS,
        "qva_trainable_parameters": qva,
        "qva_minus_local_parameters": qva - EXPECTED_LOCAL_PARAMETERS,
        "delta_fraction_vs_local": delta,
        "within_point_one_percent": abs(delta) <= PARAMETER_MISMATCH_LIMIT,
    }
    del model
    return result


def build_engineering_triplet(
    *,
    seed: int = ENGINEERING_SEED,
    device: torch.device | None = None,
) -> tuple[CHMV1100MLocalLM, CHMV1100MEIEMLM, CHMV2100MEIEMQVA]:
    """CPU-only matched initialization for Stage-A engineering checks."""
    validate_engineering_seed(seed)
    device = torch.device("cpu") if device is None else torch.device(device)
    if device.type != "cpu":
        raise RuntimeError("#1147 Stage A is CPU-only")

    torch.manual_seed(seed)
    local = CHMV1100MLocalLM().to(device)
    torch.manual_seed(seed)
    v1 = CHMV1100MEIEMLM().to(device)
    torch.manual_seed(seed)
    qva = CHMV2100MEIEMQVA().to(device)

    local_backbone = local.backbone.state_dict()
    v1_backbone = v1.backbone.state_dict()
    qva_backbone = qva.backbone.state_dict()
    if local_backbone.keys() != v1_backbone.keys() or local_backbone.keys() != qva_backbone.keys():
        raise RuntimeError("#1147 paired backbone state-key drift")
    for name, local_value in local_backbone.items():
        if not torch.equal(local_value, v1_backbone[name]):
            raise RuntimeError(f"#1147 LOCAL/v1 backbone init mismatch at {name}")
        if not torch.equal(local_value, qva_backbone[name]):
            raise RuntimeError(f"#1147 LOCAL/QVA backbone init mismatch at {name}")

    v1_named = dict(v1.named_parameters())
    qva_named = dict(qva.named_parameters())
    for name in ("query_address.weight", "key_address.weight", "memory_gate_logit"):
        if not torch.equal(v1_named[name], qva_named[name]):
            raise RuntimeError(f"#1147 inherited EIEM parameter init mismatch at {name}")

    return local, v1, qva


def _chunked(tokens: torch.Tensor):
    if tokens.ndim != 2 or tokens.shape[1] != SESSION_LEN:
        raise ValueError(f"#1147 expected [batch,{SESSION_LEN}] contiguous sessions")
    for start in range(0, SESSION_LEN, LOCAL_WINDOW):
        yield tokens[:, start : start + LOCAL_WINDOW]


def qva_flat_training_session_logits(
    model: CHMV2100MEIEMQVA,
    tokens: torch.Tensor,
    *,
    temperature: float = FLAT_TRAIN_TEMPERATURE,
) -> torch.Tensor:
    """Differentiable frozen two-hop soft retrieval with QVA integration only."""
    if float(temperature) != FLAT_TRAIN_TEMPERATURE:
        raise RuntimeError(
            f"#1147 training retrieval temperature is frozen at {FLAT_TRAIN_TEMPERATURE}"
        )
    if RETRIEVAL_HOPS != 2:
        raise RuntimeError("#1147 inherits exactly two retrieval hops")

    memory_keys: torch.Tensor | None = None
    memory_values: torch.Tensor | None = None
    logits: list[torch.Tensor] = []

    for chunk in _chunked(tokens):
        hidden = _hidden(model.backbone, chunk)
        keys = model.key_for(hidden)
        query_state = hidden
        if memory_keys is not None:
            assert memory_values is not None
            for _ in range(RETRIEVAL_HOPS):
                queries = model.query_for(query_state)
                score = torch.einsum("btd,bsd->bts", queries, memory_keys) / FLAT_TRAIN_TEMPERATURE
                weights = torch.softmax(score.float(), dim=-1).to(hidden.dtype)
                memory = torch.matmul(weights, memory_values)
                query_state = model._integrate(query_state, memory)
        logits.append(model.backbone.lm_head(query_state))
        memory_keys = keys if memory_keys is None else torch.cat((memory_keys, keys), dim=1)
        memory_values = hidden if memory_values is None else torch.cat((memory_values, hidden), dim=1)

    return torch.cat(logits, dim=1)


@torch.no_grad()
def qva_hard_flat_final_logits(
    model: CHMV2100MEIEMQVA,
    prompt_ids: Sequence[int],
) -> torch.Tensor:
    """Use the exact #986 hard-flat evaluator with QVA integration."""
    return eiem_flat_final_logits(model, prompt_ids)


def stage_a_preflight(*, instantiate: bool = False) -> dict[str, Any]:
    if D_MODEL != 512 or QVA_RANK != 32:
        raise RuntimeError("#1147 QVA geometry drift")
    if SESSION_LEN != 2 * LOCAL_WINDOW:
        raise RuntimeError("#1147 session/chunk geometry drift")
    if RETRIEVAL_HOPS != 2:
        raise RuntimeError("#1147 hop-count drift")
    if FLAT_TRAIN_TEMPERATURE != 0.10:
        raise RuntimeError("#1147 inherited soft-training temperature drift")

    analytical = analytical_parameter_accounting()
    if int(analytical["qva_added_parameters"]) != EXPECTED_QVA_ADDED_PARAMETERS:
        raise RuntimeError(f"#1147 QVA added-parameter drift: {analytical}")
    if int(analytical["qva_trainable_parameters"]) != EXPECTED_QVA_PARAMETERS:
        raise RuntimeError(f"#1147 QVA total-parameter drift: {analytical}")
    if int(analytical["qva_minus_local_parameters"]) != EXPECTED_QVA_MINUS_LOCAL:
        raise RuntimeError(f"#1147 QVA-vs-LOCAL parameter drift: {analytical}")
    if not bool(analytical["within_point_one_percent"]):
        raise RuntimeError(f"#1147 parameter fairness gate failed: {analytical}")

    instantiated = instantiated_parameter_accounting() if instantiate else None
    if instantiated is not None and instantiated != analytical:
        raise RuntimeError(
            f"#1147 analytical/instantiated accounting mismatch: {analytical} != {instantiated}"
        )

    return {
        "classification": "CHM_V2_100M_QVA_STAGE_A_IMPLEMENTATION_PREFLIGHT_PASS",
        "issue": ISSUE,
        "preregistered_main_sha": PREREGISTERED_MAIN_SHA,
        "implementation_base_main_sha": IMPLEMENTATION_BASE_MAIN_SHA,
        "implementation_base_tree_sha": IMPLEMENTATION_BASE_TREE_SHA,
        "configuration": asdict(transformer_100m_config()),
        "qva_rank": QVA_RANK,
        "qva_gate_init": QVA_GATE_INIT,
        "engineering_seed": ENGINEERING_SEED,
        "address_dim": ADDRESS_DIM,
        "local_window": LOCAL_WINDOW,
        "retrieval_hops": RETRIEVAL_HOPS,
        "soft_training_temperature": FLAT_TRAIN_TEMPERATURE,
        "legacy_memory_gate_retained_but_inert": True,
        "output_projection_zero_initialized": True,
        "parameter_accounting": analytical,
        "instantiated_parameter_accounting": instantiated,
        "gpu_authorized": False,
        "modal_authorized": False,
        "paid_compute_authorized": False,
        "scientific_training_authorized": False,
        "scientific_seed_authorized": False,
        "stage_b_authorized": False,
        "stage_c_authorized": False,
        "interpretation_ceiling": (
            "Stage-A implementation evidence only; no modeling result, scaling claim, or breakthrough claim."
        ),
    }
