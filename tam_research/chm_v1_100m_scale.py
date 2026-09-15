from __future__ import annotations

"""CHM-v1 ~100M Stage-A matched-model engineering harness (#977).

This module is intentionally implementation/preflight only. It defines the
frozen ~100M LOCAL/EIEM-FLAT model geometry, verifies exact trainable parameter
counts and paired backbone initialization, and freezes the first-screen token
budget. It contains no corpus loader, optimizer, training loop, GPU launcher,
Modal path, or scientific-seed execution authority.
"""

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chm_v1_small_lm import ADDRESS_DIM, LOCAL_WINDOW, _hidden
from .models import ModelConfig, ResearchLM, parameter_count

ISSUE = 977
STARTING_MAIN_SHA = "c2fe1575cedfab4d489476d24249bd93f0faf975"

VOCAB_SIZE = 50_257
D_MODEL = 512
N_LAYERS = 24
N_HEADS = 8
MAX_SEQ_LEN = 1_024
FF_MULT = 4
HEAD_DIM = D_MODEL // N_HEADS

PRIOR_25M_TOKEN_BUDGET = 8_388_608
FIRST_SCREEN_TOKEN_BUDGET = 33_554_432
CONFIRMATORY_CANDIDATE_TOKEN_BUDGET = 268_435_456

ENGINEERING_SMOKE_SEED = 977_099
FUTURE_SCREEN_SEED = 977_001
BLOCKED_PRIOR_SEEDS = (
    19_591,
    19_592,
    19_593,
    971_001,
    971_002,
    973_001,
    973_002,
)

EXPECTED_LOCAL_PARAMETERS = 101_803_520
EXPECTED_EIEM_PARAMETERS = 101_836_800
EXPECTED_EIEM_EXTRA_PARAMETERS = 33_280
PARAMETER_MISMATCH_LIMIT = 0.001  # <=0.1%


def transformer_100m_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        ff_mult=FF_MULT,
        architecture="transformer",
    )


class CHMV1100MLocalLM(nn.Module):
    """~100M bounded-local control using the unchanged ResearchLM blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResearchLM(transformer_100m_config())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = _hidden(self.backbone, tokens)
        return self.backbone.lm_head(hidden)


class CHMV1100MEIEMLM(nn.Module):
    """Parameter-matched ~100M EIEM model for future flat-memory training.

    Stage A defines only the trainable model structure and the differentiable
    address/integration primitives. Scientific training/evaluation must live in
    a separately reviewed protocol and remains unauthorized by #977.
    """

    def __init__(self, *, address_dim: int = ADDRESS_DIM) -> None:
        super().__init__()
        cfg = transformer_100m_config()
        self.backbone = ResearchLM(cfg)
        self.address_dim = int(address_dim)
        self.query_address = nn.Linear(cfg.d_model, self.address_dim, bias=False)
        self.key_address = nn.Linear(cfg.d_model, self.address_dim, bias=False)
        self.memory_gate_logit = nn.Parameter(torch.full((cfg.d_model,), -4.0))

    def query_for(self, representation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(representation), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.memory_gate_logit).to(hidden.dtype)
        return hidden + gate * memory

    def forward_local(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = _hidden(self.backbone, tokens)
        return self.backbone.lm_head(hidden)


def validate_engineering_seed(seed: int) -> int:
    seed = int(seed)
    if seed in BLOCKED_PRIOR_SEEDS:
        raise RuntimeError(f"prior/blocked CHM seed refused by #977 Stage A: {seed}")
    if seed == FUTURE_SCREEN_SEED:
        raise RuntimeError("future #977 scientific screen seed is reserved and unauthorized")
    if seed != ENGINEERING_SMOKE_SEED:
        raise RuntimeError(f"Stage A accepts only engineering smoke seed {ENGINEERING_SMOKE_SEED}")
    return seed


def _analytical_local_parameter_count(cfg: ModelConfig) -> int:
    # Tied lm_head/token_emb weight is counted once, matching nn.Module.parameters().
    embedding = cfg.vocab_size * cfg.d_model + cfg.max_seq_len * cfg.d_model
    per_block = (
        4 * cfg.d_model * cfg.d_model  # qkv + attention output
        + 2 * cfg.ff_mult * cfg.d_model * cfg.d_model  # two bias-free FF matrices
        + 4 * cfg.d_model  # two LayerNorm weight+bias pairs
    )
    final_norm = 2 * cfg.d_model
    return embedding + cfg.n_layers * per_block + final_norm


def analytical_parameter_accounting() -> dict[str, int | float | bool]:
    cfg = transformer_100m_config()
    local = _analytical_local_parameter_count(cfg)
    extra = 2 * cfg.d_model * ADDRESS_DIM + cfg.d_model
    eiem = local + extra
    delta = (eiem - local) / local
    return {
        "local_trainable_parameters": local,
        "eiem_trainable_parameters": eiem,
        "eiem_extra_parameters": extra,
        "delta_fraction": delta,
        "within_point_one_percent": abs(delta) <= PARAMETER_MISMATCH_LIMIT,
    }


def build_engineering_pair(
    *, seed: int = ENGINEERING_SMOKE_SEED,
    device: torch.device | None = None,
) -> tuple[CHMV1100MLocalLM, CHMV1100MEIEMLM]:
    """Instantiate the exact Stage-A pair while refusing scientific seeds."""
    validate_engineering_seed(seed)
    device = torch.device("cpu") if device is None else device
    if device.type != "cpu":
        raise RuntimeError("#977 Stage A is CPU-only")

    torch.manual_seed(seed)
    local = CHMV1100MLocalLM().to(device)
    torch.manual_seed(seed)
    eiem = CHMV1100MEIEMLM().to(device)

    local_state = local.backbone.state_dict()
    eiem_state = eiem.backbone.state_dict()
    if local_state.keys() != eiem_state.keys():
        raise RuntimeError("paired 100M backbone state keys differ")
    for name, local_value in local_state.items():
        if not torch.equal(local_value, eiem_state[name]):
            raise RuntimeError(f"paired 100M backbone initialization mismatch at {name}")
    return local, eiem


def instantiated_parameter_accounting() -> dict[str, int | float | bool]:
    local, eiem = build_engineering_pair()
    local_count = parameter_count(local)
    eiem_count = parameter_count(eiem)
    delta = (eiem_count - local_count) / local_count
    result: dict[str, int | float | bool] = {
        "local_trainable_parameters": local_count,
        "eiem_trainable_parameters": eiem_count,
        "eiem_extra_parameters": eiem_count - local_count,
        "delta_fraction": delta,
        "within_point_one_percent": abs(delta) <= PARAMETER_MISMATCH_LIMIT,
    }
    del local, eiem
    return result


def stage_a_preflight() -> dict[str, Any]:
    cfg = transformer_100m_config()
    if HEAD_DIM != 64:
        raise RuntimeError("frozen #977 head dimension drift")
    if MAX_SEQ_LEN != 2 * LOCAL_WINDOW:
        raise RuntimeError("frozen #977 session must remain two 512-token local windows")
    if FIRST_SCREEN_TOKEN_BUDGET != 4 * PRIOR_25M_TOKEN_BUDGET:
        raise RuntimeError("first-screen token budget is not exactly 4x the 25M budget")

    analytical = analytical_parameter_accounting()
    instantiated = instantiated_parameter_accounting()
    if analytical != instantiated:
        raise RuntimeError(
            f"analytical/instantiated parameter accounting mismatch: {analytical} != {instantiated}"
        )
    if int(instantiated["local_trainable_parameters"]) != EXPECTED_LOCAL_PARAMETERS:
        raise RuntimeError(f"LOCAL parameter count drift: {instantiated}")
    if int(instantiated["eiem_trainable_parameters"]) != EXPECTED_EIEM_PARAMETERS:
        raise RuntimeError(f"EIEM parameter count drift: {instantiated}")
    if int(instantiated["eiem_extra_parameters"]) != EXPECTED_EIEM_EXTRA_PARAMETERS:
        raise RuntimeError(f"EIEM extra parameter count drift: {instantiated}")
    if not bool(instantiated["within_point_one_percent"]):
        raise RuntimeError(f"parameter fairness gate failed: {instantiated}")

    return {
        "classification": "CHM_V1_100M_STAGE_A_ENGINEERING_PREFLIGHT_PASS",
        "research_issue": ISSUE,
        "starting_main_sha": STARTING_MAIN_SHA,
        "configuration": asdict(cfg),
        "head_dim": HEAD_DIM,
        "local_window": LOCAL_WINDOW,
        "address_dim": ADDRESS_DIM,
        "engineering_smoke_seed": ENGINEERING_SMOKE_SEED,
        "future_scientific_screen_seed_reserved_not_authorized": FUTURE_SCREEN_SEED,
        "prior_25m_token_budget_per_model": PRIOR_25M_TOKEN_BUDGET,
        "first_screen_token_budget_per_model": FIRST_SCREEN_TOKEN_BUDGET,
        "confirmatory_candidate_token_budget_per_model_not_authorized": CONFIRMATORY_CANDIDATE_TOKEN_BUDGET,
        "parameter_accounting": instantiated,
        "gpu_authorized": False,
        "training_authorized": False,
        "scientific_execution_authorized": False,
        "interpretation_ceiling": (
            "Stage-A engineering/configuration evidence only; no modeling, GPU, or breakthrough claim."
        ),
    }
