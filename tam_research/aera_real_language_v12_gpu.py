from __future__ import annotations

"""Guarded AERA-v12 real-language development harness.

This module deliberately reuses the already-tested v11 matched-pair training loop.
V12 is a subclass of the v11 runtime core; only the build function, progressive
routing loss, optional-router freeze boundary, and exact v12 preflight are swapped.
The development seed is adaptive evidence and can never count as an independent
replication or authorize 100M by itself.
"""

from typing import Any

from . import aera_real_language_v11 as v11
from . import aera_real_language_v12 as v12

SEED = 8231
_BASE_V11_EVALUATE_AERA = v11.evaluate_aera


def validate_protocol(data_dir: str) -> dict[str, Any]:
    cpu = v12.cpu_preflight()
    data = v12.validate_production_data(data_dir)
    return {
        **cpu,
        "data": data,
        "development_seed": SEED,
        "counts_toward_breakthrough_evidence": False,
        "gpu_authorized_for_this_guarded_dev_run": True,
    }


def _install_v12_harness() -> None:
    # v11's training loop resolves these names dynamically from its module globals.
    # V12 inherits the same runtime/result contract, so this keeps the matched pair
    # implementation identical while changing only the explicitly intended v12 pieces.
    v11.build_aera = v12.build_aera
    v11.aera_matched_loss = v12.aera_matched_loss
    v11.DENSE_WARMUP_STEPS = v12.DENSE_WARMUP_STEPS
    v11.set_stage_router_trainable = v12.set_optional_stage_router_trainable
    v11.validate_protocol = validate_protocol
    v11.evaluate_aera = _BASE_V11_EVALUATE_AERA


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"v12 development run is frozen to seed {SEED}")
    _install_v12_harness()
    result = v11.train_matched_pair(data_dir=data_dir, run_dir=run_dir, seed=seed)

    diagnostics = result["diagnostics"]
    stage_execution = float(diagnostics["stage_execution_fraction"])
    quality_gap = float(diagnostics["quality_gap_nll"])
    speed_ratio = float(diagnostics["batch8_speed_ratio"])
    state_advantage = float(diagnostics["state_advantage_nll"])

    v12_checks = {
        "quality_gap_nll_le_0_50": quality_gap <= 0.50,
        "stage_execution_between_0_35_and_0_70": 0.35 <= stage_execution <= 0.70,
        "batch8_speed_ratio_ge_0_30": speed_ratio >= 0.30,
        "state_advantage_nonnegative": state_advantage >= 0.0,
    }
    result["v12_development_checks"] = v12_checks
    result["v12_development_pass"] = all(v12_checks.values())
    result["claims"] = {
        "development_seed_only": True,
        "counts_toward_breakthrough_evidence": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    return result
