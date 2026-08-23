from __future__ import annotations

"""Guarded AERA-v13 real-language development harness.

This wrapper reuses the established v11 matched-pair training/evaluation loop while
installing the CPU-tested v13 router curriculum.  Seed8251 is adaptive development
evidence only: it cannot count as independent replication, prove a breakthrough,
or authorize 100M by itself.
"""

import json
from pathlib import Path
from typing import Any

from . import aera_real_language_v11 as v11
from . import aera_real_language_v12 as v12
from . import aera_real_language_v13 as v13

SEED = 8251
_BASE_V11_EVALUATE_AERA = v11.evaluate_aera


def validate_protocol(data_dir: str) -> dict[str, Any]:
    cpu = v13.cpu_preflight()
    data = v12.validate_production_data(data_dir)
    return {
        **cpu,
        "gpu_authorized": True,
        "gpu_authorization_scope": "one guarded AERA-v13 development seed8251 L4 run only",
        "data": data,
        "development_seed": SEED,
        "counts_toward_breakthrough_evidence": False,
    }


def _install_v13_harness() -> None:
    # v11 resolves these functions/constants from module globals during training.
    # Reusing that loop preserves the matched Transformer/data/optimizer/evaluation
    # implementation while swapping only the explicitly intended v13 router policy.
    v11.build_aera = v13.build_aera
    v11.aera_matched_loss = v13.aera_matched_loss
    v11.DENSE_WARMUP_STEPS = v13.DENSE_WARMUP_STEPS
    v11.set_stage_router_trainable = v13.set_optional_stage_router_trainable
    v11.validate_protocol = validate_protocol
    v11.evaluate_aera = _BASE_V11_EVALUATE_AERA


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"v13 development run is frozen to seed {SEED}")
    _install_v13_harness()
    result = v11.train_matched_pair(data_dir=data_dir, run_dir=run_dir, seed=seed)

    diagnostics = result["diagnostics"]
    stage_execution = float(diagnostics["stage_execution_fraction"])
    quality_gap = float(diagnostics["quality_gap_nll"])
    speed_ratio = float(diagnostics["batch8_speed_ratio"])
    state_advantage = float(diagnostics["state_advantage_nll"])

    checks = {
        "quality_gap_nll_le_0_50": quality_gap <= 0.50,
        "stage_execution_between_0_35_and_0_70": 0.35 <= stage_execution <= 0.70,
        "batch8_speed_ratio_ge_0_30": speed_ratio >= 0.30,
        "state_advantage_nonnegative": state_advantage >= 0.0,
    }
    result["v13_development_checks"] = checks
    result["v13_development_pass"] = all(checks.values())
    result["claims"] = {
        "development_seed_only": True,
        "counts_toward_breakthrough_evidence": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(result, indent=2))
    return result
