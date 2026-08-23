from __future__ import annotations

"""Guarded AERA-v15 real-language development harness.

Fresh seed8291 tests whether per-stage budget matching preserves v14's quality,
adaptivity and sparse execution while preventing the optional depth hierarchy from
collapsing. This remains adaptive development evidence only.
"""

import json
from pathlib import Path
from typing import Any

import torch

from . import aera_real_language_v11 as v11
from . import aera_real_language_v12 as v12
from . import aera_real_language_v15 as v15
from .aera_real_language import SEQ_LEN
from .data import TokenBin

SEED = 8291
TARGET_RATES = torch.tensor([0.50, 1.0 / 3.0, 1.0 / 6.0], dtype=torch.float32)


def validate_protocol(data_dir: str) -> dict[str, Any]:
    cpu = v15.cpu_preflight()
    data = v12.validate_production_data(data_dir)
    return {
        **cpu,
        "gpu_authorized": True,
        "gpu_authorization_scope": "one guarded AERA-v15 development seed8291 L4 run only",
        "data": data,
        "development_seed": SEED,
        "counts_toward_breakthrough_evidence": False,
        "frozen_optional_stage_target_rates": TARGET_RATES.tolist(),
    }


def _install_v15_harness() -> None:
    v11.build_aera = v15.build_aera
    v11.aera_matched_loss = v15.aera_matched_loss
    v11.DENSE_WARMUP_STEPS = v15.DENSE_WARMUP_STEPS
    v11.ROUTER_CALIBRATION_END = v15.ROUTER_CALIBRATION_END
    v11.SPARSE_CALIBRATION_EVERY = v15.SPARSE_CALIBRATION_EVERY
    v11.set_stage_router_trainable = v15.set_optional_stage_router_trainable
    v11.validate_protocol = validate_protocol


@torch.no_grad()
def _heldout_optional_stage_rates(*, data_dir: str, run_dir: str, seed: int) -> list[float]:
    device = torch.device("cuda")
    payload = torch.load(Path(run_dir) / "aera.pt", map_location="cpu", weights_only=False)
    if payload.get("seed") != seed:
        raise RuntimeError("v15 checkpoint seed mismatch")
    model = v15.build_aera(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    val = TokenBin(str(Path(data_dir) / "val.bin"))
    g = torch.Generator(device="cpu").manual_seed(seed + 50_000)
    runs = torch.zeros(3, dtype=torch.float64)
    total_chunks = 0
    for _ in range(20):
        x, _ = val.batch(4, SEQ_LEN, g, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, hard=True, route_mode="hard_sparse", update_memory=False)
        routes = out.get("stage_routes")
        if not isinstance(routes, list):
            raise RuntimeError("v15 held-out eval missing stage routes")
        for chunk in routes:
            if not isinstance(chunk, list) or len(chunk) != 4:
                raise RuntimeError("v15 held-out eval invalid stage routes")
            total_chunks += x.size(0)
            for i, item in enumerate(chunk[1:]):
                if not isinstance(item, dict):
                    raise RuntimeError("v15 held-out eval invalid route item")
                gate = item.get("stage_route_gate")
                if not isinstance(gate, torch.Tensor):
                    raise RuntimeError("v15 held-out eval missing stage gate")
                runs[i] += float((gate[:, 0] >= 0.5).sum())
    if total_chunks <= 0:
        raise RuntimeError("v15 held-out eval collected no chunks")
    return [float(v / total_chunks) for v in runs]


def train_matched_pair(*, data_dir: str, run_dir: str, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"v15 development run is frozen to seed {SEED}")
    _install_v15_harness()
    result = v11.train_matched_pair(data_dir=data_dir, run_dir=run_dir, seed=seed)

    diagnostics = result["diagnostics"]
    stage_execution = float(diagnostics["stage_execution_fraction"])
    quality_gap = float(diagnostics["quality_gap_nll"])
    speed_ratio = float(diagnostics["batch8_speed_ratio"])
    state_advantage = float(diagnostics["state_advantage_nll"])
    optional_rates = _heldout_optional_stage_rates(data_dir=data_dir, run_dir=run_dir, seed=seed)
    target_rates = TARGET_RATES.tolist()
    rate_mae = sum(abs(a - b) for a, b in zip(optional_rates, target_rates)) / len(target_rates)
    ordered = optional_rates[0] >= optional_rates[1] >= optional_rates[2]

    diagnostics["optional_stage_run_fractions"] = optional_rates
    diagnostics["optional_stage_target_rate_mae"] = rate_mae
    checks = {
        "quality_gap_nll_le_0_50": quality_gap <= 0.50,
        "stage_execution_between_0_35_and_0_70": 0.35 <= stage_execution <= 0.70,
        "batch8_speed_ratio_ge_0_30": speed_ratio >= 0.30,
        "state_advantage_nonnegative": state_advantage >= 0.0,
        "all_optional_stages_run_ge_0_05": all(rate >= 0.05 for rate in optional_rates),
        "optional_stage_rates_shallow_to_deep": ordered,
        "optional_stage_target_mae_le_0_12": rate_mae <= 0.12,
    }
    result["v15_development_checks"] = checks
    result["v15_development_pass"] = all(checks.values())
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
