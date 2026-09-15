from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase25 as p25
from tam_research import physics_tokenizer_phase28 as p28

CONTEXT = 8
RIDGE = 0.001
REGISTERED_BASES = (831315000, 831315100, 831315200)
SMOKE_BASE = 931315000


def seeds_from_base(b: int) -> p28.Seeds:
    return p28.Seeds(
        b + 1, b + 2, b + 3, b + 4, b + 5, b + 6,
        b + 11, b + 12, b + 13,
        b + 21, b + 22, b + 23, b + 24,
        b + 31, b + 32, b + 33, b + 34,
        b + 41, b + 42, b + 43, b + 44,
        b + 51, b + 52, b + 53, b + 54,
    )


def _mse_per_sequence(y: np.ndarray, target: np.ndarray) -> np.ndarray:
    d = (y - target).astype(np.float64)
    return np.mean(d * d, axis=(1, 2, 3))


def diagnose(s: p28.Seeds, *, grid=16, train_n=64, eval_n=48,
             code_sizes=(2048, 128), iters=20, max_points=60000):
    t0 = time.time()
    state, dm, ds, basis = p28.build_all(s, grid, train_n, code_sizes, iters, max_points)
    raw_all = p28.build_eval(s, grid, eval_n)
    raw = raw_all["nls_id_h8"]

    qc, context_diag = p10.encode_comp_seq(raw[:, :CONTEXT], state, dm, ds, True)
    choose62, mse26_val, mse62_val, score26, score62 = p28.select_models(qc, basis)
    m26 = p25.fit_maps(qc, p25.base26_features, 26, basis, RIDGE)
    m62 = p25.fit_maps(qc, p25.interaction62_features, 62, basis, RIDGE)
    start = qc[:, -1]

    horizons = {}
    for h in range(1, 9):
        target = raw[:, CONTEXT + h - 1]
        persist = raw[:, CONTEXT - 1]
        pm = p10.mse(persist, target)

        ys, cs = p28.selected_rollout(start, m26, m62, choose62, h, basis, dm, ds)
        y26, c26 = p25.token_rollout(start, m26, h, p25.base26_features, basis, dm, ds)
        y62, c62 = p25.token_rollout(start, m62, h, p25.interaction62_features, basis, dm, ds)

        ms = p10.mse(ys, target)
        m26e = p10.mse(y26, target)
        m62e = p10.mse(y62, target)
        per26 = _mse_per_sequence(y26, target)
        per62 = _mse_per_sequence(y62, target)
        perseq_oracle = float(np.mean(np.minimum(per26, per62)))
        aggregate_oracle = min(m26e, m62e)

        horizons[str(h)] = {
            "persistence_mse": pm,
            "selected_mse": ms,
            "selected_ratio_to_persistence": ms / pm,
            "simple26_mse": m26e,
            "simple26_ratio_to_persistence": m26e / pm,
            "interaction62_mse": m62e,
            "interaction62_ratio_to_persistence": m62e / pm,
            "aggregate_oracle_parent_mse": aggregate_oracle,
            "aggregate_oracle_parent_ratio_to_persistence": aggregate_oracle / pm,
            "per_sequence_oracle_parent_mse": perseq_oracle,
            "per_sequence_oracle_parent_ratio_to_persistence": perseq_oracle / pm,
            "selected_rollout_clip_fraction": cs,
            "simple26_rollout_clip_fraction": c26,
            "interaction62_rollout_clip_fraction": c62,
        }

    h8 = horizons["8"]
    if h8["selected_ratio_to_persistence"] < 1.0:
        classification = "PASS_STABLE"
    elif h8["aggregate_oracle_parent_ratio_to_persistence"] < 1.0:
        classification = "SELECTOR_LIMIT"
    else:
        classification = "MODEL_LIMIT"

    return {
        "status": "PHASE31_NLS_STABILITY_DIAGNOSTIC",
        "seeds": asdict(s),
        "context_frames": CONTEXT,
        "ridge": RIDGE,
        "selector": "phase28_heldout_context_bic_nested_26_vs_62",
        "split": "nls_id_h8",
        "selected_interaction_fraction": float(np.mean(choose62)),
        "validation_mse26_mean": float(np.mean(mse26_val)),
        "validation_mse62_mean": float(np.mean(mse62_val)),
        "validation_score26_mean": float(np.mean(score26)),
        "validation_score62_mean": float(np.mean(score62)),
        "context_mse": p10.mse(qc, raw[:, :CONTEXT]),
        "context_clip_fraction": context_diag["clip_fraction"],
        "horizons": horizons,
        "h8_classification": classification,
        "runtime_seconds": time.time() - t0,
        "claim_limit": "Synthetic NLS mechanism diagnostic only; no new-physics or unrestricted discovery claim.",
    }


def validate_invariants():
    sets = [seeds_from_base(b) for b in REGISTERED_BASES]
    vals = [v for s in sets for v in asdict(s).values()]
    assert len(vals) == 75
    assert len(vals) == len(set(vals))
    assert CONTEXT == p28.CONTEXT == 8
    assert RIDGE == p28.RIDGE == 0.001
    assert p28.FIT_FRAMES == 6
    assert p28.VALIDATION_PAIRS == ((5, 6), (6, 7))
    return {
        "registered_bases": list(REGISTERED_BASES),
        "registered_derived_seed_count": len(vals),
        "registered_derived_seeds_unique": True,
        "smoke_base": SMOKE_BASE,
        "scientific_and_smoke_namespaces_disjoint": all(not (SMOKE_BASE <= v < SMOKE_BASE + 1000) for v in vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    inv = validate_invariants()

    if a.smoke:
        result = diagnose(
            seeds_from_base(SMOKE_BASE),
            grid=8,
            train_n=10,
            eval_n=4,
            code_sizes=(16, 4),
            iters=2,
            max_points=1500,
        )
        result["status"] = "SMOKE_ONLY_NONSCIENTIFIC_ENGINEERING_SEED"
    else:
        if a.rep not in (1, 2, 3):
            raise SystemExit("--rep must be 1, 2, or 3")
        result = diagnose(seeds_from_base(REGISTERED_BASES[a.rep - 1]))
        result["replicate"] = a.rep
        result["status"] = "PHASE31_NLS_STABILITY_REGISTERED_DIAGNOSTIC"

    result["invariants"] = inv
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
