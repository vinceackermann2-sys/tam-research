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
from tam_research import physics_tokenizer_phase30 as p30

CONTEXT = 8
RIDGE = 0.001
MEDIAN_GAIN_MIN = 0.10
GUARD_TRANSITIONS = (1, 2, 3)
GUARD_START_FRAME = 4
GUARD_TARGET_FRAME = 7
GUARD_HORIZON = GUARD_TARGET_FRAME - GUARD_START_FRAME
DAMPING_GRID = (1.0, 0.5, 0.25, 0.125)
REGISTERED_BASES = (831310000, 831310100, 831310200)
REGISTERED = tuple(p30._seedset(base) for base in REGISTERED_BASES)
FORBIDDEN_BASES = (
    830300000, 830300100, 830300200,
    830301000, 830301100, 830301200,
    830310000, 830310100, 830310200,
)
FAMILIES = ('wave', 'gray_scott', 'nls', 'advection', 'burgers', 'hamilton_jacobi')


def _per_trajectory_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b).astype(np.float64)
    return np.mean(d * d, axis=(1, 2, 3))


def _damped_quantized_step(
    cur: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    dm,
    ds,
):
    lam = np.asarray(lambdas, dtype=np.float32).reshape(-1, 1, 1, 1)
    q, diag = p10.q_comp_delta((y - cur) * lam, dm, ds, True)
    return (cur + q).astype(np.float32), diag


def damped_model_rollout(start, maps, h, feature_fn, basis, dm, ds, lambdas):
    cur = start.copy()
    clips = []
    for _ in range(h):
        y = p25.predict(cur, maps, feature_fn, basis)
        cur, diag = _damped_quantized_step(cur, y, lambdas, dm, ds)
        clips.append(diag['clip_fraction'])
    return cur, float(np.mean(clips)) if clips else 0.0


def damped_selected_rollout(start, m26, m62, choose62, h, basis, dm, ds, lambdas):
    cur = start.copy()
    clips = []
    mask = choose62[:, None, None, None]
    for _ in range(h):
        y26 = p25.predict(cur, m26, p25.base26_features, basis)
        y62 = p25.predict(cur, m62, p25.interaction62_features, basis)
        y = np.where(mask, y62, y26).astype(np.float32)
        cur, diag = _damped_quantized_step(cur, y, lambdas, dm, ds)
        clips.append(diag['clip_fraction'])
    return cur, float(np.mean(clips)) if clips else 0.0


def select_damping(qc, basis, dm, ds, use82_group, choose62):
    """Choose lambda only from observed decoded context; no target after t=7 is touched."""
    m26 = p30.fit_transition_subset(
        qc, p25.base26_features, 26, basis, GUARD_TRANSITIONS, RIDGE
    )
    m62 = p30.fit_transition_subset(
        qc, p25.interaction62_features, 62, basis, GUARD_TRANSITIONS, RIDGE
    )
    m82 = None
    if use82_group:
        m82 = p30.fit_transition_subset(
            qc, p30.p29.odd_spatial_quadratic82, 82, basis, GUARD_TRANSITIONS, RIDGE
        )

    start = qc[:, GUARD_START_FRAME]
    target = qc[:, GUARD_TARGET_FRAME]
    errors = []
    for lam in DAMPING_GRID:
        lambdas = np.full(len(qc), lam, dtype=np.float32)
        if use82_group:
            pred, _ = damped_model_rollout(
                start, m82, GUARD_HORIZON,
                p30.p29.odd_spatial_quadratic82, basis, dm, ds, lambdas,
            )
        else:
            pred, _ = damped_selected_rollout(
                start, m26, m62, choose62, GUARD_HORIZON,
                basis, dm, ds, lambdas,
            )
        errors.append(_per_trajectory_mse(pred, target))

    err = np.stack(errors, axis=0)
    # DAMPING_GRID is descending, so np.argmin gives the larger lambda on exact ties.
    best_idx = np.argmin(err, axis=0)
    grid = np.asarray(DAMPING_GRID, dtype=np.float32)
    chosen = grid[best_idx]
    best = err[best_idx, np.arange(err.shape[1])]
    persistence = _per_trajectory_mse(start, target)
    return chosen, best, persistence, err


def eval_split(raw, state, dm, ds, basis, h):
    qc, diag = p10.encode_comp_seq(raw[:, :CONTEXT], state, dm, ds, True)
    target = raw[:, CONTEXT + h - 1]
    persist = raw[:, CONTEXT - 1]
    persistence_mse = p10.mse(persist, target)

    median_gain, gains = p30.shared_support_gain(qc, basis)
    use82_group = bool(median_gain > MEDIAN_GAIN_MIN)
    choose62, *_ = p28.select_models(qc, basis)
    lambdas, guard_best, guard_persistence, guard_all = select_damping(
        qc, basis, dm, ds, use82_group, choose62
    )

    if use82_group:
        m82 = p30.fit_transition_subset(
            qc,
            p30.p29.odd_spatial_quadratic82,
            82,
            basis,
            p30.CLEAN_ALL_TRANSITIONS,
            RIDGE,
        )
        pred, clips = damped_model_rollout(
            qc[:, -1], m82, h, p30.p29.odd_spatial_quadratic82,
            basis, dm, ds, lambdas,
        )
        parent_interaction_fraction = None
    else:
        m26 = p25.fit_maps(qc, p25.base26_features, 26, basis, RIDGE)
        m62 = p25.fit_maps(qc, p25.interaction62_features, 62, basis, RIDGE)
        pred, clips = damped_selected_rollout(
            qc[:, -1], m26, m62, choose62, h, basis, dm, ds, lambdas
        )
        parent_interaction_fraction = float(np.mean(choose62))

    selected_mse = p10.mse(pred, target)
    counts = {str(lam): int(np.sum(lambdas == lam)) for lam in DAMPING_GRID}
    return {
        'persistence_mse': persistence_mse,
        'selected_mse': selected_mse,
        'selected_ratio': selected_mse / persistence_mse,
        'use82_group': use82_group,
        'median_heldout_gain82_vs62': median_gain,
        'gain_positive_fraction': float(np.mean(gains > 0)),
        'parent_interaction_selection_fraction': parent_interaction_fraction,
        'damping_counts': counts,
        'damping_mean': float(np.mean(lambdas)),
        'damping_full_fraction': float(np.mean(lambdas == 1.0)),
        'guard_best_mse_mean': float(np.mean(guard_best)),
        'guard_persistence_mse_mean': float(np.mean(guard_persistence)),
        'guard_best_to_persistence_mean': float(np.mean(guard_best / (guard_persistence + 1e-30))),
        'guard_improves_persistence_fraction': float(np.mean(guard_best < guard_persistence)),
        'guard_grid_mse_mean': {
            str(lam): float(np.mean(guard_all[i])) for i, lam in enumerate(DAMPING_GRID)
        },
        'context_mse': p10.mse(qc, raw[:, :CONTEXT]),
        'context_clip_fraction': diag['clip_fraction'],
        'rollout_clip_fraction': clips,
    }


def run_rep(seedset, grid=16, train_n=64, eval_n=48, code_sizes=(2048, 128), iters=20, max_points=60000):
    t0 = time.time()
    state, dm, ds, basis = p30.build_all(seedset, grid, train_n, code_sizes, iters, max_points)
    raw = p30.build_eval(seedset, grid, eval_n)
    splits = {
        name: eval_split(x, state, dm, ds, basis, 8 if name.endswith('h8') else 3)
        for name, x in raw.items()
    }
    primary = {name: float(v['selected_ratio']) for name, v in splits.items()}
    per_family = {}
    for family in FAMILIES:
        names = [name for name in primary if p30.family_name(name) == family]
        vals = [primary[name] for name in names]
        per_family[family] = {
            'passing_cells': int(sum(x < 1.0 for x in vals)),
            'total_cells': len(vals),
            'all_below_persistence': bool(all(x < 1.0 for x in vals)),
            'worst_ratio': float(max(vals)),
            'use82_splits': int(sum(bool(splits[name]['use82_group']) for name in names)),
            'mean_full_damping_fraction': float(np.mean([splits[name]['damping_full_fraction'] for name in names])),
        }
    return {
        'seeds': asdict(seedset),
        'selector': 'phase30_selector_plus_context_recursive_damping',
        'damping_grid': list(DAMPING_GRID),
        'guard_transitions': list(GUARD_TRANSITIONS),
        'guard_start_frame': GUARD_START_FRAME,
        'guard_target_frame': GUARD_TARGET_FRAME,
        'context_frames': CONTEXT,
        'ridge': RIDGE,
        'representation_training_families': ['wave', 'gray_scott'],
        'no_family_labels': True,
        'no_future_targets': True,
        'primary_ratios': primary,
        'primary_passing_cells': int(sum(x < 1.0 for x in primary.values())),
        'primary_total_cells': len(primary),
        'primary_all_23_below_persistence': bool(all(x < 1.0 for x in primary.values())),
        'per_family': per_family,
        'splits': splits,
        'runtime_seconds': time.time() - t0,
    }


def validate_invariants():
    values = []
    for seedset in REGISTERED:
        values.extend(asdict(seedset).values())
    assert len(values) == 87
    assert len(values) == len(set(values))
    assert tuple(sorted(DAMPING_GRID, reverse=True)) == DAMPING_GRID
    assert 0.0 not in DAMPING_GRID
    assert GUARD_TRANSITIONS == (1, 2, 3)
    assert GUARD_START_FRAME == 4 and GUARD_TARGET_FRAME == 7 and GUARD_HORIZON == 3
    assert CONTEXT == p30.CONTEXT == 8
    assert RIDGE == p30.RIDGE == 0.001
    assert MEDIAN_GAIN_MIN == p30.MEDIAN_GAIN_MIN == 0.10
    assert not set(REGISTERED_BASES).intersection(FORBIDDEN_BASES)
    return {
        'registered_seed_count': len(values),
        'registered_bases': list(REGISTERED_BASES),
        'damping_grid': list(DAMPING_GRID),
        'guard_context_only': True,
        'guard_last_observed_frame': GUARD_TARGET_FRAME,
        'phase30_selector_unchanged': True,
        'family_label_used': False,
        'registered_execution_enabled': False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--rep', type=int, default=0)
    args = ap.parse_args()
    invariants = validate_invariants()

    if not args.smoke:
        raise SystemExit(
            'Phase-31 registered execution is deliberately disabled pending separate authority; '
            'use --smoke only.'
        )

    seedset = p30._seedset(999311000)
    result = run_rep(
        seedset,
        grid=8,
        train_n=10,
        eval_n=4,
        code_sizes=(16, 4),
        iters=2,
        max_points=1500,
    )
    result['status'] = 'PHASE31_SMOKE_ONLY_NONREGISTERED_SEED'
    result['invariants'] = invariants
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'status': result['status'],
        'primary_cells': result['primary_total_cells'],
        'invariants': invariants,
    }, indent=2))


if __name__ == '__main__':
    main()
