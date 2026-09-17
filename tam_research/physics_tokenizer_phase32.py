from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase25 as p25
from tam_research import physics_tokenizer_phase28 as p28
from tam_research import physics_tokenizer_phase30 as p30
from tam_research import physics_tokenizer_phase31 as p31

CONTEXT = 8
RIDGE = 0.001
MEDIAN_GAIN_MIN = 0.10
DAMPING_GRID = (1.0, 0.5, 0.25, 0.125)
GUARD_WINDOWS = (
    ((0, 1), 2, 7),
    ((0, 1, 2), 3, 7),
    ((0, 1, 2, 3), 4, 7),
)
REGISTERED_BASES = (832320000, 832320100, 832320200)
REGISTERED = tuple(p30._seedset(base) for base in REGISTERED_BASES)
FORBIDDEN_BASES = (
    830300000, 830300100, 830300200,
    830301000, 830301100, 830301200,
    830310000, 830310100, 830310200,
    831310000, 831310100, 831310200,
    999311000,
)
FAMILIES = ('wave', 'gray_scott', 'nls', 'advection', 'burgers', 'hamilton_jacobi')
EPS = 1e-30


def _per_trajectory_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b).astype(np.float64)
    return np.mean(d * d, axis=(1, 2, 3))


def _fit_guard_maps(qc, basis, fit_transitions, use82_group):
    if use82_group:
        return p30.fit_transition_subset(
            qc, p30.p29.odd_spatial_quadratic82, 82, basis, fit_transitions, RIDGE
        )
    m26 = p30.fit_transition_subset(
        qc, p25.base26_features, 26, basis, fit_transitions, RIDGE
    )
    m62 = p30.fit_transition_subset(
        qc, p25.interaction62_features, 62, basis, fit_transitions, RIDGE
    )
    return m26, m62


def _guard_rollout(start, maps, horizon, basis, dm, ds, lambdas, use82_group, choose62):
    if use82_group:
        return p31.damped_model_rollout(
            start, maps, horizon, p30.p29.odd_spatial_quadratic82, basis, dm, ds, lambdas
        )
    m26, m62 = maps
    return p31.damped_selected_rollout(
        start, m26, m62, choose62, horizon, basis, dm, ds, lambdas
    )


def _select_from_window_errors(window_errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select by per-trajectory worst normalized context error; ties favor larger lambda."""
    if window_errors.ndim != 3:
        raise ValueError('window_errors must have shape [windows, lambdas, trajectories]')
    candidate_scores = np.max(window_errors, axis=0)
    best_idx = np.argmin(candidate_scores, axis=0)
    grid = np.asarray(DAMPING_GRID, dtype=np.float32)
    chosen = grid[best_idx]
    best = candidate_scores[best_idx, np.arange(candidate_scores.shape[1])]
    return chosen, best


def select_multihorizon_damping(qc, basis, dm, ds, use82_group, choose62):
    """Choose lambda from observed context only using 5/4/3-step forward-chaining windows."""
    per_window = []
    raw_window = []
    persistence_window = []

    for fit_transitions, start_frame, target_frame in GUARD_WINDOWS:
        maps = _fit_guard_maps(qc, basis, fit_transitions, use82_group)
        start = qc[:, start_frame]
        target = qc[:, target_frame]
        horizon = target_frame - start_frame
        persistence = _per_trajectory_mse(start, target)
        candidate_errors = []
        candidate_normalized = []
        for lam in DAMPING_GRID:
            lambdas = np.full(len(qc), lam, dtype=np.float32)
            pred, _ = _guard_rollout(
                start, maps, horizon, basis, dm, ds, lambdas, use82_group, choose62
            )
            err = _per_trajectory_mse(pred, target)
            candidate_errors.append(err)
            candidate_normalized.append(err / (persistence + EPS))
        raw_window.append(np.stack(candidate_errors, axis=0))
        per_window.append(np.stack(candidate_normalized, axis=0))
        persistence_window.append(persistence)

    normalized = np.stack(per_window, axis=0)
    raw_errors = np.stack(raw_window, axis=0)
    persistence_errors = np.stack(persistence_window, axis=0)
    chosen, worst_best = _select_from_window_errors(normalized)
    return chosen, worst_best, normalized, raw_errors, persistence_errors


def eval_split(raw, state, dm, ds, basis, h):
    qc, diag = p10.encode_comp_seq(raw[:, :CONTEXT], state, dm, ds, True)
    target = raw[:, CONTEXT + h - 1]
    persist = raw[:, CONTEXT - 1]
    persistence_mse = p10.mse(persist, target)

    median_gain, gains = p30.shared_support_gain(qc, basis)
    use82_group = bool(median_gain > MEDIAN_GAIN_MIN)
    choose62, *_ = p28.select_models(qc, basis)

    lambdas, guard_worst_best, guard_norm, guard_raw, guard_persistence = select_multihorizon_damping(
        qc, basis, dm, ds, use82_group, choose62
    )
    phase31_lambdas, *_ = p31.select_damping(qc, basis, dm, ds, use82_group, choose62)

    if use82_group:
        m82 = p30.fit_transition_subset(
            qc, p30.p29.odd_spatial_quadratic82, 82, basis,
            p30.CLEAN_ALL_TRANSITIONS, RIDGE,
        )
        pred, clips = p31.damped_model_rollout(
            qc[:, -1], m82, h, p30.p29.odd_spatial_quadratic82,
            basis, dm, ds, lambdas,
        )
        parent_interaction_fraction = None
    else:
        m26 = p25.fit_maps(qc, p25.base26_features, 26, basis, RIDGE)
        m62 = p25.fit_maps(qc, p25.interaction62_features, 62, basis, RIDGE)
        pred, clips = p31.damped_selected_rollout(
            qc[:, -1], m26, m62, choose62, h, basis, dm, ds, lambdas
        )
        parent_interaction_fraction = float(np.mean(choose62))

    selected_mse = p10.mse(pred, target)
    counts = {str(lam): int(np.sum(lambdas == lam)) for lam in DAMPING_GRID}
    window_names = [f'h{target - start}' for _, start, target in GUARD_WINDOWS]
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
        'phase31_lambda_difference_fraction': float(np.mean(lambdas != phase31_lambdas)),
        'guard_worst_normalized_best_mean': float(np.mean(guard_worst_best)),
        'guard_window_normalized_mse_mean': {
            window_names[w]: {
                str(lam): float(np.mean(guard_norm[w, i]))
                for i, lam in enumerate(DAMPING_GRID)
            }
            for w in range(len(GUARD_WINDOWS))
        },
        'guard_window_raw_mse_mean': {
            window_names[w]: {
                str(lam): float(np.mean(guard_raw[w, i]))
                for i, lam in enumerate(DAMPING_GRID)
            }
            for w in range(len(GUARD_WINDOWS))
        },
        'guard_window_persistence_mse_mean': {
            window_names[w]: float(np.mean(guard_persistence[w]))
            for w in range(len(GUARD_WINDOWS))
        },
        'context_mse': p10.mse(qc, raw[:, :CONTEXT]),
        'context_clip_fraction': diag['clip_fraction'],
        'rollout_clip_fraction': clips,
    }


def run_rep(seedset, grid=16, train_n=64, eval_n=48, code_sizes=(2048, 128), iters=20, max_points=60000):
    state, dm, ds, basis = p30.build_all(seedset, grid, train_n, code_sizes, iters, max_points)
    raw = p30.build_eval(seedset, grid, eval_n)
    splits = {
        name: eval_split(x, state, dm, ds, basis, 8 if name.endswith('h8') else 3)
        for name, x in raw.items()
    }
    primary = {name: float(v['selected_ratio']) for name, v in splits.items()}
    return {
        'seeds': asdict(seedset),
        'selector': 'phase30_selector_plus_phase32_multihorizon_context_guard',
        'damping_grid': list(DAMPING_GRID),
        'guard_windows': [
            {
                'fit_transitions': list(fit),
                'start_frame': start,
                'target_frame': target,
                'rollout_horizon': target - start,
            }
            for fit, start, target in GUARD_WINDOWS
        ],
        'context_frames': CONTEXT,
        'ridge': RIDGE,
        'representation_training_families': ['wave', 'gray_scott'],
        'no_family_labels': True,
        'no_future_targets': True,
        'primary_ratios': primary,
        'primary_passing_cells': int(sum(x < 1.0 for x in primary.values())),
        'primary_total_cells': len(primary),
        'primary_all_23_below_persistence': bool(all(x < 1.0 for x in primary.values())),
        'splits': splits,
    }


def validate_invariants():
    values = []
    for seedset in REGISTERED:
        values.extend(asdict(seedset).values())
    assert len(values) == 87
    assert len(values) == len(set(values))
    assert not set(REGISTERED_BASES).intersection(FORBIDDEN_BASES)
    assert tuple(sorted(DAMPING_GRID, reverse=True)) == DAMPING_GRID
    assert 0.0 not in DAMPING_GRID
    assert CONTEXT == p30.CONTEXT == p31.CONTEXT == 8
    assert RIDGE == p30.RIDGE == p31.RIDGE == 0.001
    assert MEDIAN_GAIN_MIN == p30.MEDIAN_GAIN_MIN == p31.MEDIAN_GAIN_MIN == 0.10
    assert tuple(target - start for _, start, target in GUARD_WINDOWS) == (5, 4, 3)
    for fit, start, target in GUARD_WINDOWS:
        assert target == CONTEXT - 1
        assert max(fit) + 1 <= start
        assert all(0 <= t < CONTEXT - 1 for t in fit)
    return {
        'registered_seed_count': len(values),
        'registered_bases': list(REGISTERED_BASES),
        'damping_grid': list(DAMPING_GRID),
        'guard_horizons': [target - start for _, start, target in GUARD_WINDOWS],
        'guard_context_only': True,
        'phase30_selector_unchanged': True,
        'phase31_scientific_rollout_update_unchanged': True,
        'family_label_used': False,
        'registered_execution_enabled': False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--registered', action='store_true')
    args = ap.parse_args()
    invariants = validate_invariants()
    if args.registered:
        raise SystemExit(
            'Phase-32 registered execution is deliberately disabled pending separate explicit authority.'
        )
    if args.out:
        Path(args.out).write_text(json.dumps({'status': 'PHASE32_SEEDLESS_VALIDATION_ONLY', 'invariants': invariants}, indent=2))
    print(json.dumps(invariants, indent=2))


if __name__ == '__main__':
    main()
