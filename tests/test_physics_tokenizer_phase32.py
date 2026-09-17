import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tam_research import physics_tokenizer_phase32 as p32


def test_registered_seed_namespace_is_unique_and_disjoint():
    values = []
    for seedset in p32.REGISTERED:
        values.extend(asdict(seedset).values())
    assert len(values) == 87
    assert len(values) == len(set(values))
    assert set(p32.REGISTERED_BASES).isdisjoint(p32.FORBIDDEN_BASES)


def test_protocol_matches_frozen_constants():
    protocol = json.loads(Path('results/physics_tokens/phase32_multihorizon_guard_protocol.json').read_text())
    frozen = protocol['frozen_scientific_configuration']
    assert frozen['context_frames'] == p32.CONTEXT == 8
    assert frozen['ridge'] == p32.RIDGE == 0.001
    assert frozen['median_gain_min'] == p32.MEDIAN_GAIN_MIN == 0.10
    assert tuple(frozen['damping_grid']) == p32.DAMPING_GRID
    windows = protocol['phase32_only_change']['windows']
    expected = tuple(
        (tuple(w['fit_transitions']), w['start_frame'], w['target_frame'])
        for w in windows
    )
    assert expected == p32.GUARD_WINDOWS


def test_guard_is_forward_chaining_context_only_with_543_horizons():
    assert tuple(target - start for _, start, target in p32.GUARD_WINDOWS) == (5, 4, 3)
    for fit, start, target in p32.GUARD_WINDOWS:
        assert target == p32.CONTEXT - 1
        assert max(fit) + 1 <= start
        assert all(0 <= t < p32.CONTEXT - 1 for t in fit)


def test_worst_window_selection_and_tie_break_prefer_larger_lambda():
    # shape [windows, lambdas, trajectories]
    errors = np.array([
        [[0.2, 0.5], [0.3, 0.4], [0.5, 0.3], [0.7, 0.2]],
        [[0.6, 0.5], [0.4, 0.4], [0.3, 0.3], [0.2, 0.2]],
        [[0.4, 0.5], [0.4, 0.4], [0.4, 0.3], [0.4, 0.2]],
    ], dtype=np.float64)
    chosen, best = p32._select_from_window_errors(errors)
    # trajectory 0: lambda 0.5 has worst score .4 versus .6/.5/.7
    assert chosen[0] == np.float32(0.5)
    assert best[0] == pytest.approx(0.4)
    # trajectory 1: lambdas 1.0 and 0.5 tie at worst score .5/.4? Actually lambda 0.5 wins .4.
    assert chosen[1] == np.float32(0.5)

    exact_tie = np.ones((3, 4, 2), dtype=np.float64)
    tied_chosen, tied_best = p32._select_from_window_errors(exact_tie)
    assert np.all(tied_chosen == np.float32(1.0))
    assert np.all(tied_best == 1.0)


def test_phase32_reuses_phase31_scientific_rollout_update():
    # The Phase-32 hypothesis changes only guard selection. Forecast rollouts are delegated
    # to the frozen Phase-31 damped rollout functions rather than defining a new update rule.
    assert p32.p31.damped_model_rollout is not None
    assert p32.p31.damped_selected_rollout is not None
    assert not hasattr(p32, '_damped_quantized_step')


def test_registered_execution_is_fail_closed(monkeypatch):
    monkeypatch.setattr('sys.argv', ['physics_tokenizer_phase32', '--registered'])
    with pytest.raises(SystemExit, match='deliberately disabled'):
        p32.main()


def test_invariants_report_no_registered_execution():
    inv = p32.validate_invariants()
    assert inv['registered_execution_enabled'] is False
    assert inv['guard_context_only'] is True
    assert inv['guard_horizons'] == [5, 4, 3]
    assert inv['phase30_selector_unchanged'] is True
    assert inv['phase31_scientific_rollout_update_unchanged'] is True
    assert inv['family_label_used'] is False
