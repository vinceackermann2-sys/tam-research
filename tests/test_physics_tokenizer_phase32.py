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
    # trajectory 1: lambda 0.125 has the smallest worst-window score (.2).
    assert chosen[1] == np.float32(0.125)
    assert best[1] == pytest.approx(0.2)

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


from tam_research import physics_tokenizer_phase32_rep1_attempt1 as rep1


def test_rep1_runner_matches_frozen_phase32_namespace():
    inv = rep1.validate_invariants()
    assert inv['attempt_id'] == 'phase32-registered-rep1-attempt1'
    assert inv['replicate'] == 1
    assert inv['seed_base'] == 832320000
    assert inv['registered_seed_count'] == 29
    assert inv['future_registered_bases_locked'] == [832320100, 832320200]
    assert inv['expected_primary_cells'] == 23
    assert inv['authorization_required'] is True


def test_rep1_authorization_ledger_is_closed_and_untouched():
    auth = json.loads(Path('results/physics_tokens/phase32_rep1_execution_authorization.json').read_text())
    assert auth['status'] == 'NOT_AUTHORIZED_PENDING_EXPLICIT_USER_AUTHORIZATION'
    assert auth['authorization_scope']['replicate'] == 1
    assert auth['authorization_scope']['seed_base'] == 832320000
    assert auth['authorization_scope']['attempt_id'] == 'phase32-registered-rep1-attempt1'
    assert auth['registered_seed_bases']['replicate_1']['status'] == 'UNTOUCHED_NOT_AUTHORIZED'
    assert auth['registered_seed_bases']['replicate_2']['status'] == 'LOCKED_UNCONSUMED'
    assert auth['registered_seed_bases']['replicate_3']['status'] == 'LOCKED_UNCONSUMED'


def test_rep1_prepare_cannot_reach_scientific_build_without_authorization(tmp_path, monkeypatch):
    touched = {'build_all': False}

    def forbidden_build_all(*args, **kwargs):
        touched['build_all'] = True
        raise AssertionError('scientific build_all must not run before authorization')

    monkeypatch.setattr(rep1.p30, 'build_all', forbidden_build_all)
    with pytest.raises(RuntimeError, match='not authorized'):
        rep1.prepare(str(tmp_path / 'prep.pkl'), str(tmp_path / 'marker.json'))
    assert touched['build_all'] is False
    assert not (tmp_path / 'marker.json').exists()


def _write_family_records(tmp_path, failing_name=None):
    paths = []
    for family in rep1.FAMILIES:
        names = [n for n in rep1.EXPECTED_SPLITS if rep1.split_family(n) == family]
        ratios = {n: (1.25 if n == failing_name else 0.5) for n in names}
        payload = {
            'attempt_id': rep1.ATTEMPT_ID,
            'replicate': rep1.REPLICATE,
            'seed_base': rep1.SEED_BASE,
            'family': family,
            'primary_ratios': ratios,
            'passing_cells': sum(v < 1.0 for v in ratios.values()),
            'total_cells': len(ratios),
            'all_below_persistence': all(v < 1.0 for v in ratios.values()),
        }
        p = tmp_path / f'{family}.json'
        p.write_text(json.dumps(payload))
        paths.append(str(p))
    return paths


def test_rep1_aggregate_enforces_decisive_23_cell_stop_rule(tmp_path):
    paths = _write_family_records(tmp_path, failing_name='hj_id_h8')
    out = tmp_path / 'summary.json'
    result = rep1.aggregate(paths, str(out))
    assert result['primary_total_cells'] == 23
    assert result['primary_passing_cells'] == 22
    assert result['failing_cells'] == {'hj_id_h8': 1.25}
    assert result['overall_gate_status'] == 'FAIL_DECISIVE'
    assert result['replicate_2_status'] == 'LOCKED_UNCONSUMED'
    assert result['replicate_3_status'] == 'LOCKED_UNCONSUMED'
    assert result['retry_authorized'] is False


def test_rep1_aggregate_23_of_23_still_requires_rep2_authorization(tmp_path):
    paths = _write_family_records(tmp_path)
    out = tmp_path / 'summary.json'
    result = rep1.aggregate(paths, str(out))
    assert result['primary_passing_cells'] == 23
    assert result['replicate_all_23_below_persistence'] is True
    assert result['overall_gate_status'] == 'UNRESOLVED_REQUIRES_SEPARATE_REP2_AUTHORIZATION'
    assert result['replicate_2_status'] == 'LOCKED_UNCONSUMED'
