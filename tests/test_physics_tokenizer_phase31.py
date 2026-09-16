from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase30 as p30
from tam_research import physics_tokenizer_phase31 as p31
from tam_research import physics_tokenizer_phase31_rep1_attempt1 as r31


def test_phase31_registered_seed_namespace_is_unique_and_fresh():
    new_values = []
    for seedset in p31.REGISTERED:
        new_values.extend(asdict(seedset).values())
    assert len(new_values) == 87
    assert len(new_values) == len(set(new_values))

    forbidden_values = set()
    for base in p31.FORBIDDEN_BASES:
        forbidden_values.update(asdict(p30._seedset(base)).values())
    assert not set(new_values).intersection(forbidden_values)


def test_phase31_protocol_matches_frozen_code_constants():
    payload = json.loads(
        Path('results/physics_tokens/phase31_recursive_damping_protocol.json').read_text()
    )
    assert tuple(payload['damping_grid']) == p31.DAMPING_GRID
    assert tuple(payload['guard_fit_transitions']) == p31.GUARD_TRANSITIONS
    assert payload['guard_start_frame'] == p31.GUARD_START_FRAME
    assert payload['guard_target_frame'] == p31.GUARD_TARGET_FRAME
    assert payload['guard_recursive_horizon'] == p31.GUARD_HORIZON
    assert [x['seed_base'] for x in payload['registered_replicates']] == list(p31.REGISTERED_BASES)
    assert '69/69' in payload['frozen_pass_rule']


def test_phase31_guard_is_context_only_and_has_no_persistence_candidate():
    assert max(p31.GUARD_TRANSITIONS) < p31.GUARD_START_FRAME
    assert p31.GUARD_TARGET_FRAME < p31.CONTEXT
    assert p31.GUARD_HORIZON == 3
    assert p31.DAMPING_GRID == tuple(sorted(p31.DAMPING_GRID, reverse=True))
    assert p31.DAMPING_GRID[0] == 1.0
    assert min(p31.DAMPING_GRID) > 0.0
    assert 0.0 not in p31.DAMPING_GRID


def test_phase31_lambda_one_is_exact_phase30_companded_update():
    cur = np.linspace(-0.4, 0.4, 36, dtype=np.float32).reshape(2, 2, 3, 3)
    delta = np.linspace(-0.08, 0.12, 36, dtype=np.float32).reshape(2, 2, 3, 3)
    y = (cur + delta).astype(np.float32)
    dm = np.array([0.01, -0.02], dtype=np.float32)
    ds = np.array([0.15, 0.25], dtype=np.float32)

    got, _ = p31._damped_quantized_step(
        cur, y, np.ones(2, dtype=np.float32), dm, ds
    )
    expected = (cur + p10.q_comp_delta(y - cur, dm, ds)).astype(np.float32)
    np.testing.assert_array_equal(got, expected)


def test_phase31_per_trajectory_mse_is_not_batch_averaged():
    a = np.zeros((2, 1, 2, 2), dtype=np.float32)
    b = np.zeros_like(a)
    b[1] = 2.0
    got = p31._per_trajectory_mse(a, b)
    assert got.shape == (2,)
    assert got[0] == 0.0
    assert got[1] == 4.0


def test_phase31_registered_cli_is_fail_closed(monkeypatch, tmp_path):
    out = tmp_path / 'must_not_exist.json'
    monkeypatch.setattr(
        sys,
        'argv',
        ['phase31', '--out', str(out), '--rep', '1'],
    )
    with pytest.raises(SystemExit, match='registered execution is deliberately disabled'):
        p31.main()
    assert not out.exists()


def test_phase31_invariants_report_pre_authority_state():
    inv = p31.validate_invariants()
    assert inv['registered_seed_count'] == 87
    assert inv['registered_bases'] == list(p31.REGISTERED_BASES)
    assert inv['phase30_selector_unchanged'] is True
    assert inv['family_label_used'] is False
    assert inv['registered_execution_enabled'] is False


def test_phase31_rep1_attempt_is_hard_locked_to_authorized_seed_only():
    inv = r31.validate_invariants()
    assert r31.ATTEMPT_ID == 'phase31-registered-rep1-attempt1'
    assert r31.REPLICATE == 1
    assert r31.SEED_BASE == 831310000
    assert r31.LOCKED_FUTURE_BASES == (831310100, 831310200)
    assert inv['seed_base'] == 831310000
    assert inv['future_registered_bases_locked'] == [831310100, 831310200]
    assert inv['expected_primary_cells'] == 23


def _write_fake_family_results(tmp_path, failing_split=None):
    paths = []
    for family in r31.FAMILIES:
        names = [x for x in r31.EXPECTED_SPLITS if r31.split_family(x) == family]
        ratios = {name: (1.0 if name == failing_split else 0.5) for name in names}
        payload = {
            'attempt_id': r31.ATTEMPT_ID,
            'replicate': r31.REPLICATE,
            'seed_base': r31.SEED_BASE,
            'family': family,
            'primary_ratios': ratios,
            'passing_cells': sum(x < 1.0 for x in ratios.values()),
            'total_cells': len(ratios),
            'all_below_persistence': all(x < 1.0 for x in ratios.values()),
        }
        path = tmp_path / f'{family}.json'
        path.write_text(json.dumps(payload))
        paths.append(str(path))
    return paths


def test_phase31_rep1_aggregate_any_ratio_ge_one_is_decisive_fail(tmp_path):
    paths = _write_fake_family_results(tmp_path, failing_split='nls_id_h8')
    out = tmp_path / 'summary.json'
    result = r31.aggregate(paths, str(out))
    assert result['status'] == 'PHASE31_DECISIVE_FAIL_REGISTERED_REP1'
    assert result['overall_gate_status'] == 'FAIL_DECISIVE'
    assert result['failing_cells'] == {'nls_id_h8': 1.0}
    assert result['replicate_2_status'] == 'LOCKED_UNCONSUMED'
    assert result['replicate_3_status'] == 'LOCKED_UNCONSUMED'
    assert result['retry_authorized'] is False


def test_phase31_rep1_aggregate_23_of_23_keeps_overall_gate_unresolved(tmp_path):
    paths = _write_fake_family_results(tmp_path)
    out = tmp_path / 'summary.json'
    result = r31.aggregate(paths, str(out))
    assert result['status'] == 'PHASE31_REGISTERED_REP1_PASS_23_OF_23_OVERALL_GATE_UNRESOLVED'
    assert result['primary_passing_cells'] == 23
    assert result['failing_cells'] == {}
    assert result['overall_gate_status'] == 'UNRESOLVED_REQUIRES_SEPARATE_REP2_AUTHORIZATION'
    assert result['replicate_2_status'] == 'LOCKED_UNCONSUMED'


def test_phase31_rep1_aggregate_rejects_missing_family(tmp_path):
    paths = _write_fake_family_results(tmp_path)[:-1]
    with pytest.raises(RuntimeError, match='exactly 6 family files required'):
        r31.aggregate(paths, str(tmp_path / 'summary.json'))
