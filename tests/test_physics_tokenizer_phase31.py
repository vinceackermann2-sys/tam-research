from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tam_research import physics_tokenizer_phase30 as p30
from tam_research import physics_tokenizer_phase31 as p31


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
    assert min(p31.DAMPING_GRID) > 0.0
    assert 0.0 not in p31.DAMPING_GRID


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
