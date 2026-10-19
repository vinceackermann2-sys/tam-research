"""History-necessity world: structural checks only, NO scientific scores."""
from dataclasses import fields
from unittest.mock import patch

import pytest
import torch

from tam_research.si_pcma_v1 import PublicObservation, FactualTransition
from tam_research.si_pcma_v1_factual_learning import causal_tokens
from tam_research.si_pcma_v1_ood_world import OODWorldConfig
from tam_research.si_pcma_v1_ood_controls import (
    FixedGuessControl, SlidingWindowFactualControl,
)
from tam_research.si_pcma_v1_history_world import (
    HistoryNecessaryDevelopmentWorld, randomized_history_development_pairs,
)
from tam_research.si_pcma_v1_neural_probe import randomized_pairs_to_causal_samples
from tam_research.si_pcma_v1_strong_controls import fit_classical_two_step
from tam_research.si_pcma_v1_world_ipc import run_ipc_development_sessions


def _direct(policy, config, sessions=2):
    world = HistoryNecessaryDevelopmentWorld(config)
    result = []
    for session in range(sessions):
        if session:
            world.restart_session()
            policy.restart_session()
        trace = []
        for _ in range(config.session_horizon):
            observation = world.observe()
            action = policy.choose(observation)
            event = world.step(action)
            policy.observe(event)
            trace.append(event)
        result.append(tuple(trace))
    return tuple(result)


def test_same_public_symbol_is_ambiguous_over_distinct_hidden_states():
    # TEST-ONLY private-state intervention; these variables never enter the
    # policy/IPC features. Exact same public symbol and budget can require
    # two different correct guesses, so simple symbol/action lookup cannot
    # represent the entire correct-action function.
    cfg = OODWorldConfig(50, "development")
    a = HistoryNecessaryDevelopmentWorld(cfg)
    b = HistoryNecessaryDevelopmentWorld(cfg)
    for w, context, phase in ((a, 0, 0), (b, 1, 2)):
        w._context, w._phase, w._mode, w._offset = context, phase, 0, 0
        w._symbol = w._public_truth()
    assert a.observe() == b.observe() == PublicObservation(0, 6)
    assert a._correct_guess() != b._correct_guess()
    chosen = a._correct_guess()
    assert a.step(chosen).reward == 1.0
    assert b.step(chosen).reward == 0.0


def test_action_changes_hidden_future_and_probe_reports_only_actual_truth():
    cfg = OODWorldConfig(51, "development", "noisy")
    a = HistoryNecessaryDevelopmentWorld(cfg)
    b = HistoryNecessaryDevelopmentWorld(cfg)
    before = (a._context, a._phase, a._mode)
    event = a.step(3)
    assert event.before.steps_remaining == 6
    assert event.after.steps_remaining == 5
    assert event.action == 3 and event.reward == 0.0 and event.fee == .1
    assert event.after.symbol == a._public_truth()
    assert (a._context, a._phase, a._mode) == before
    # Different actually executed actions change phase differently.
    a = HistoryNecessaryDevelopmentWorld(cfg)
    b = HistoryNecessaryDevelopmentWorld(cfg)
    a.step(0)
    b.step(1)
    assert a._phase != b._phase
    assert a._time == b._time == 1


def test_development_only_world_never_admits_heldout_config():
    with pytest.raises(ValueError, match="heldout"):
        HistoryNecessaryDevelopmentWorld(OODWorldConfig(10, "heldout"))
    with patch("tam_research.si_pcma_v1_history_world.HistoryNecessaryDevelopmentWorld") as factory:
        with pytest.raises(ValueError, match="development"):
            randomized_history_development_pairs(
                (OODWorldConfig(1, "development"),
                 OODWorldConfig(2, "heldout")),
                fixture_randomization_seed=4)
        factory.assert_not_called()


def test_history_public_data_contract_excludes_hidden_modes_and_oracle():
    w = HistoryNecessaryDevelopmentWorld(OODWorldConfig(19, "development"))
    obs = w.observe()
    assert [x.name for x in fields(obs)] == ["symbol", "steps_remaining"]
    event = w.step(1)
    assert [x.name for x in fields(event)] == [
        "before", "action", "after", "reward", "fee",
        "time", "provenance", "done",
    ]
    assert not hasattr(event, "_context")
    assert not hasattr(event, "_phase")
    assert not hasattr(event, "_mode")
    assert not hasattr(w, "oracle") and not hasattr(w, "counterfactual")


@pytest.mark.parametrize("scenario", ["stable", "shift", "noisy", "long"])
def test_ipc_history_trajectory_exactly_matches_own_trusted_reference(scenario):
    cfg = OODWorldConfig(61, "development", scenario)
    expected = _direct(FixedGuessControl(1), cfg, sessions=2)
    observed = run_ipc_development_sessions(
        FixedGuessControl(1), cfg, sessions=2, world_kind="history")
    assert observed == expected
    assert [event.time for part in observed for event in part] == list(
        range(1, 2 * cfg.session_horizon + 1))
    assert all(event.provenance == f"history-factual-{event.time}"
               for part in observed for event in part)


def test_ipc_probe_carries_trusted_symbol_and_actual_fee():
    cfg = OODWorldConfig(62, "development", "noisy")
    p1 = SlidingWindowFactualControl(
        window=4, probe_period=3, exploration=0)
    p2 = SlidingWindowFactualControl(
        window=4, probe_period=3, exploration=0)
    direct = _direct(p1, cfg)
    ipc = run_ipc_development_sessions(p2, cfg, world_kind="history")
    assert ipc == direct
    assert sum(e.action == 3 for t in ipc for e in t) == 4
    assert sum(e.fee for t in ipc for e in t) == pytest.approx(.4)


def test_history_shift_preserves_latents_over_session_restart():
    cfg = OODWorldConfig(12, "development", "shift")
    world = HistoryNecessaryDevelopmentWorld(cfg)
    initial = world._mode
    with pytest.raises(RuntimeError):
        world.restart_session()
    for _ in range(6):
        world.step(0)
    before = (world._context, world._phase, world._mode)
    world.restart_session()
    assert (world._context, world._phase, world._mode) == before
    for _ in range(3):
        world.step(0)
    assert world._mode == initial
    world.step(0)
    assert world._mode == (initial + 1) % 3


def test_randomized_history_pairs_have_real_only_targets_and_unique_groups():
    cfgs = (
        OODWorldConfig(71, "development", "stable"),
        OODWorldConfig(72, "development", "long"),
    )
    pairs = randomized_history_development_pairs(
        cfgs, fixture_randomization_seed=5, sessions=2)
    assert len(pairs) == 22
    assert len({(p.world_index,p.session_index,p.pair_index) for p in pairs}) == 22
    assert all(p.assignment_probability == .25 for p in pairs)
    assert all(p.first_event.action == p.assigned_first_action for p in pairs)
    assert all(p.second_event.action < 3 for p in pairs)
    assert all(p.second_event.time == p.first_event.time + 1 for p in pairs)
    samples = randomized_pairs_to_causal_samples(pairs)
    assert len(samples) == len(pairs)
    for pair, sample in zip(pairs, samples):
        assert sample.observation == pair.first_observation
        assert sample.observed_net_reward == pytest.approx(pair.observed_net)
        assert all(x.time < sample.logical_time for x in sample.prior)
        obs, actions, modes = causal_tokens(
            sample.observation, sample.action, sample.prior,
            logical_time=sample.logical_time)
        assert obs[0, -1, 2].item() == 0.0
        assert modes[0,-1].item() == 1
    for kind in ("constant", "symbol", "ridge"):
        model = fit_classical_two_step(kind, samples)
        assert model.training_rows == 22


def test_history_world_rejects_duplicate_development_fixture_before_creation():
    with patch("tam_research.si_pcma_v1_history_world.HistoryNecessaryDevelopmentWorld") as factory:
        with pytest.raises(ValueError, match="duplicate"):
            randomized_history_development_pairs(
                (OODWorldConfig(5, "development"),
                 OODWorldConfig(5, "development")),
                fixture_randomization_seed=3)
        factory.assert_not_called()


def test_ipc_history_type_must_be_explicit_and_heldout_still_denied():
    with pytest.raises(ValueError, match="unsupported"):
        run_ipc_development_sessions(FixedGuessControl(),
            OODWorldConfig(3, "development"), world_kind="unknown")
    with patch("tam_research.si_pcma_v1_world_ipc.subprocess.Popen") as launch:
        with pytest.raises(ValueError, match="heldout"):
            run_ipc_development_sessions(FixedGuessControl(),
                OODWorldConfig(3, "heldout"), world_kind="history")
        launch.assert_not_called()


def test_history_dev_randomized_dataset_stays_exactly_reproducible():
    cfg = (OODWorldConfig(70, "development", "noisy"),)
    a = randomized_history_development_pairs(cfg,
        fixture_randomization_seed=8, sessions=2)
    b = randomized_history_development_pairs(cfg,
        fixture_randomization_seed=8, sessions=2)
    assert a == b
