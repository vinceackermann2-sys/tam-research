"""Contract-only engineering fixtures; NEVER promote fixture values as evidence."""
from dataclasses import fields
import pytest

from tam_research.si_pcma_v1 import PublicObservation, FactualTransition
from tam_research.si_pcma_v1_ood_world import (
    DEVELOPMENT_RULES, HELDOUT_RULES, OODWorldConfig, OODRuleWorld,
    engineering_fixture_configs, run_ood_sessions,
)
from tam_research.si_pcma_v1_ood_controls import (
    UniformActionControl, FixedGuessControl, SlidingWindowFactualControl,
    SurpriseResetFactualControl, audit_factual_trace,
)


def test_rule_family_separation_by_exact_truth_table():
    dev, held = set(DEVELOPMENT_RULES), set(HELDOUT_RULES)
    assert len(dev) == len(held) == 3
    assert not dev.intersection(held)
    assert len(dev | held) == 6
    assert all(sorted(rule) == [0, 1, 2] for rule in dev | held)


def test_same_fixture_integer_never_implies_same_train_and_heldout_law():
    # Private fields accessed only in test; DO NOT supply them to policies.
    for i in range(30):
        d = OODRuleWorld(OODWorldConfig(i, "development"))
        h = OODRuleWorld(OODWorldConfig(i, "heldout"))
        assert d._rule in DEVELOPMENT_RULES
        assert h._rule in HELDOUT_RULES
        assert d._rule != h._rule


def test_public_contract_excludes_hidden_law_context_and_seed():
    world = OODRuleWorld(OODWorldConfig(6, "heldout"))
    obs = world.observe()
    assert [x.name for x in fields(obs)] == ["symbol", "steps_remaining"]
    assert obs.symbol in range(3)
    assert obs.steps_remaining == 6
    assert not hasattr(world, "oracle")
    assert not hasattr(world, "counterfactual")


def test_probe_cost_and_factual_trusted_symbol_before_next_action():
    world = OODRuleWorld(OODWorldConfig(7, "development", "noisy"))
    before_context = world._context  # privileged TEST assertion only
    e = world.step(3)
    assert e.action == 3
    assert e.reward == 0.0
    assert e.fee == pytest.approx(0.10)
    assert e.after.symbol == before_context
    assert world._context == before_context
    assert [x.name for x in fields(e)] == [
        "before", "action", "after", "reward", "fee", "time", "provenance", "done"]


def test_guesses_have_factual_reward_and_advance_context():
    world = OODRuleWorld(OODWorldConfig(8, "heldout"))
    context = world._context
    correct = world._rule[context]  # test only; policy never receives this
    e = world.step(correct)
    assert e.reward == 1.0 and e.fee == 0.0
    assert world._context == (context + 1) % 3
    assert e.action == correct and e.time == 1


def test_context_state_survives_restart_and_horizon_is_real():
    cfg = OODWorldConfig(9, "development", "long")
    assert cfg.session_horizon == 16
    world = OODRuleWorld(cfg)
    with pytest.raises(RuntimeError):
        world.restart_session()
    for i in range(16):
        last = world.step(0)
    assert last.done and last.time == 16
    previous_context = world._context
    with pytest.raises(RuntimeError):
        world.step(0)
    new_session = world.restart_session()
    assert new_session.steps_remaining == 16
    assert world._context == previous_context
    assert world.step(2).time == 17


def test_hidden_rule_shift_does_not_expose_shift_marker():
    world = OODRuleWorld(OODWorldConfig(10, "heldout", "shift"))
    before = world._rule
    for _ in range(6):
        world.step(0)
    world.restart_session()
    for _ in range(3):
        world.step(0)
    assert world._rule == before
    world.step(0)  # global time 9: change before executing this choice
    assert world._rule != before
    assert world._rule in HELDOUT_RULES
    assert world._shifted


def test_independent_action_paths_factual_only_and_reproducible():
    cfg = OODWorldConfig(11, "development", "stable")
    a = run_ood_sessions(FixedGuessControl(0), cfg, sessions=2)
    b = run_ood_sessions(FixedGuessControl(0), cfg, sessions=2)
    assert a == b
    assert [e.time for group in a for e in group] == list(range(1, 13))
    assert all(e.action == 0 for group in a for e in group)
    assert all(e.done for e in [a[0][-1], a[1][-1]])
    other = run_ood_sessions(UniformActionControl(2), cfg, sessions=2)
    assert len(other) == 2
    assert all(e.action in range(4) for group in other for e in group)


def test_policy_receives_no_hidden_world_handle():
    class SpyPolicy:
        def __init__(self):
            self.received = []
        def choose(self, observation):
            self.received.append(observation)
            return 0
        def observe(self, event):
            self.received.append(event)
        def restart_session(self, *, erase_persistent=False):
            pass
    spy = SpyPolicy()
    run_ood_sessions(spy, OODWorldConfig(12, "heldout"), sessions=1)
    assert len(spy.received) == 12
    assert all(type(x) in (PublicObservation, FactualTransition)
               for x in spy.received)


def test_sliding_window_and_reset_ablation():
    actor = SlidingWindowFactualControl(window=3, exploration=0.0,
                                        probe_period=0, fixture_seed=0)
    traces = run_ood_sessions(actor, OODWorldConfig(13, "development"), sessions=2)
    assert len(actor.events) == 3
    assert all(e.action != 3 for group in traces for e in group)
    actor.restart_session()
    assert len(actor.events) == 3
    actor.restart_session(erase_persistent=True)
    assert actor.events == ()


def test_fixed_probe_uses_real_budget_and_total_fee():
    actor = SlidingWindowFactualControl(window=4, probe_period=3, exploration=0.0)
    out = run_ood_sessions(actor, OODWorldConfig(14, "heldout"), sessions=2)
    stats = audit_factual_trace(out)
    assert stats.actions == 12
    assert stats.paid_probes == 4
    assert stats.total_fees == pytest.approx(0.4)
    assert stats.net_reward == pytest.approx(stats.gross_reward - 0.4)


def test_surprise_change_control_updates_only_executed_guess():
    actor = SurpriseResetFactualControl(rate=0.20, threshold=0.3)
    fake_before = PublicObservation(0, 6)
    event = FactualTransition(fake_before, 0, PublicObservation(1, 5),
                              0.0, 0.0, 1, "fixture", False)
    actor.q[(0, 0)] = 0.98
    actor.observe(event)
    assert actor.resets == 1
    assert actor.q[(0, 0)] == pytest.approx(1.0 / 3.0)
    assert all(action == 0 for (_, action) in actor.q)
    actor.restart_session(erase_persistent=True)
    assert actor.q == {} and actor.resets == 0


def test_trace_audit_rejects_repeated_or_truncated_feedback():
    a = run_ood_sessions(FixedGuessControl(), OODWorldConfig(1,"development"), sessions=1)
    with pytest.raises(ValueError):
        audit_factual_trace((a[0], a[0]))
    with pytest.raises(ValueError):
        audit_factual_trace((a[0][:-1],))


def test_fixture_configuration_only_no_scientific_seed_reservation():
    c = engineering_fixture_configs()
    assert len(c) == 5
    assert {x.scenario for x in c} == {"stable", "shift", "noisy", "long"}
    assert {x.split for x in c} == {"development", "heldout"}


@pytest.mark.parametrize("kwargs", [
    {"seed": -1, "split": "development"},
    {"seed": 1.2, "split": "development"},
    {"seed": 1, "split": "missing"},
    {"seed": 1, "split": "heldout", "scenario": "other"},
])
def test_invalid_world_plan_rejected(kwargs):
    with pytest.raises(ValueError):
        OODWorldConfig(**kwargs)


def test_invalid_actions_or_session_count_rejected():
    world = OODRuleWorld(OODWorldConfig(1, "development"))
    with pytest.raises(ValueError):
        world.step(4)
    with pytest.raises(ValueError):
        world.step(True)
    with pytest.raises(ValueError):
        run_ood_sessions(FixedGuessControl(), OODWorldConfig(1, "heldout"), sessions=0)
