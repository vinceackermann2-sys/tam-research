"""Known-family public Bayesian paid-probe control: factual development tests."""
from dataclasses import replace
from math import isfinite

import pytest

from tam_research.si_pcma_v1 import PublicObservation
from tam_research.si_pcma_v1_belief_probe_falsifier import (
    CYCLIC_FAMILY,HYPOTHESES,NOISE_LEVELS,DEV_SEEDS,SCENARIOS,
    KnownFamilyBeliefPolicy, _uniform_belief,_sensor_probability,
    _branches,_best_guess,_next,frozen_belief_probe_dev_screen,
)
from tam_research.si_pcma_v1_ood_world import OODRuleWorld,OODWorldConfig,run_ood_sessions
from tam_research.si_pcma_v1_world_ipc import run_ipc_development_sessions
from tam_research.si_pcma_v1_ood_controls import audit_factual_trace


def test_known_family_prior_is_public_model_not_selected_hidden_rule():
    assert len(CYCLIC_FAMILY)==3
    assert len(HYPOTHESES)==36 and len(set(HYPOTHESES))==36
    assert NOISE_LEVELS==(.12,.5)
    prior=_uniform_belief()
    assert len(prior)==36
    assert sum(prior.values())==pytest.approx(1.)
    assert not hasattr(KnownFamilyBeliefPolicy,"oracle")
    assert not hasattr(KnownFamilyBeliefPolicy,"hidden_context")
    assert not hasattr(KnownFamilyBeliefPolicy,"get_state")


def test_observation_likelihood_normalizes_for_each_model_hypothesis():
    for h in HYPOTHESES:
        for trusted in (True,False):
            probabilities=tuple(_sensor_probability(h,s,trusted=trusted)
                                for s in range(3))
            assert sum(probabilities)==pytest.approx(1.)
            if trusted:
                assert probabilities[h[1]]==1.
                assert sorted(probabilities)==[0.,0.,1.]
            else:
                assert all(0<p<1 for p in probabilities)


def test_current_guess_and_paid_probe_produce_only_public_future_branches():
    belief=_uniform_belief()
    for action in range(4):
        branches=_branches(belief,action,0)
        assert sum(p for p,_,_,_ in branches)==pytest.approx(1.)
        assert all(0<=symbol<3 for _,_,symbol,_ in branches)
        assert all(0<=reward<=1 for _,reward,_,_ in branches)
        assert all(sum(posterior.values())==pytest.approx(1.)
                   for _,_,_,posterior in branches)
        if action==3:
            assert all(reward==0 for _,reward,_,_ in branches)
            assert all(all(h[1]==symbol for h in posterior)
                       for _,_,symbol,posterior in branches)
        else:
            assert {reward for _,reward,_,_ in branches}=={0,1}
    assert _best_guess(belief,0)[1]==pytest.approx(1/3)


def test_shift_transition_only_at_original_tenth_global_action():
    h=(0,2,0,1)  # rule, context, noise index, shift-enabled
    a,r=_next(h,0,8)
    b,_=_next(h,0,9)
    assert a[0]==0 and b[0]==1
    assert a[1]==b[1]==0
    c,_=_next(b,0,10)
    assert c[0]==1
    h2=(0,2,0,0)
    assert _next(h2,0,9)[0][0]==0
    assert _next(h,3,9)[0][1]==h[1]


def test_policy_resolves_one_actual_action_before_next_query():
    cfg=OODWorldConfig(21,"development","noisy")
    world=OODRuleWorld(cfg)
    agent=KnownFamilyBeliefPolicy(allow_probe=True)
    observation=world.observe()
    action=agent.choose(observation)
    assert action in range(4)
    with pytest.raises(RuntimeError,match="observe actual"):
        agent.choose(observation)
    event=world.step(action)
    with pytest.raises(ValueError,match="mismatched"):
        agent.observe(replace(event,fee=1.0))
    assert agent.observed_events==0
    agent.observe(event)
    assert agent.time==1 and agent.observed_events==1
    assert len(agent.belief)==len(HYPOTHESES) or len(agent.belief)<len(HYPOTHESES)
    assert sum(agent.belief.values())==pytest.approx(1.)
    assert agent.probes==int(action==3)


def test_no_probe_ablation_never_chooses_paid_sensor_even_in_high_noise_world():
    cfg=OODWorldConfig(31,"development","noisy")
    a=KnownFamilyBeliefPolicy(allow_probe=False)
    traces=run_ood_sessions(a,cfg,sessions=2)
    assert a.probes==0
    assert a.observed_events==2*cfg.session_horizon
    assert all(e.action<3 for session in traces for e in session)
    assert audit_factual_trace(traces).total_fees==0


def test_persistent_belief_across_session_restart_and_explicit_erase():
    cfg=OODWorldConfig(41,"development")
    agent=KnownFamilyBeliefPolicy(allow_probe=False)
    world=OODRuleWorld(cfg)
    for _ in range(cfg.session_horizon):
        agent.observe(world.step(agent.choose(world.observe())))
    stored=agent.belief.copy()
    assert len(stored)>0 and agent.time==6
    agent.restart_session(erase_persistent=False)
    assert agent.belief==stored
    assert agent._fresh_sensor
    world.restart_session()
    chosen=agent.choose(world.observe())
    assert chosen in range(3) and not agent._fresh_sensor
    agent.observe(world.step(chosen))
    agent.restart_session(erase_persistent=True)
    assert agent.belief==_uniform_belief()
    assert agent.time==7
    with pytest.raises(ValueError):
        KnownFamilyBeliefPolicy(allow_probe="yes")


def test_public_ipc_matches_same_process_development_traces():
    cfg=OODWorldConfig(55,"development","shift")
    one=KnownFamilyBeliefPolicy(allow_probe=False)
    two=KnownFamilyBeliefPolicy(allow_probe=False)
    direct=run_ood_sessions(one,cfg,sessions=2)
    ipc=run_ipc_development_sessions(two,cfg,sessions=2)
    assert direct==ipc
    assert one.observed_events==two.observed_events==12
    assert one.probes==two.probes==0


def test_device_free_development_probe_and_no_probe_share_fixed_action_budget():
    result=frozen_belief_probe_dev_screen()
    assert result.status=="EXPLORATORY_KNOWN_FAMILY_DEVELOPMENT_ONLY"
    assert result.original_v26=="NO_GO"
    assert result.original_pcma_status=="UNKNOWN"
    assert result.independent_dev_worlds==24
    assert len(DEV_SEEDS)==len(result.per_world)==24
    assert result.sessions_per_world==4
    assert result.total_action_opportunities_per_policy==816
    assert result.scenarios==SCENARIOS
    assert result.total_belief_probe_actions==sum(
        r.belief_allow_probe_count for r in result.per_world)
    assert result.mean_belief_allow_probe_minus_no_probe==pytest.approx(
        sum(r.belief_allow_probe_minus_no_probe
            for r in result.per_world)/24)
    assert all(isfinite(r.belief_allow_probe_net) and
               isfinite(r.belief_no_probe_net) and
               isfinite(r.sliding_window_no_probe_net)
               for r in result.per_world)
    assert all(r.actions in (24,64) for r in result.per_world)
    assert result.neural_models_evaluated==result.scientific_seeds_consumed==0
    assert not result.scientific_gate_passed
    assert not result.learned_architecture_superiority
    assert result.model_informed_world_family_prior


def test_model_and_training_authority_not_leaked_into_public_policy():
    import tam_research.si_pcma_v1_belief_probe_falsifier as module
    assert not hasattr(module,"HELDOUT_RULES")
    assert not hasattr(module.KnownFamilyBeliefPolicy,"world")
    assert not hasattr(module.KnownFamilyBeliefPolicy,"counterfactual_outcomes")
    for action in (-1,4,True):
        with pytest.raises(ValueError):
            _next(HYPOTHESES[0],action,0)
    with pytest.raises(ValueError):
        _sensor_probability(HYPOTHESES[0],3,trusted=False)
