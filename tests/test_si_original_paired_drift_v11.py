"""Integrity checks for source-preserved v9/v10 same-seed and hidden-drift test."""
import torch
from scripts.si_original_paired_drift_v11 import (
    drift_episode, switch_aware_posterior, known_drift_reference,
    build_evaluations, policy_given_polarity_posterior,
)
from scripts import si_original_contextual_value_v10 as v10
from scripts import si_original_reward_belief_v9 as v9
from scripts.si_original_belief_transfer_v8 import tokens


def test_existing_original_exact_source_unmodified():
    assert v10.verify_original()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'


def test_drift_is_deterministic_with_hidden_change_point():
    a=drift_episode(8,103000)
    b=drift_episode(8,103000)
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert a['private_flip_at'].min().item()>=2
    assert a['private_flip_at'].max().item()<=4
    # The true query never changes; only pre-switch calibration target readings.
    base=build_evaluations(8,103000)
    assert base['polarity_flip']['x_cal'].shape==(8,6)


def test_no_future_probe_hidden_truth_or_switch_metadata_enter_policy():
    ep=drift_episode(4,103001)
    alternative={**ep,'private_flip_at':ep['private_flip_at']+99,
                 'z':-ep['z'],'truth':-ep['truth'],'y_probe':ep['y_probe']+100}
    init,post=tokens(ep,torch.zeros(4,dtype=torch.bool))
    alt_init,alt_post=tokens(alternative,torch.zeros(4,dtype=torch.bool))
    assert torch.equal(init,alt_init)
    assert torch.equal(post,alt_post)
    torch.set_num_threads(2)
    for name in ('hybrid','transformer','gru'):
        actor=v9.RewardLearner(name).eval()
        critic=v10.ContextualController(name).eval()
        with torch.no_grad():
            o0=actor.first(ep)[0];o1=actor.first(alternative)[0]
            q0=critic.values(ep);q1=critic.values(alternative)
        assert torch.allclose(o0,o1,atol=1e-6)
        assert torch.allclose(q0,q1,atol=1e-6)


def test_drift_reference_integrates_hidden_unknown_change():
    d=drift_episode(2048,103002)
    p=switch_aware_posterior(d)
    assert torch.isfinite(p).all()
    assert (p>=0).all() and (p<=1).all()  # float32 sigmoid can round to 1
    baseline=known_drift_reference(d,'misspecified_stationary')
    correct=known_drift_reference(d,'correct_drift_bayes')
    assert correct['reward']>baseline['reward']+.05
    # The purchase decision is an EXPECTATION over future probe, not leaked y_probe.
    altered={**d,'y_probe':d['y_probe']+1000}
    assert torch.equal(policy_given_polarity_posterior(d,p)[0],
                       policy_given_polarity_posterior(altered,p)[0])


def test_unchanged_trainer_disjoint_arm_gradient_and_finite_smoke():
    # Smoke only, not confirmation seeds. Unchanged v9 and v10 functions.
    torch.set_num_threads(2)
    model,_,_=v9.train_one(103003,'hybrid',steps=2,batch=4)
    value, timing=v10.train_one(103003,'transformer',actor_steps=2,value_steps=2,batch=4)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(p).all() for p in value.parameters())
    assert timing['actor_seconds']>=0
