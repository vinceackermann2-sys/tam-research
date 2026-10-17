"""V25 true factual first-action Q0 checks."""
import torch
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_joint_logged_v21 as v21
from scripts.si_original_first_acquisition_v25 import (
    before_first,logged_first_return,train_acquisition,NewFirstPolicy,factual_first_arm_mse)


def model(seed=117000):
    return v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),seed))


def test_exact_original_source_and_canonical_param_budget():
    assert v18.verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in v18.LoggedController('hybrid').backbone.parameters())==64614
    assert sum(p.numel() for p in v18.LoggedController('transformer').backbone.parameters())==64637


def test_first_features_ignore_hidden_outcome_and_unpurchased_sensors():
    e=v19.evaluation_world(5,117001,'base')
    altered={**e,'probe_low':e['probe_low']+1000,'probe_high':e['probe_high']-1000,'latent_answer':-e['latent_answer']}
    assert torch.equal(before_first(e),before_first(altered))
    assert before_first(e).shape==(10,12)


def test_first_rollout_only_buys_intended_arm():
    torch.set_num_threads(2)
    e=v19.evaluation_world(5,117002,'base');m=model(117002)
    # No probe => exact balanced 50% with constant action +1.
    action=torch.zeros(10,dtype=torch.long)
    cont,rew,bought=logged_first_return(m,e,action)
    assert not bought.any()
    assert float(rew.mean())==.5
    assert torch.equal(cont,rew)


def test_return_really_charges_public_costs():
    torch.set_num_threads(2)
    e=v19.evaluation_world(5,117003,'base');m=model(117003)
    action=torch.ones(10,dtype=torch.long)
    cont,net,second=logged_first_return(m,e,action)
    assert torch.allclose(cont-net,e['cost_low'],atol=1e-7)
    assert torch.isfinite(net).all()


def test_no_direct_oracle_reward_in_training():
    import inspect
    source=inspect.getsource(train_acquisition)
    assert 'teacher_first' not in source and 'second_purchase' not in source
    assert 'logged_first_return' in source
    assert 'latent_answer' not in source


def test_random_arm_coverage_and_reproducibility():
    torch.set_num_threads(2)
    m=model(117004)
    a=train_acquisition(m,117004,n_train=600)
    b=train_acquisition(m,117004,n_train=600)
    assert a['counts']==b['counts'] and min(a['counts'])>100
    assert all(torch.equal(x,y) for x,y in zip(a['weights'],b['weights']))


def test_training_never_mutates_original_backbone():
    torch.set_num_threads(2)
    m=model(117005); before=[p.detach().clone() for p in m.parameters()]
    learned=train_acquisition(m,117005,n_train=600)
    assert all(torch.equal(x,p) for x,p in zip(before,m.parameters()))
    e=v19.evaluation_world(8,117005,'base')
    policy=NewFirstPolicy(m,learned,'ridge')
    assert torch.isfinite(policy.q0(e)).all()
    assert 0<=v18.score_logged(e,policy)['first_high']<=1


def test_public_cost_subtraction_exact_in_q0():
    e=v19.evaluation_world(5,117006,'base');m=model(117006)
    learned=train_acquisition(m,117006,n_train=600)
    policy=NewFirstPolicy(m,learned,'ridge')
    high={**e,'cost_high':e['cost_high']+.1}
    with torch.no_grad():
        q0=policy.q0(e);q1=policy.q0(high)
        f0=before_first(e);f1=before_first(high)
        w=learned['weights'][2]
        assert torch.allclose(q1[:,2]-q0[:,2],(f1-f0)@w-.1,atol=1e-6)


def test_heldout_factual_q0_calibration_finite():
    e=v19.evaluation_world(8,117007,'base');m=model(117007)
    learned=train_acquisition(m,117007,n_train=600)
    policy=NewFirstPolicy(m,learned,'ridge')
    r=factual_first_arm_mse(policy,e,117007)
    assert 0<=r['mse']<10 and sum(r['coverage'].values())==len(e['mu_low'])
