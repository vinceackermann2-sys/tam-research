"""Causal two-stage experimental design, exact twin certificate and Bayes DP checks."""
import torch
import torch.nn.functional as F
from scripts.si_original_active_experiment_v17 import (
    verify_original, SequentialController, world, token_data, teacher_first,
    second_purchase, score_policy, normal_cdf, second_values,
    expected_first_value, subset,
)


def test_original_hash_and_parameter_parity():
    assert verify_original()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in SequentialController('hybrid').parameters())==64614
    assert sum(p.numel() for p in SequentialController('transformer').parameters())==64637


def test_counterfactual_twins_have_identical_first_stage_observation():
    e=world(16,109001)
    for k in ('observed_calibration_x','observed_calibration_y','observed_query',
              'mu_low','mu_high','cost_low','cost_high'):
        assert torch.equal(e[k][:16],e[k][16:])
    assert torch.equal(e['latent_answer'][:16],-e['latent_answer'][16:])
    assert torch.equal(token_data(subset(e,torch.arange(32)<16)),
                       token_data(subset(e,torch.arange(32)>=16)))


def test_purchase_outcomes_are_excluded_from_first_and_second_decisions():
    torch.set_num_threads(2)
    e=world(6,109002)
    altered={**e,'probe_high':e['probe_high']+1000,'latent_answer':-e['latent_answer']}
    assert torch.equal(token_data(e),token_data(altered))
    first=torch.ones(12,dtype=torch.long)  # cheap probe purchased, high NOT
    assert torch.equal(token_data(e,first),token_data(altered,first))
    for name in ('hybrid','transformer','gru'):
        model=SequentialController(name).eval()
        with torch.no_grad():
            assert torch.allclose(model(token_data(e)),model(token_data(altered)),atol=1e-6)
            assert torch.allclose(model(token_data(e,first)),model(token_data(altered,first)),atol=1e-6)
    assert not torch.equal(token_data(e,first,torch.full_like(first,2)),
                           token_data(altered,first,torch.full_like(first,2)))


def test_second_stage_policy_depends_on_observed_first_only():
    e=world(20,109003)
    first=torch.ones(40,dtype=torch.long)
    s,l=second_purchase(e,first)
    changed={**e,'probe_high':e['probe_high']+1000}
    t,m=second_purchase(changed,first)
    assert torch.equal(s,t) and torch.equal(l,m)
    # Very high known cost -> never buy the remaining probe.
    _,buy,_=second_values(torch.zeros(40),e['mu_high'],torch.ones(40))
    assert not buy.any()


def test_dynamic_bayes_quadrature_stable_and_knowledge_control():
    torch.set_num_threads(2)
    e=world(300,109004)
    low=e['mu_low'];cost=e['cost_low'];other=e['mu_high'];other_cost=e['cost_high']
    v24=expected_first_value(low,cost,other,other_cost,24)
    v48=expected_first_value(low,cost,other,other_cost,48)
    assert float((v24-v48).abs().mean())<.004
    first,values=teacher_first(e)
    assert values.shape==(600,3)
    assert torch.isfinite(values).all()
    assert ((first>=0)&(first<=2)).all()
    assert (values[:,0]==.5).all()


def test_strong_bayes_probe_controls_and_cost_scoring():
    torch.set_num_threads(2)
    e=world(2048,109005)
    never=score_policy(e,'never')
    low=score_policy(e,'always_low')
    both=score_policy(e,'always_both')
    bayes=score_policy(e,'bayes')
    assert never['reward']==.5 and never['avg_probe_count']==0
    assert both['avg_probe_count']==2
    assert low['avg_probe_count']==1
    assert bayes['reward']>never['reward']+.15
    assert bayes['reward']>both['reward']+.03
    assert 0<bayes['second_buy']<.8


def test_differentiable_neural_policy_outputs():
    torch.set_num_threads(2)
    e=world(5,109006)
    first=torch.ones(10,dtype=torch.long)
    other=torch.full_like(first,2)
    for name in ('hybrid','transformer','gru'):
        model=SequentialController(name)
        y0=model(token_data(e));y1=model(token_data(e,first));y2=model(token_data(e,first,other))
        assert y0.shape==y1.shape==y2.shape==(10,3)
        loss=F.cross_entropy(y0,torch.zeros(10,dtype=torch.long))+(y1.square()+y2.square()).mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
