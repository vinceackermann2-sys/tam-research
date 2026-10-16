import torch
from scripts.si_original_belief_task_v7 import (sample_world,bayesian_decisions,DenseControl,RecurrentControl,
                                      compact_features,realized_reward,initial_log_odds,Config,run)

def test_causal_probe_decision_and_future_mask():
    torch.manual_seed(99400)
    d=sample_world(128,99400)
    shifted={**d,'probe':d['probe']+1234}
    for cls in [DenseControl,RecurrentControl]:
        policy=cls().eval()
        with torch.no_grad():
            before=policy(d,torch.zeros(128,dtype=torch.bool))[0]
            after=policy(shifted,torch.zeros(128,dtype=torch.bool))[0]
        assert torch.equal(before,after)
    assert torch.equal(bayesian_decisions(d)[0],bayesian_decisions(shifted)[0])

def test_bayes_closed_form_value_of_information():
    d=sample_world(1000,99400)
    probe,choice,gain,posterior=bayesian_decisions(d)
    assert ((gain>=-1e-7)&(gain<=.5+1e-7)).all()
    assert torch.equal(probe,gain>d['cost'])
    assert torch.isfinite(initial_log_odds(d,.2)).all()

def test_reward_and_determinism():
    d1=sample_world(16,99400,flipped=True)
    d2=sample_world(16,99400,flipped=True)
    assert all(torch.equal(d1[k],d2[k]) for k in d1)
    good=realized_reward(d1['theta'],torch.zeros(16,dtype=torch.bool),d1)
    assert torch.allclose(good,torch.ones_like(good))
    bad=realized_reward(-d1['theta'],torch.zeros(16,dtype=torch.bool),d1)
    assert torch.equal(bad,torch.zeros_like(bad))

def test_differentiable_neural_policies():
    d=sample_world(32,99400)
    for cls in [DenseControl,RecurrentControl]:
        model=cls()
        probe,action=model(d,torch.ones(32,dtype=torch.bool))
        loss=(probe.square()+action.square()).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
