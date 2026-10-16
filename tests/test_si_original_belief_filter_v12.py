import torch
import torch.nn.functional as F
from scripts.si_original_belief_filter_v12 import (
    verify_original, episode, belief, hypothesis_log_joint, encoded_tokens,
    Controller, train_one, evaluate, bayes_reference,
)


def test_source_parameter_parity():
    assert verify_original() == '575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    for feature in (False,True):
        assert sum(p.numel() for p in Controller('hybrid',feature).parameters()) == 64614
        assert sum(p.numel() for p in Controller('transformer',feature).parameters()) == 64637


def test_private_labels_and_unpurchased_probe_do_not_affect_belief_or_policy():
    torch.set_num_threads(2)
    e=episode(4,104007,'one_flip')
    changed={**e,'z':-e['z'],'truth':-e['truth'],
             'private_flip_at':e['private_flip_at']+100,
             'y_probe':e['y_probe']+200}
    for u,v in zip(belief(e),belief(changed)):
        assert torch.equal(u,v)
    never=torch.zeros(4,dtype=torch.bool)
    t0,_=encoded_tokens(e,never,True)
    t1,_=encoded_tokens(changed,never,True)
    assert torch.equal(t0,t1)
    for kind in ('hybrid','transformer','gru'):
        torch.manual_seed(42)
        model=Controller(kind,True).eval()
        with torch.no_grad():
            assert torch.allclose(model.values(e),model.values(changed),atol=1e-6)
            assert torch.allclose(model.decision_logits(e,never),model.decision_logits(changed,never),atol=1e-6)


def test_posterior_normalizes_and_recognizes_one_change():
    a=episode(2048,104008,'base')
    b=episode(2048,104009,'one_flip')
    p0,c0=belief(a)
    p1,c1=belief(b)
    assert torch.isfinite(p0).all() and torch.isfinite(p1).all()
    assert ((p0>=0)&(p0<=1)).all()
    assert c1.mean() > c0.mean()+0.2
    assert hypothesis_log_joint(a).shape==(2048,2,4)


def test_identical_observed_data_across_matched_inputs():
    ep=episode(5,104010,'one_flip')
    none=torch.zeros(5,dtype=torch.bool)
    raw,_=encoded_tokens(ep,none,False)
    derived,_=encoded_tokens(ep,none,True)
    assert torch.equal(raw[:,:-1],derived[:,:-1])
    assert torch.equal(raw[:,-1,[0,1,2,4,5]],derived[:,-1,[0,1,2,4,5]])
    assert torch.isfinite(derived).all()


def test_logged_arm_training_finite_and_undisclosed_arm_gradient_zero():
    torch.set_num_threads(2)
    net,t=train_one(104011,'hybrid',True,actor_steps=2,critic_steps=2,batch=6)
    ep=episode(6,104012,'double_flip')
    result=evaluate(net,ep)
    assert 0<=result['probe_fraction']<=1
    assert torch.isfinite(torch.tensor(result['logged_arm_brier']))
    assert all(torch.isfinite(p).all() for p in net.parameters())
    v=Controller('hybrid',True)
    q=v.values(ep)
    buy=torch.ones(6,dtype=torch.long)
    F.binary_cross_entropy_with_logits(q.gather(1,buy[:,None]).squeeze(-1),torch.ones(6)).backward()
    head=v.model.heads.world
    assert torch.equal(head.weight.grad[1],torch.zeros_like(head.weight.grad[1]))
    assert head.weight.grad[2].abs().sum()>0


def test_bayesian_reference_and_double_flip_are_finite():
    for family in ('base','one_flip','early_flip','double_flip','weak_calibration'):
        ep=episode(256,104013,family)
        out=bayes_reference(ep,'mixture')
        assert -1<=out['reward']<=1
        assert 0<=out['probe_fraction']<=1
