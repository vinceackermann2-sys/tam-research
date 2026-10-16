"""Test strict causal contextual critic, source identity and feedback contracts."""
import torch
import torch.nn.functional as F
from scripts.si_original_contextual_value_v10 import (
    ContextualController, verify_original, rand_buy, feedback, train_one,
    evaluate, randomized_value_calibration,
)
from scripts.si_original_belief_transfer_v8 import make_episode


def test_full_original_source_and_equal_nominal_budget():
    assert verify_original() == '575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    h = ContextualController('hybrid')
    t = ContextualController('transformer')
    assert sum(p.numel() for p in h.parameters()) == 64614
    assert sum(p.numel() for p in t.parameters()) == 64637


def test_unpurchased_measurements_and_hidden_labels_not_used():
    torch.set_num_threads(2)
    e = make_episode(6, 102001)
    altered={**e, 'y_probe':e['y_probe']+200,
             'truth':-e['truth'], 'z':-e['z']}
    none=torch.zeros(6,dtype=torch.bool)
    for arch in ('hybrid','transformer','gru'):
        net=ContextualController(arch).eval()
        with torch.no_grad():
            assert torch.allclose(net.values(e),net.values(altered),atol=1e-6)
            assert torch.allclose(net.decision_logits(e,none),net.decision_logits(altered,none),atol=1e-6)


def test_only_action_correctness_feedback_reconstructs_true_action():
    e=make_episode(12, 102002)
    act=rand_buy(12,102003)
    purchase=rand_buy(12,102004)
    success,label,net=feedback(e,act,purchase)
    assert torch.equal(label,e['truth'].gt(0).float())
    assert torch.allclose(net,success-purchase.float()*e['c'])


def test_critic_gradient_only_on_randomly_observed_arm():
    torch.manual_seed(102005)
    for arch in ('hybrid','transformer','gru'):
        model=ContextualController(arch)
        e=make_episode(10,102005)
        q=model.values(e)
        use=torch.ones(10,dtype=torch.long) # all buy; q_skip is NOT observed
        selected=q.gather(1,use[:,None]).squeeze(1)
        F.binary_cross_entropy_with_logits(selected,torch.ones(10)).backward()
        head=model.head if arch=='gru' else model.model.heads.world
        assert torch.allclose(head.weight.grad[1],torch.zeros_like(head.weight.grad[1]))
        assert head.weight.grad[2].abs().sum()>0


def test_bounded_training_finite_and_reproducible_value_sampling():
    torch.set_num_threads(2)
    assert torch.equal(rand_buy(12,102006),rand_buy(12,102006))
    model,t=train_one(102007,'hybrid',actor_steps=2,value_steps=2,batch=6)
    ep=make_episode(6,102008)
    metrics=evaluate(model,ep)
    assert -1 <= metrics['reward'] <=1
    assert torch.isfinite(torch.tensor(randomized_value_calibration(model,ep,102009)))
    assert all(torch.isfinite(p).all() for p in model.parameters())
