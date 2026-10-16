"""Canonical-source, privacy and logged-arm tests for v18."""
import inspect
import torch
import torch.nn.functional as F
from scripts.si_original_active_experiment_v17 import world
from scripts.si_original_logged_value_v18 import (
    LoggedController, verify_source, random_first, random_boolean,
    logged_feedback, decoded_label, train_one, score_logged,
)


def test_full_original_source_and_extra_head_parity():
    assert verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    a=LoggedController('hybrid');b=LoggedController('transformer')
    assert sum(p.numel() for p in a.backbone.parameters())==64614
    assert sum(p.numel() for p in b.backbone.parameters())==64637
    assert sum(p.numel() for p in a.parameters())-64614 == sum(p.numel() for p in b.parameters())-64637


def test_unpurchased_second_does_not_change_first_purchase_or_second_stage():
    torch.set_num_threads(2)
    e=world(5,110001)
    alter={**e,'probe_high':e['probe_high']+1000.,'latent_answer':-e['latent_answer']}
    f=torch.ones(10,dtype=torch.long)
    for a in ('hybrid','transformer','gru'):
        m=LoggedController(a).eval()
        with torch.no_grad():
            assert torch.allclose(m.q0(e),m.q0(alter),atol=1e-6)
            assert torch.allclose(m.q1(e,f),m.q1(alter,f),atol=1e-6)
            assert torch.allclose(m.decision(e,f),m.decision(alter,f),atol=1e-6)
        # Once both are purchased, second observation can influence decisions.
        # It is not legitimate input before that purchase.


def test_correctness_feedback_contains_no_private_label_but_allows_post_action_reconstruction():
    e=world(18,110002)
    choice=random_boolean(36,110002)
    correct=logged_feedback(e,choice)
    recovered=decoded_label(choice,correct)
    assert torch.equal(recovered,e['latent_answer'].gt(0).float())
    assert torch.equal(random_first(36,110003),random_first(36,110003))
    assert set(random_first(1000,110003).unique().tolist())=={0,1,2}


def test_fitted_critic_loss_updates_only_observed_arm():
    torch.set_num_threads(2)
    e=world(4,110004)
    m=LoggedController('hybrid')
    q=m.q1(e,torch.ones(8,dtype=torch.long))
    chosen=torch.ones(8,dtype=torch.long)
    F.mse_loss(q.gather(1,chosen[:,None]).squeeze(1),torch.ones(8)).backward()
    assert torch.allclose(m.q1_head.weight.grad[0],torch.zeros_like(m.q1_head.weight.grad[0]))
    assert m.q1_head.weight.grad[1].abs().sum()>0


def test_no_bayesian_teacher_actions_inside_training():
    src=inspect.getsource(train_one)
    assert 'teacher_first(' not in src
    assert 'second_purchase(' not in src
    assert 'score_policy(' not in src
    assert 'decoded_label(' in src
    assert 'random_first(' in src
    assert 'F.mse_loss(' in src


def test_smoke_training_and_finite_adaptive_rewards():
    torch.set_num_threads(2)
    e=world(4,110005)
    for arch in ('hybrid','transformer','gru'):
        m,t=train_one(110006,arch,actor_steps=2,critic_steps=2,batch=8)
        result=score_logged(e,m)
        assert -1.0 <= result['reward']<=1.0
        assert 0<=result['second_buy']<=1
        assert all(torch.isfinite(p).all() for p in m.parameters())
        assert t['actor_cpu_seconds']>=0 and t['critic_cpu_seconds']>=0
