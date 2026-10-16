"""v19 exact-source, logged-arm, cross-fit, twin and causal tests."""
import inspect
import torch
import torch.nn.functional as F
from scripts.si_original_active_experiment_v17 import world
from scripts.si_original_logged_value_v18 import train_one as old_train, LoggedController
from scripts.si_original_crossfit_value_v19 import (
    evaluation_world, CrossfitDoubleQ, crossfit_bootstrap, crossfit_selected_losses,
    train_one, logged_arm_calibration)
from scripts.si_original_logged_value_v18 import verify_source


def test_canonical_source_and_nominal_parity():
    assert verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in LoggedController('hybrid').parameters())==64638
    assert sum(p.numel() for p in LoggedController('transformer').parameters())==64661
    assert sum(p.numel() for p in LoggedController('gru').parameters())==10307


def test_crossfit_selected_arm_only_and_disjoint_folds():
    a=torch.randn(6,2,requires_grad=True)
    b=torch.randn(6,2,requires_grad=True)
    bought=torch.tensor([0,1,0,1,1,0])
    targets=torch.linspace(0,1,6)
    crossfit_selected_losses(a,b,bought,targets).backward()
    for i in range(6):
        assert a.grad[i].abs().sum()>0 if i%2==0 else a.grad[i].abs().sum()==0
        assert b.grad[i].abs().sum()>0 if i%2==1 else b.grad[i].abs().sum()==0
        assert a.grad[i,1-bought[i]]==0
        assert b.grad[i,1-bought[i]]==0


def test_double_q_action_selection_value_separation():
    a=torch.tensor([[1.,3.],[9.,-5.]])
    b=torch.tensor([[100.,2.],[4.,9.]])
    price=torch.tensor([.2,.4])
    y=crossfit_bootstrap(a,b,price)
    assert torch.allclose(y,torch.tensor([1.8,3.6]))
    assert not torch.allclose(y,torch.max(a,b).max(-1).values)


def test_no_future_probe_and_hidden_label_leakage():
    torch.set_num_threads(2)
    ep=world(4,111001)
    changed={**ep,'probe_high':ep['probe_high']+99,'latent_answer':-ep['latent_answer']}
    base=LoggedController('hybrid')
    net=CrossfitDoubleQ(base,111001)
    first=torch.ones(8,dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(net.q0(ep),net.q0(changed))
        assert torch.equal(net.q1(ep,first),net.q1(changed,first))
        assert torch.equal(net.decision(ep,first),net.decision(changed,first))


def test_actor_training_identical_to_v18_without_critic():
    torch.set_num_threads(2)
    base,_=old_train(111002,'transformer',actor_steps=3,critic_steps=0,batch=8)
    cross,_=train_one(111002,'transformer',actor_steps=3,critic_steps=1,batch=8)
    for (p,x),(q,y) in zip(base.backbone.named_parameters(),cross.actor.backbone.named_parameters()):
        assert p==q and torch.equal(x,y)
    assert sum(p.numel() for p in cross.parameters())==sum(p.numel() for p in base.parameters())+8


def test_strong_discount_is_public_cost_only_and_preprobe_equal():
    base=world(12,111003,'base')
    other=evaluation_world(12,111003,'strong_discount_ood')
    for k in ('latent_answer','probe_low','probe_high','mu_low','mu_high','observed_query'):
        assert torch.equal(base[k],other[k])
    assert (other['cost_high']<=base['cost_high']).all()
    assert (other['cost_low']>base['cost_low']).all()
    assert torch.equal(other['cost_high'][:12],other['cost_high'][12:])


def test_no_teacher_labels_and_finite_logged_calibration():
    src=inspect.getsource(train_one)
    assert 'teacher_first' not in src and 'second_purchase(' not in src
    assert 'v18.logged_feedback' in src
    torch.set_num_threads(2)
    m,_=train_one(111004,'gru',actor_steps=2,critic_steps=2,batch=8)
    ep=world(4,111005)
    err=logged_arm_calibration(m,ep,111005)
    assert 0<=err<10
    assert all(torch.isfinite(p).all() for p in m.parameters())
