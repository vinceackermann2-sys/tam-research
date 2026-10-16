"""SI original v20: first-action observed-return causal/provenance controls."""
import inspect
import torch
import torch.nn.functional as F
from scripts.si_original_active_experiment_v17 import world
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_logged_value_v18 as v18
from scripts.si_original_direct_first_value_v20 import (
    logged_first_trajectories, first_logged_q_mse, refit_q0, clone_freeze,
)


def test_canonical_full_original_and_matched_backbone():
    assert v18.verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    for arch,num in (('hybrid',64638),('transformer',64661),('gru',10307)):
        assert sum(p.numel() for p in v18.LoggedController(arch).parameters())==num


def test_factual_returns_ignore_unpurchased_probe():
    torch.set_num_threads(2)
    e=world(6,112001);f=torch.ones(12,dtype=torch.long)
    second=torch.zeros(12,dtype=torch.bool)
    stop=torch.zeros(12,dtype=torch.bool)
    base=v19.CrossfitDoubleQ(v18.LoggedController('hybrid'),112001)
    with torch.no_grad():
        a,ok=logged_first_trajectories(e,base,f,second,stop)
        modified={**e,'probe_high':e['probe_high']+999}
        b,ok2=logged_first_trajectories(modified,base,f,second,stop)
        assert torch.equal(a,b) and torch.equal(ok,ok2)
        assert torch.equal(base.q0(e),base.q0(modified))
        assert torch.equal(base.q1(e,f),base.q1(modified,f))


def test_only_logged_first_arm_and_policy_matching_are_fit():
    torch.set_num_threads(2)
    e=world(8,112002)
    first=torch.tensor([0,0,1,1,2,2,1,2,0,0,1,1,2,2,1,2])
    second=torch.tensor([0,1]*8,dtype=torch.bool)
    stopped=torch.tensor([1,0]*8,dtype=torch.bool)
    model=v19.CrossfitDoubleQ(v18.LoggedController('gru'),112002)
    with torch.no_grad():
        r,ok=logged_first_trajectories(e,model,first,second,stopped)
        assert ok[first==0].all()
        assert ((ok&(first!=0)).sum() <= (first!=0).sum())
        assert torch.isfinite(r).all()
    q=torch.randn(16,3,requires_grad=True)
    F.mse_loss(q.gather(1,first[:,None]).squeeze(1)[ok],r[ok]).backward()
    for i in range(16):
        for arm in range(3):
            if arm!=first[i] or not ok[i]:
                assert q.grad[i,arm] == 0


def test_second_logging_agreement_is_outcome_independent():
    torch.set_num_threads(2)
    e=world(6,112003)
    f=torch.ones(12,dtype=torch.long)
    second=torch.tensor([True,False]*6)
    stop=torch.zeros(12,dtype=torch.bool)
    net=v19.CrossfitDoubleQ(v18.LoggedController('gru'),112003)
    with torch.no_grad():
        _,mask=logged_first_trajectories(e,net,f,second,stop)
        alter={**e,'latent_answer':-e['latent_answer']}
        _,mask2=logged_first_trajectories(alter,net,f,second,stop)
        assert torch.equal(mask,mask2)


def test_q0_refit_preserves_all_actor_and_q1_weights():
    torch.set_num_threads(2)
    base,_=v19.train_one(112004,'gru',actor_steps=2,critic_steps=2,batch=8)
    direct,_=refit_q0(base,112004,'direct',steps=3,batch=8)
    boot,_=refit_q0(base,112004,'bootstrap',steps=3,batch=8)
    for name,param in base.named_parameters():
        if 'q0_head' not in name:
            assert torch.equal(param,dict(direct.named_parameters())[name])
            assert torch.equal(param,dict(boot.named_parameters())[name])


def test_no_analytic_teacher_or_simulated_counterfactual_first_target():
    source=inspect.getsource(refit_q0)
    assert 'teacher_first(' not in source and 'second_purchase(' not in source
    assert 'logged_first_trajectories(' in source
    assert 'crossfit_bootstrap(' in source
    # No direct use of private latent answer outside environment feedback.
    assert "['latent_answer']" not in source


def test_finite_heldout_observed_first_arm_calibration():
    torch.set_num_threads(2)
    base,_=v19.train_one(112005,'transformer',actor_steps=2,critic_steps=2,batch=8)
    e=world(12,112005)
    res=first_logged_q_mse(base,e,112005)
    assert 0 < res['accepted_fraction'] <= 1
    assert 0<=res['mse']<10
    assert sum(res['arm_coverage'].values())>0
