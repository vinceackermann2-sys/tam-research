"""The model may learn ONLY from observable predictions/actions/feedback, not z labels."""
import torch
from scripts.si_original_reward_belief_v9 import (RewardLearner,verify_source,observed_feedback,
    inferred_outcome_label, self_supervised_target, train_one, evaluate_model)
from scripts.si_original_belief_transfer_v8 import make_episode


def test_source_exact_and_model_parameters():
    assert verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in RewardLearner('hybrid').parameters())==64614
    assert sum(p.numel() for p in RewardLearner('transformer').parameters())==64637


def test_probe_future_and_hidden_truth_never_enter_network():
    torch.set_num_threads(2)
    ep=make_episode(3,101000)
    modified={**ep,'y_probe':ep['y_probe']+200, 'z':-ep['z'], 'truth':-ep['truth']}
    for arch in ('hybrid','transformer','gru'):
        torch.manual_seed(101000)
        m=RewardLearner(arch).eval()
        with torch.no_grad():
            a=m.first(ep); b=m.first(modified)
            c=m.finish(ep,torch.zeros(3,dtype=torch.bool))
            d=m.finish(modified,torch.zeros(3,dtype=torch.bool))
        assert all(torch.allclose(x,y,atol=1e-6) for x,y in zip(a,b))
        assert torch.allclose(c,d,atol=1e-6)


def test_infer_label_from_post_action_feedback():
    ep=make_episode(24,101001)
    random_action=torch.randint(0,2,(24,),generator=torch.Generator().manual_seed(27)).bool()
    bought=torch.randint(0,2,(24,),generator=torch.Generator().manual_seed(28)).bool()
    feedback,net=observed_feedback(ep,bought,random_action)
    inferred=inferred_outcome_label(random_action,feedback)
    # Only as a test: evaluate whether action + feedback reconstructs the truth.
    assert torch.equal(inferred,ep['truth'].gt(0).float())
    assert torch.isfinite(net).all()


def test_self_supervision_only_next_observable_calibration_values():
    ep=make_episode(8,101002)
    target=self_supervised_target(ep)
    assert target.shape==(8,5)
    changed={**ep,'z':-ep['z'],'truth':-ep['truth'],'y_probe':ep['y_probe']+500}
    assert torch.equal(target,self_supervised_target(changed))


def test_smoke_gradients_through_causal_reward_learning():
    torch.set_num_threads(2)
    for arch in ('hybrid','transformer','gru'):
        model,_,metrics=train_one(101003,arch,steps=2,batch=6)
        assert len(metrics)==2
        assert all(torch.isfinite(p).all() for p in model.parameters())
        e=make_episode(6,101004)
        out=evaluate_model(model,e)
        assert -1 < out['reward'] <= 1
