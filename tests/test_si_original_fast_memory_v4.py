"""Causality, gradient and isolation controls for fast episodic SI memory."""
import torch
from tam_research.si_original_hybrid_v0 import SIConfig, OriginalSIHybrid, TransformerControl
from scripts.si_original_planning_transfer_v3 import world, model_action_prediction
from scripts.si_original_fast_memory_v4 import FastEpisodicMemory


def test_fast_memory_only_reads_past_completed_transitions():
    torch.manual_seed(97300)
    model=FastEpisodicMemory(OriginalSIHybrid(SIConfig())).eval()
    a=world(3,20,97300,'switch')
    obs,actions=a[:2]
    future=obs.clone()
    future[:,12:]+=7
    with torch.no_grad():
        one=model(obs,actions)['prediction'][:,:12]
        two=model(future,actions)['prediction'][:,:12]
    assert torch.allclose(one,two,atol=1e-6)


def test_memory_resets_on_every_forward_and_query_not_written():
    torch.manual_seed(97300)
    model=FastEpisodicMemory(TransformerControl(SIConfig())).eval()
    a=world(3,20,97300,'direct')
    obs,actions=a[:2]
    with torch.no_grad():
        x=model(obs,actions)['prediction']
        y=model(obs,actions)['prediction']
    assert torch.equal(x,y)
    # The unobserved next state at the final query must not be used.
    a_changed=list(a)
    a_changed[2]=a[2].clone()
    a_changed[2][:,-1]+=500
    with torch.no_grad():
        left=model_action_prediction(model,a)
        right=model_action_prediction(model,tuple(a_changed))
    assert torch.allclose(left,right,atol=1e-6)


def test_key_encoder_receives_gradients_through_episodic_writes():
    torch.manual_seed(97300)
    model=FastEpisodicMemory(OriginalSIHybrid(SIConfig()))
    obs,actions,target=world(3,20,97300,'delayed')[:3]
    out=model(obs,actions)
    loss=(out['prediction']-target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.key_encoder[0].weight.grad is not None
    assert model.key_encoder[0].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_same_memory_parameter_increment_for_both_backbones():
    c=SIConfig()
    original=OriginalSIHybrid(c)
    transformer=TransformerControl(c)
    augmented_original=FastEpisodicMemory(OriginalSIHybrid(c))
    augmented_trans=FastEpisodicMemory(TransformerControl(c))
    count=lambda m:sum(p.numel() for p in m.parameters())
    assert count(augmented_original)-count(original)==913
    assert count(augmented_trans)-count(transformer)==913
