import torch
from tam_research.si_original_hybrid_v0 import (
    SIConfig, OriginalSIHybrid, TransformerControl,
    EpisodicMemory, generate_world_batch, prediction_loss, count_parameters,
)


def test_forward_backward_all_modules():
    torch.manual_seed(1)
    config=SIConfig()
    model=OriginalSIHybrid(config)
    x, actions, target=generate_world_batch(2, 12, 412)
    out=model(x,actions)
    assert out["prediction"].shape == target.shape
    assert len(out["expert_routes"])==config.layers
    loss,_=prediction_loss(out,target)
    loss.backward()
    assert torch.isfinite(loss)
    for name in ["encoder", "mixer", "router", "experts", "refiner", "halt", "heads"]:
        assert any(name in key and p.grad is not None and torch.isfinite(p.grad).all() for key,p in model.named_parameters()), name


def test_strict_causality():
    torch.manual_seed(4)
    model=OriginalSIHybrid(SIConfig()).eval()
    x,a,_=generate_world_batch(2,12,554)
    y=model(x,a)["prediction"].detach()
    x2=x.clone();a2=a.clone();x2[:,8:]+=100.;a2[:,8:]=0.
    y2=model(x2,a2)["prediction"].detach()
    assert torch.allclose(y[:,:8],y2[:,:8],atol=1e-5)


def test_masked_compute_and_world_model_interfaces():
    model=OriginalSIHybrid(SIConfig()).eval()
    x,a,t=generate_world_batch(2,8,555)
    assert model(x,a,skip_optional_refine=True)["did_refine"] is False
    assert model.plan_one_step(x,a,torch.eye(4)[None].expand(2,-1,-1),t[:,-1]).shape == (2,)
    assert abs(count_parameters(model)-count_parameters(TransformerControl(SIConfig())))/count_parameters(model)<0.001


def test_versioned_external_memory():
    store=EpisodicMemory(2)
    store.write_verified(1,torch.tensor([1.,0.]),torch.tensor([11.]))
    store.write_verified(2,torch.tensor([1.,0.]),torch.tensor([99.]))
    assert store.read(1,torch.tensor([1.,0.])).item()==11.
    assert store.read(3,torch.tensor([1.,0.])) is None
    store.write_verified(1,torch.tensor([0.,1.]),torch.tensor([22.]))
    assert store.read(2,torch.tensor([1.,0.])).item()==99.
    assert store.read(1,torch.tensor([1.,0.])).item()==22.


def test_reproducible_world_batches():
    b1=generate_world_batch(2,8,67)
    b2=generate_world_batch(2,8,67)
    assert all(torch.equal(a,b) for a,b in zip(b1,b2))
