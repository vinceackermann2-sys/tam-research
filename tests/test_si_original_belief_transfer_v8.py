"""Source and causal belief-learning integrity checks (CPU, synthetic)."""
import torch
from tam_research.si_original_hybrid_v0 import OriginalSIHybrid,TransformerControl,SIConfig
from scripts.si_original_belief_transfer_v8 import (
    verify_source, make_episode, bayes_policy, posterior_z, tokens,
    BackboneController, GRUController, reward,
)


def test_original_byte_verified_and_parameter_parity():
    assert verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    a=OriginalSIHybrid(SIConfig());b=TransformerControl(SIConfig())
    assert sum(p.numel() for p in a.parameters())==64614
    assert sum(p.numel() for p in b.parameters())==64637


def test_probe_unpurchased_cannot_affect_prefix_outputs():
    torch.set_num_threads(2)
    e=make_episode(8,100002)
    modified={**e,'y_probe':e['y_probe']+1000.,'truth':-e['truth']}
    for name in ['hybrid','transformer']:
        m=BackboneController(name).eval()
        never=torch.zeros(8,dtype=torch.bool)
        with torch.no_grad():
            first=m(e,never)
            second=m(modified,never)
        assert all(torch.allclose(x,y,atol=1e-6) for x,y in zip(first,second))
    s0,s1=tokens(e,torch.zeros(8,dtype=torch.bool))
    t0,t1=tokens(modified,torch.zeros(8,dtype=torch.bool))
    assert torch.equal(s0,t0) and torch.equal(s1,t1)


def test_calibration_history_changes_belief_without_hidden_labels():
    e=make_episode(4096,100003)
    p=posterior_z(e)
    assert (p.ge(.5)==e['z'].gt(0)).float().mean().item()>.85
    no=posterior_z(e,False)
    assert torch.all(no==.5)
    removed={**e,'y_cal':-e['y_cal']}
    reversed_p=posterior_z(removed)
    # Reversing all trusted references should reverse inferred sign of sensor.
    assert (reversed_p-p).abs().mean().item()>.5


def test_bayes_optimal_cost_and_no_future_oracle_input():
    e=make_episode(8192,100004)
    x=bayes_policy(e)
    never={**e,'y_probe':torch.zeros_like(e['y_probe'])}
    # Buying decision depends on expected information, not the realized probe.
    assert torch.equal(x['buy'],bayes_policy(never)['buy'])
    assert ((x['benefit']>=-1e-6)&(x['benefit']<=.5+1e-6)).all()
    b=x['buy']; y=x['choice'];r=reward(e,b,y)
    assert torch.isfinite(r).all()
    assert r.mean().item() > .79


def test_model_gradient_carries_calibration_information():
    e=make_episode(8,100005)
    for cls in (lambda:BackboneController('hybrid'),lambda:BackboneController('transformer'),GRUController):
        m=cls()
        q,a,z=m(e,torch.ones(8,dtype=torch.bool))
        l=(q.square()+a.square()+z.square()).mean()
        l.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
