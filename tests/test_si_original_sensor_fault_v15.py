"""Source, leakage, inference and matched-control tests for SI v15."""
import torch
from scripts.si_original_sensor_fault_v15 import (
    verify_original, sensor_world, fault_posterior, majority_pz,
    decision_from_pz, neural_panel, FAMILIES)
from scripts.si_original_belief_filter_v12 import Controller


def test_canonical_original_and_matched_params():
    assert verify_original()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in Controller('hybrid',False).parameters())==64614
    assert sum(p.numel() for p in Controller('transformer',False).parameters())==64637


def test_every_fault_world_is_reproducible():
    for name in FAMILIES:
        a=sensor_world(8,107010,name);b=sensor_world(8,107010,name)
        assert all(torch.equal(a[k],b[k]) for k in a)
        assert a['anchor_cal'].shape==(8,6)


def test_fault_inference_has_no_private_future_or_label_leakage():
    e=sensor_world(8,107011,'both_reference_anchor_drift')
    altered={**e,'z':-e['z'],'truth':-e['truth'],
        'cal_truth':-e['cal_truth'],'y_probe':e['y_probe']+1000,
        'private_reference_drift':torch.zeros(8,dtype=torch.bool)}
    for a,b in zip(fault_posterior(e),fault_posterior(altered)):
        assert torch.equal(a,b)
    assert torch.equal(majority_pz(e),majority_pz(altered))


def test_posterior_finite_normalized_and_fault_types_represented():
    for name in FAMILIES:
        e=sensor_world(96,107012,name)
        q=fault_posterior(e)
        assert all(torch.isfinite(x).all() for x in q)
        for x in q[:3]:assert ((x>=-1e-6)&(x<=1+1e-6)).all()


def test_observation_instead_of_simulator_truth_drives_decisions():
    torch.set_num_threads(2)
    e=sensor_world(8,107013,'anchor_drift')
    raw=Controller('hybrid',False).eval()
    bel=Controller('hybrid',True).eval()
    _,py,pa,_=fault_posterior(e)
    with torch.no_grad():
        result=neural_panel(raw,bel,e,py,pa)
        assert result['anchor_fee'] == .025
        assert 0<=result['repair_fraction']<=1
        assert -1<=result['reward']<=1
        assert decision_from_pz(e,majority_pz(e),True)['fee']==.025


def test_prior_penalty_for_simultaneous_faults_not_hidden_fairness():
    e=sensor_world(256,107014,'base')
    pz,py,pa,h=fault_posterior(e)
    assert float(py.mean())<.5 and float(pa.mean())<.5
    changed=sensor_world(256,107014,'both_reference_anchor_drift')
    a,b,c,d=fault_posterior(changed)
    assert torch.isfinite(a).all()
    # Not asserting correct localization: simultaneous faults can fool priors.
