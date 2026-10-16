"""Exact observational-equivalence certificate and independent probe tests."""
import torch
from scripts.si_original_identifiability_v16 import (
    identical_opposite_worlds, assert_twin_certificate, concatenate_twins,
    trusted_probe_baselines, neural_score, classical_scores)
from scripts.si_original_belief_filter_v12 import Controller,verify_original


def test_original_source_verified_and_same_nominal_parameter_budget():
    assert verify_original()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in Controller('hybrid',False).parameters())==64614
    assert sum(p.numel() for p in Controller('transformer',False).parameters())==64637


def test_twin_observations_are_exactly_equal_but_truth_opposite():
    a,b=identical_opposite_worlds(64,108000)
    assert_twin_certificate(a,b)
    assert torch.equal(a['anchor_cal'],b['anchor_cal'])
    assert torch.equal(a['x_query'],b['x_query'])
    assert torch.equal(a['c'],b['c'])
    assert torch.equal(a['mu_p'],b['mu_p'])


def test_every_unprobed_neural_action_same_on_opposite_twins():
    torch.set_num_threads(2)
    a,b=identical_opposite_worlds(5,108001)
    zeros=torch.zeros(5,dtype=torch.bool)
    for arch in ('hybrid','transformer','gru'):
        for feats in (False,True):
            model=Controller(arch,feats).eval()
            with torch.no_grad():
                qa=model.values(a);qb=model.values(b)
                ca=model.decision_logits(a,zeros)
                cb=model.decision_logits(b,zeros)
            assert torch.equal(qa,qb)
            assert torch.equal(ca,cb)
            assert ((ca>=0)==(a['truth']>0)).float().mean()+((cb>=0)==(b['truth']>0)).float().mean()==1


def test_classical_fault_posterior_cannot_discriminate_twins():
    torch.set_num_threads(2)
    a,b=identical_opposite_worlds(7,108002)
    from scripts.si_original_sensor_fault_v15 import fault_posterior
    with torch.no_grad():
        for p,q in zip(fault_posterior(a),fault_posterior(b)):
            assert torch.equal(p,q)


def test_trusted_probe_has_positive_information_and_nontrivial_purchase():
    a,b=identical_opposite_worlds(4096,108003)
    d=concatenate_twins(a,b)
    z=trusted_probe_baselines(d)
    assert z['never_probe']['accuracy']==.5
    assert .1<z['bayes_selective']['probe_fraction']<.9
    assert z['bayes_selective']['reward']>z['always_probe']['reward']+.025
    assert z['bayes_selective']['reward']>z['never_probe']['reward']+.025


def test_mutating_private_labels_does_not_change_no_probe_inputs():
    a,b=identical_opposite_worlds(6,108004)
    for k in ('x_cal','y_cal','anchor_cal','x_query','mu_p','c'):
        assert torch.equal(a[k],b[k])
    # Only independent trusted probe is allowed to differ among observed inputs.
    assert not torch.equal(a['y_probe'],b['y_probe'])
