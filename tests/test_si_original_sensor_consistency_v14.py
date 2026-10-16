"""Causal and paired control tests for original SI v14 sensor redundancy."""
import torch
from scripts.si_original_sensor_consistency_v14 import (
    verify_original, world_with_anchor, consistency, repaired_observation,
    score_policy, evaluate_cached, detector_metrics, ANCHOR_FEE,
)
from scripts.si_original_belief_filter_v12 import Controller


def test_unmodified_original_and_matched_parameter_counts():
    assert verify_original() == '575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    for b in (True,False):
        assert sum(p.numel() for p in Controller('hybrid',b).parameters())==64614
        assert sum(p.numel() for p in Controller('transformer',b).parameters())==64637


def test_identical_physical_world_is_reproducible():
    for fam in ('base','double_flip','reference_drift','anchor_drift','reference_dropout'):
        a=world_with_anchor(7,106000,fam)
        b=world_with_anchor(7,106000,fam)
        assert all(torch.equal(a[k],b[k]) for k in a)
        assert a['anchor_cal'].shape==(7,6)


def test_detector_cannot_read_latent_future_or_unpurchased_query():
    e=world_with_anchor(9,106001,'reference_drift')
    altered={**e,'z':-e['z'],'truth':-e['truth'],'cal_truth':-e['cal_truth'],
             'y_probe':e['y_probe']+999,'private_reference_drift':torch.zeros(9,dtype=torch.bool)}
    assert torch.equal(consistency(e),consistency(altered))
    assert detector_metrics(e)==detector_metrics(altered)
    assert torch.equal(repaired_observation(e)['y_cal'],repaired_observation(altered)['y_cal'])


def test_changed_anchor_can_be_mistaken_for_reference_failure():
    normal=world_with_anchor(1024,106002,'base')
    bad_anchor=world_with_anchor(1024,106002,'anchor_drift')
    # Same base x/y observations, only the independently measured anchor changed.
    assert torch.equal(normal['x_cal'],bad_anchor['x_cal'])
    assert torch.equal(normal['y_cal'],bad_anchor['y_cal'])
    assert float((consistency(bad_anchor)<.5).float().mean()) > .75


def test_cached_evaluation_exactly_matches_uncached_mode_and_fee():
    torch.set_num_threads(2)
    e=world_with_anchor(9,106003,'double_flip')
    torch.manual_seed(17)
    raw=Controller('hybrid',False).eval()
    bel=Controller('hybrid',True).eval()
    modes=('raw','belief','v13_fallback','always_anchor_belief',
           'consistency_repair','shuffled_repair','repair_plus_v13_fallback')
    with torch.no_grad():
        fast=evaluate_cached(raw,bel,e,106003)
        for mode in modes:
            slow=score_policy(raw,bel,e,mode,106003)
            assert fast[mode]==slow, (mode,fast[mode],slow)
    assert fast['raw']['anchor_fee']==0.
    assert fast['belief']['anchor_fee']==0.
    assert fast['consistency_repair']['anchor_fee']==ANCHOR_FEE
    assert fast['always_anchor_belief']['anchor_fee']==ANCHOR_FEE


def test_neural_decisions_do_not_receive_new_anchor_directly():
    torch.set_num_threads(2)
    e=world_with_anchor(4,106004,'base')
    mutated={**e,'anchor_cal':e['anchor_cal']+100}
    with torch.no_grad():
        for a in ('hybrid','transformer','gru'):
            for beliefs in (False,True):
                model=Controller(a,beliefs).eval()
                assert torch.allclose(model.values(e),model.values(mutated),atol=1e-7)
                zeros=torch.zeros(4,dtype=torch.bool)
                assert torch.allclose(model.decision_logits(e,zeros),model.decision_logits(mutated,zeros),atol=1e-7)


def test_observation_substitution_uses_measured_sensor_not_private_truth():
    e=world_with_anchor(12,106005,'reference_dropout')
    transformed=repaired_observation(e)
    assert torch.equal(transformed['x_cal'],e['x_cal'])
    assert torch.equal(transformed['y_cal'],e['anchor_cal']*(e['mu_ref']/e['anchor_mu'])[:,None])
    assert not torch.equal(transformed['y_cal'],e['cal_truth'])
