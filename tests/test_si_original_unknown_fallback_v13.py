"""Causal observability, exact source and conservative fallback integrity checks."""
import torch
from scripts.si_original_unknown_fallback_v13 import (
    verify_original, evidence_by_pattern, unknown_model_belief, make_world,
    evaluate_pair, bayes_references, Controller, KNOWN_PATTERNS,
    UNKNOWN_PATTERNS, model_actions,
)


def test_canonical_source_and_nominal_parameter_parity():
    assert verify_original() == '575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert len(KNOWN_PATTERNS) == 4
    assert len(UNKNOWN_PATTERNS) == 28
    for k in (False,True):
        h=Controller('hybrid',k); t=Controller('transformer',k)
        assert sum(p.numel() for p in h.parameters()) == 64614
        assert sum(p.numel() for p in t.parameters()) == 64637


def test_only_observed_calibration_affects_unknown_mass_and_actions():
    torch.set_num_threads(2)
    e=make_world(6,105001,'double_flip')
    manipulated={**e, 'z':-e['z'], 'truth':-e['truth'],
        'y_probe':e['y_probe']+1000,'private_flip_at':torch.arange(6)+10}
    for x,y in zip(unknown_model_belief(e),unknown_model_belief(manipulated)):
        assert torch.equal(x,y)
    for kind in ('hybrid','transformer','gru'):
        for has_belief in (False,True):
            policy=Controller(kind,has_belief).eval()
            with torch.no_grad():
                assert torch.allclose(policy.values(e),policy.values(manipulated),atol=1e-6)
                no_probe=torch.zeros(6,dtype=torch.bool)
                assert torch.allclose(policy.decision_logits(e,no_probe),
                    policy.decision_logits(manipulated,no_probe),atol=1e-6)


def test_mass_is_normalized_and_uniform_patterns_have_valid_likelihood():
    torch.set_num_threads(2)
    for f in ('base','one_flip','early_flip','double_flip','weak_calibration','reference_drift'):
        e=make_world(96,105002,f)
        loglike=evidence_by_pattern(e)
        unknown,pz=unknown_model_belief(e)
        assert loglike.shape==(96,2,32)
        assert torch.isfinite(loglike).all()
        assert torch.isfinite(unknown).all() and torch.isfinite(pz).all()
        assert ((unknown>=0)&(unknown<=1)).all()
        assert ((pz>=0)&(pz<=1)).all()


def test_unknown_detects_more_two_changes_than_stationary_in_large_development_panel():
    a=make_world(1024,105003,'base')
    b=make_world(1024,105004,'double_flip')
    ua,_=unknown_model_belief(a); ub,_=unknown_model_belief(b)
    assert float((ub>.5).float().mean()) > float((ua>.5).float().mean())+.15


def test_safe_fallback_extremes_match_exact_single_policies():
    torch.set_num_threads(2)
    e=make_world(8,105005,'one_flip')
    raw=Controller('hybrid',False).eval()
    bel=Controller('hybrid',True).eval()
    with torch.no_grad():
        result=evaluate_pair(raw,bel,e,105005,threshold=.5)
        assert 0<=result['fallback']['fallback_fraction']<=1
        assert result['raw']['fallback_fraction']==1.
        assert result['belief']['fallback_fraction']==0.
        assert -1 <=result['fallback']['reward']<=1
        assert torch.isfinite(torch.tensor(result['fallback']['logged_arm_brier']))


def test_robust_baselines_finite_even_under_unknown_reference_drift():
    e=make_world(128,105006,'reference_drift')
    out=bayes_references(e)
    assert all(-1<=v['reward']<=1 for v in out.values())
    assert set(out)=={'expanded_pattern_bayes','stationary_bayes','stationary_always','stationary_never'}
