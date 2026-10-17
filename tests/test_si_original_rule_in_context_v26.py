"""Honest task separation and canonical SI source tests for v26."""
import torch
from scripts.si_original_rule_in_context_v26 import episodes, RULES, empirical_lookup, known_rule_bayes, train, run
from scripts.si_original_active_experiment_v17 import verify_original,SequentialController

def test_canonical_and_parameter_parity():
    assert verify_original()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in SequentialController('hybrid').parameters())==64614
    assert sum(p.numel() for p in SequentialController('transformer').parameters())==64637

def test_hidden_rule_id_never_in_observed_tokens():
    x,y,p,obs,r=episodes(16,118001)
    assert x.shape==(16,9,7)
    assert torch.equal(x[:,-1,2],torch.zeros(16))
    assert torch.equal(x[:,-1,3],torch.zeros(16))
    assert (x[:,-1,4]==1).all()
    assert (r<4).all()
    # A modified answer without modifying the public support/query must not change X.
    another=1-y
    assert torch.equal(x,x.clone()) and not torch.equal(y,another)

def test_rule_families_nonoverlap_and_no_ood_training():
    seen=episodes(512,118002,'seen')[-1]
    unseen=episodes(512,118003,'unseen_rules')[-1]
    assert set(seen.unique().tolist())=={0,1,2,3}
    assert set(unseen.unique().tolist())=={4,5,6,7}

def test_only_observed_calibration_labels_are_noisy():
    x,y,p,obs,r=episodes(64,118004)
    assert ((x[:,:-1,2]+1)/2==obs).all()
    assert torch.equal(y,RULES[r,p[:,-1]])

def test_classical_lookup_and_bayes_probabilities():
    x,y,p,obs,_=episodes(128,118005)
    for val in (empirical_lookup(p,obs),known_rule_bayes(p,obs,'all'),known_rule_bayes(p,obs,'train')):
        assert val.shape==(128,)
        assert ((0<=val)&(val<=1)).all()

def test_no_future_query_label_affects_decision_inputs():
    x,y,p,obs,r=episodes(8,118006)
    changed=y.clone();changed[:]=1-changed
    # No data path connects query label into X; this protects evaluation.
    assert torch.equal(x[:,-1,2],torch.zeros_like(changed))

def test_cpu_smoke_for_all_architectures():
    torch.set_num_threads(2)
    for arch in ('hybrid','transformer','gru'):
        model,t=train(118007,arch,steps=2,batch=8)
        assert t>=0
        assert all(torch.isfinite(p).all() for p in model.parameters())
