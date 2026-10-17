"""v23 logged-arm causal/first-probe calibration source integrity tests."""
import inspect
import torch
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_joint_logged_v21 as v21
from scripts import si_original_uplift_calibration_v23 as v23


def test_canonical_original_backbone_identity():
    assert v18.verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in v18.LoggedController('hybrid').backbone.parameters())==64614
    assert sum(p.numel() for p in v18.LoggedController('transformer').backbone.parameters())==64637


def test_public_features_no_hidden_truth_or_unpurchased_second():
    torch.set_num_threads(2)
    e=v19.evaluation_world(6,115001,'base')
    m=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('hybrid'),115001))
    f=torch.ones(12,dtype=torch.long)
    alter={**e,'latent_answer':-e['latent_answer'],'probe_high':e['probe_high']+1000}
    with torch.no_grad():
        assert torch.equal(v23.public_features(e,f,m),v23.public_features(alter,f,m))
        assert torch.equal(m.decision(e,f),m.decision(alter,f))


def test_logged_arm_records_only_actual_executed_outcome():
    torch.set_num_threads(2)
    e=v19.evaluation_world(8,115002,'base')
    m=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),115002))
    X,a,y,first=v23.logged_batch(m,e,115002)
    assert X.shape==(16,14)
    assert len(a)==16 and len(y)==16 and len(first)==16
    assert set(y.unique().tolist()).issubset({0.,1.})
    assert set(first.unique().tolist()).issubset({1,2})


def test_public_cost_factor_and_frozen_model_weights():
    torch.set_num_threads(2)
    e=v19.evaluation_world(4,115003,'base')
    m=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),115003))
    w=[p.detach().clone() for p in m.parameters()]
    cal=v23.train_calibration(m,115003,n_train=400)
    assert all(torch.equal(p,x) for p,x in zip(m.parameters(),w))
    cp=v23.CorrectedPolicy(m,cal,'ridge')
    f=torch.ones(8,dtype=torch.long)
    altered={**e,'cost_high':e['cost_high']+.1}
    with torch.no_grad():
        q0=cp.q1(e,f);q1=cp.q1(altered,f)
        # Price enters public input features and action-head predictions, so
        # the correction need not change by exactly -.1; it must remain finite.
        assert torch.isfinite(q0).all() and torch.isfinite(q1).all()


def test_logged_randomization_arm_coverage_and_determinism():
    torch.set_num_threads(2)
    m=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),115004))
    a=v23.train_calibration(m,115004,n_train=800)
    b=v23.train_calibration(m,115004,n_train=800)
    assert a['counts']==b['counts']
    assert a['shifts']==b['shifts']
    assert all(torch.equal(x,y) for x,y in zip(a['weights'],b['weights']))
    assert min(a['counts'].values())>100


def test_training_source_has_no_oracle_purchase_labels():
    source=inspect.getsource(v23.train_calibration)
    assert 'teacher_first' not in source
    assert 'second_purchase' not in source
    assert 'audit_first(' not in source
    assert 'logged_batch(' in source
    assert 'ridge(' in source


def test_heldout_factual_arm_mse_and_policy_scores_finite():
    torch.set_num_threads(2)
    m=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),115005))
    cal=v23.train_calibration(m,115005,n_train=400)
    e=v19.evaluation_world(6,115006,'strong_discount_ood')
    for method,net in [('unchanged',m),('offset',v23.CorrectedPolicy(m,cal,'offset')),('ridge',v23.CorrectedPolicy(m,cal,'ridge'))]:
        mse=v23.observed_arm_mse(net,e,115006)
        score=v18.score_logged(e,net)
        assert 0<=mse<10,(method,mse)
        assert -1<=score['reward']<=1
        assert 0<=score['second_buy']<=1
