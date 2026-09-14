import numpy as np
from tam_research import physics_tokenizer_phase19 as p19


def test_basis_deterministic_and_feature_dim():
    rng=np.random.default_rng(1919)
    joint=rng.normal(size=(4,4,2,8,8)).astype(np.float32)
    a=p19.fit_basis(joint); b=p19.fit_basis(joint)
    np.testing.assert_array_equal(a['means'],b['means'])
    np.testing.assert_array_equal(a['components'],b['components'])
    assert p19.learned_features(joint[:,0],a).shape[-1]==18


def test_predictive_scores_sorted_and_finite():
    rng=np.random.default_rng(1920)
    joint=rng.normal(size=(4,4,2,8,8)).astype(np.float32)
    b=p19.fit_basis(joint)
    for s in b['predictive_scores']:
        assert len(s)==p19.K and all(np.isfinite(s))
        assert all(s[i] >= s[i+1]-1e-10 for i in range(len(s)-1))


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p19.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))


def test_unseen_families_excluded_from_basis_training():
    names=p19.build_all.__code__.co_names
    assert 'gen_wave' in names and 'gen_gray' in names
    assert 'gen_nls' not in names and 'gen_adv' not in names


def test_smoke_returns_all_fifteen_splits():
    s=p19.Seeds(999192001,999192002,999192003,999192004,999192005,999192006,999192011,999192012,999192013,999192021,999192022,999192023,999192024,999192031,999192032,999192033,999192034,999192041,999192042,999192043,999192044)
    r=p19.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_companded_ratios'])==15
    assert r['feature_dim']==18
    assert r['nls_used_in_representation_or_basis_training'] is False
    assert r['advection_used_in_representation_or_basis_training'] is False
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
