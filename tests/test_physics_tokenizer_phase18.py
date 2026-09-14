import numpy as np
from tam_research import physics_tokenizer_phase18 as p18


def test_basis_deterministic_and_feature_dim():
    rng=np.random.default_rng(1818)
    joint=rng.normal(size=(4,3,2,8,8)).astype(np.float32)
    a=p18.fit_basis(joint); b=p18.fit_basis(joint)
    np.testing.assert_array_equal(a['means'],b['means'])
    np.testing.assert_array_equal(a['components'],b['components'])
    assert p18.learned_features(joint[:,0],a).shape[-1]==18


def test_basis_is_compressed():
    assert p18.K==5
    assert 2*p18.K < 18


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p18.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))


def test_unseen_families_excluded_from_basis_training():
    names=p18.build_all.__code__.co_names
    assert 'gen_wave' in names and 'gen_gray' in names
    assert 'gen_nls' not in names and 'gen_adv' not in names


def test_smoke_returns_all_fifteen_splits():
    s=p18.Seeds(998182001,998182002,998182003,998182004,998182005,998182006,998182011,998182012,998182013,998182021,998182022,998182023,998182024,998182031,998182032,998182033,998182034,998182041,998182042,998182043,998182044)
    r=p18.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_companded_ratios'])==15
    assert r['feature_dim']==18
    assert r['nls_used_in_representation_or_basis_training'] is False
    assert r['advection_used_in_representation_or_basis_training'] is False
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
