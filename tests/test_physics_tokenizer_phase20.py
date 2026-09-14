import numpy as np
from tam_research import physics_tokenizer_phase20 as p20


def test_random_basis_deterministic_orthonormal():
    a=p20.random_basis(20,21); b=p20.random_basis(20,21)
    np.testing.assert_array_equal(a,b)
    for ch in range(2): np.testing.assert_allclose(a[ch]@a[ch].T,np.eye(p20.K),atol=1e-6)


def test_feature_dim():
    z=np.zeros((2,2,8,8),np.float32); b=p20.random_basis(22,23)
    assert p20.projected_features(z,b).shape[-1]==18


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p20.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))


def test_basis_build_uses_only_seed_not_physics_data():
    names=p20.random_basis.__code__.co_names
    assert 'default_rng' in names and 'qr' in names
    assert 'gen_wave' not in names and 'gen_gray' not in names and 'gen_nls' not in names and 'gen_adv' not in names


def test_smoke_returns_all_fifteen_splits():
    s=p20.Seeds(999202001,999202002,999202003,999202004,999202005,999202006,999202007,999202008,999202011,999202012,999202013,999202021,999202022,999202023,999202024,999202031,999202032,999202033,999202034,999202041,999202042,999202043,999202044)
    r=p20.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_companded_ratios'])==15
    assert r['basis_data_independent'] is True
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
