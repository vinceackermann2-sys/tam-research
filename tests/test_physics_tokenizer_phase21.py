import numpy as np
from tam_research import physics_tokenizer_phase21 as p21


def test_full_basis_deterministic_and_orthonormal():
    a=p21.full_random_basis(21001,21002); b=p21.full_random_basis(21001,21002)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(2,9,9)
    for ch in range(2): np.testing.assert_allclose(a[ch]@a[ch].T,np.eye(9),atol=1e-6)


def test_nested_feature_dimensions():
    b=p21.full_random_basis(21003,21004); z=np.zeros((2,2,8,8),np.float32)
    assert [p21.projected_features(z,b,k).shape[-1] for k in p21.RANKS]==[18,20,22,24,26]


def test_rank9_spans_raw_patch():
    rng=np.random.default_rng(21005); z=rng.normal(size=(3,2,8,8)).astype(np.float32); b=p21.full_random_basis(21006,21007)
    for ch in range(2):
        P=p21.patch9(z,ch)
        Y=np.einsum('bhwp,kp->bhwk',P,b[ch],optimize=True)
        rec=np.einsum('bhwk,kp->bhwp',Y,b[ch],optimize=True)
        np.testing.assert_allclose(P,rec,rtol=1e-5,atol=1e-5)


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p21.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))


def test_smoke_returns_five_ranks_and_fifteen_splits_each():
    s=p21.Seeds(999212001,999212002,999212003,999212004,999212005,999212006,999212007,999212008,999212011,999212012,999212013,999212021,999212022,999212023,999212024,999212031,999212032,999212033,999212034,999212041,999212042,999212043,999212044)
    r=p21.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert list(r['ranks'].keys())==['5','6','7','8','9']
    for k,v in r['ranks'].items():
        assert len(v['primary_companded_ratios'])==15
        assert v['feature_dim']==8+2*int(k)
        assert all(np.isfinite(x) for x in v['primary_companded_ratios'].values())
