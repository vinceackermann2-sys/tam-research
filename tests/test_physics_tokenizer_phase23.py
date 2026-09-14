import numpy as np
from tam_research import physics_tokenizer_phase23 as p23


def test_feature_dimensions():
    z=np.zeros((2,2,8,8),np.float32)
    assert p23.geometry_features(z,4).shape[-1]==16
    assert p23.geometry_features(z,5).shape[-1]==18


def test_geometry_semantics():
    rng=np.random.default_rng(23); z=rng.normal(size=(2,2,8,8)).astype(np.float32)
    f=p23.geometry_features(z,4)
    for ch in range(2):
        off=1+ch*4; x=z[:,ch]
        axial=np.roll(x,1,-2)+np.roll(x,-1,-2)+np.roll(x,-1,-1)+np.roll(x,1,-1)
        np.testing.assert_allclose(f[...,off+1],axial)
        np.testing.assert_allclose(f[...,off+2],np.roll(x,-1,-1)-np.roll(x,1,-1))
        np.testing.assert_allclose(f[...,off+3],np.roll(x,-1,-2)-np.roll(x,1,-2))


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p23.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))


def test_unseen_families_excluded_from_representation_training():
    names=p23.build_all.__code__.co_names
    assert 'build_codecs' in names
    assert 'gen_nls' not in names and 'gen_adv' not in names


def test_tiny_smoke_returns_paired_fifteen_splits():
    s=p23.Seeds(999232001,999232002,999232003,999232004,999232005,999232006,999232011,999232012,999232013,999232021,999232022,999232023,999232024,999232031,999232032,999232033,999232034,999232041,999232042,999232043,999232044)
    r=p23.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['rank4']['primary_companded_ratios'])==15
    assert len(r['rank5']['primary_companded_ratios'])==15
    assert all(np.isfinite(v) for v in r['rank4']['primary_companded_ratios'].values())
    assert all(np.isfinite(v) for v in r['rank5']['primary_companded_ratios'].values())
