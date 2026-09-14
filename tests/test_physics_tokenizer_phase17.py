import numpy as np
from tam_research import physics_tokenizer_phase17 as p17


def test_feature_dim_and_operator_span():
    z=np.random.default_rng(4).normal(size=(2,2,8,8)).astype(np.float32)
    f=p17.stencil_features(z)
    assert f.shape[-1]==26
    for ch in range(2):
        off=1+ch*9; zz=z[:,ch]
        left,right,up,down,center=f[...,off+3],f[...,off+5],f[...,off+1],f[...,off+7],f[...,off+4]
        np.testing.assert_allclose(.5*(right-left),p17.p16.centered_dx(zz),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(.5*(down-up),p17.p16.centered_dy(zz),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(left+right+up+down-4*center,p17.p10.lap(zz),rtol=1e-6,atol=1e-6)


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p17.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))


def test_unseen_families_excluded_from_representation_training():
    names=p17.p16.build_codecs.__code__.co_names
    assert 'gen_wave' in names and 'gen_gray' in names
    assert 'gen_nls' not in names and 'gen_adv' not in names


def test_generators_deterministic():
    a=p17.p16.gen_adv(3,5,8,(.18,.30),(-.2,.2),917170001,3)[0]
    b=p17.p16.gen_adv(3,5,8,(.18,.30),(-.2,.2),917170001,3)[0]
    np.testing.assert_array_equal(a,b)
    n1=p17.p16.gen_nls(3,5,8,(.04,.08),(.10,.25),917170002,3)[0]
    n2=p17.p16.gen_nls(3,5,8,(.04,.08),(.10,.25),917170002,3)[0]
    np.testing.assert_array_equal(n1,n2)


def test_smoke_returns_all_fifteen_splits():
    s=p17.Seeds(997175001,997175002,997175003,997175004,997175005,997175006,997175011,997175012,997175013,997175021,997175022,997175023,997175024,997175031,997175032,997175033,997175034,997175041,997175042,997175043,997175044)
    r=p17.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_companded_ratios'])==15
    assert r['feature_dim']==26 and r['context_frames']==8
    assert r['nls_used_in_representation_training'] is False
    assert r['advection_used_in_representation_training'] is False
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
