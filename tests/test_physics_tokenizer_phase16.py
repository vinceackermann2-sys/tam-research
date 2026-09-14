import numpy as np
from tam_research import physics_tokenizer_phase16 as p16


def test_feature_dim_and_parity():
    inv=p16.validate_invariants(); assert inv['feature_dim']==18 and inv['parity_checks']


def test_library_contains_odd_and_even_operators():
    rng=np.random.default_rng(3); z=rng.normal(size=(1,2,8,8)).astype(np.float32)
    f=p16.generic_features(z); assert f.shape==(1,8,8,18)
    assert np.std(f[...,2])>0 and np.std(f[...,4])>0


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p16.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==63 and len(vals)==len(set(vals))


def test_generators_finite():
    a,_,_=p16.gen_adv(2,5,8,(.18,.3),(-.2,.2),991,3)
    n,_,_=p16.gen_nls(2,5,8,(.04,.08),(.1,.25),992,3)
    g,_,_=p16.gen_gray(2,5,8,(.025,.045),(.055,.065),993,True)
    assert np.isfinite(a).all() and np.isfinite(n).all() and np.isfinite(g).all()


def test_tiny_smoke_has_fifteen_splits():
    s=p16.Seeds(996161001,996161002,996161003,996161004,996161005,996161006,996161011,996161012,996161013,996161021,996161022,996161023,996161024,996161031,996161032,996161033,996161034,996161041,996161042,996161043,996161044)
    r=p16.run_rep(s,grid=8,train_n=10,eval_n=3,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_companded_ratios'])==15
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
