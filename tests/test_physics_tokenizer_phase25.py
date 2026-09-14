import numpy as np
from tam_research import physics_tokenizer_phase25 as p25


def test_burgers_generator_is_finite_and_shaped():
    x,nu,amp=p25.gen_burgers(4,7,8,(.01,.03),(.15,.35),2501,3)
    assert x.shape==(4,7,2,8,8)
    assert np.isfinite(x).all()
    assert np.all((nu>=.01)&(nu<=.03))


def test_feature_dimensions():
    z=np.zeros((2,2,8,8),np.float32);b=p25.p22.fourier_basis()
    assert p25.base26_features(z,b).shape[-1]==26
    assert p25.interaction62_features(z,b).shape[-1]==62


def test_interaction_library_contains_center_times_all_spatial_coordinates():
    rng=np.random.default_rng(25);z=rng.normal(size=(2,2,8,8)).astype(np.float32);b=p25.p22.fourier_basis()
    base=p25.base26_features(z,b);f=p25.interaction62_features(z,b);sp=base[...,1:19]
    np.testing.assert_allclose(f[...,26:44],z[:,0][...,None]*sp)
    np.testing.assert_allclose(f[...,44:62],z[:,1][...,None]*sp)


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p25.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))


def test_tiny_smoke_has_four_burgers_splits():
    s=p25.Seeds(999252001,999252002,999252003,999252004,999252005,999252006,999252011,999252012,999252013,999252014)
    r=p25.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_ratios'])==4
    assert all(np.isfinite(v) for v in r['primary_ratios'].values())
    assert all('raw62_one_step_ratio' in x for x in r['splits'].values())


def test_raw_interaction_library_spans_burgers_one_step_on_throwaway_seed():
    raw,_,_=p25.gen_burgers(4,10,8,(.01,.03),(.15,.35),999253001,3)
    b=p25.p22.fourier_basis()
    maps=p25.fit_maps(raw[:,:p25.CONTEXT],p25.interaction62_features,62,b)
    pred=p25.predict(raw[:,p25.CONTEXT-1],maps,p25.interaction62_features,b)
    target=raw[:,p25.CONTEXT]
    persist=raw[:,p25.CONTEXT-1]
    ratio=p25.p10.mse(pred,target)/p25.p10.mse(persist,target)
    assert ratio < 1e-2
