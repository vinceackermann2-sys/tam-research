import numpy as np
from tam_research import physics_tokenizer_phase29 as p29


def test_diagnostic_library_has_82_features():
    z=np.zeros((2,2,8,8),np.float32); b=p29.p22.fourier_basis()
    assert p29.odd_spatial_quadratic82(z,b).shape[-1]==82


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p29.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==30
    assert len(vals)==len(set(vals))


def test_hamilton_jacobi_generator_is_finite_and_shaped():
    x,nu,lam,amp=p29.gen_hj(4,7,8,(.015,.035),(.20,.40),(.25,.50),999293001,3)
    assert x.shape==(4,7,2,8,8)
    assert np.isfinite(x).all()
    assert nu.shape==(4,2) and lam.shape==(4,2) and amp.shape==(4,2)


def test_phase28_selector_configuration_is_unchanged():
    assert p29.p28.FIT_FRAMES==6
    assert p29.p28.VALIDATION_PAIRS==((5,6),(6,7))
    assert p29.RIDGE==p29.p28.RIDGE==0.001


def test_raw82_spans_one_step_hj_on_throwaway_seed():
    raw,*_=p29.gen_hj(4,10,8,(.015,.035),(.20,.40),(.25,.50),999293101,3)
    b=p29.p22.fourier_basis()
    m=p29.p25.fit_maps(raw[:,:p29.CONTEXT],p29.odd_spatial_quadratic82,82,b,p29.RIDGE)
    pred=p29.p25.predict(raw[:,p29.CONTEXT-1],m,p29.odd_spatial_quadratic82,b)
    target=raw[:,p29.CONTEXT]
    persist=raw[:,p29.CONTEXT-1]
    ratio=p29.p10.mse(pred,target)/p29.p10.mse(persist,target)
    assert ratio < 1e-2
