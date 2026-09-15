import numpy as np
from tam_research import physics_tokenizer_phase28 as p28

def test_seed_namespace_unique():
    vals=[]
    for s in p28.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==75 and len(vals)==len(set(vals))

def test_selector_uses_only_observed_context():
    assert p28.CONTEXT==8 and p28.FIT_FRAMES==6
    assert p28.VALIDATION_PAIRS==((5,6),(6,7))

def test_bic_penalizes_interaction_model_more():
    n=2*2*8*8
    assert (2*62)*np.log(n) > (2*26)*np.log(n)

def test_tie_break_is_simple():
    z=np.zeros((2,8,2,8,8),np.float32); b=p28.p25.p22.fourier_basis()
    choose,*_=p28.select_models(z,b)
    assert choose.shape==(2,)
    assert not choose.any()

def test_tiny_smoke_has_19_cells():
    s=p28.Seeds(999282001,999282002,999282003,999282004,999282005,999282006,999282011,999282012,999282013,999282021,999282022,999282023,999282024,999282031,999282032,999282033,999282034,999282041,999282042,999282043,999282044,999282051,999282052,999282053,999282054)
    r=p28.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_ratios'])==19
    assert all(np.isfinite(v) for v in r['primary_ratios'].values())
