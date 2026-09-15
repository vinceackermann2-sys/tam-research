import numpy as np
from tam_research import physics_tokenizer_phase30 as p30

def test_seed_namespace_unique_and_complete():
    vals=[]
    for s in p30.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==87 and len(vals)==len(set(vals))

def test_shared_support_configuration_frozen():
    assert p30.MEDIAN_GAIN_MIN==0.10
    assert p30.CLEAN_PREFIX_TRANSITIONS==(1,2,3,4)
    assert p30.HELDOUT_TRANSITIONS==(5,6)
    assert p30.CLEAN_ALL_TRANSITIONS==(1,2,3,4,5,6)

def test_82_library_shape():
    z=np.zeros((2,2,8,8),np.float32); b=p30.p25.p22.fourier_basis()
    assert p30.p29.odd_spatial_quadratic82(z,b).shape[-1]==82

def test_parent_phase28_configuration_unchanged():
    assert p30.p28.FIT_FRAMES==6
    assert p30.p28.VALIDATION_PAIRS==((5,6),(6,7))
    assert p30.RIDGE==p30.p28.RIDGE==0.001

def test_tiny_smoke_has_23_splits_and_is_finite():
    r=p30.run_rep(p30._seedset(999301000),grid=8,train_n=8,eval_n=3,code_sizes=(16,4),iters=2,max_points=1200)
    assert len(r['primary_ratios'])==23
    assert sum(v['total_cells'] for v in r['per_family'].values())==23
    assert all(np.isfinite(v) for v in r['primary_ratios'].values())
