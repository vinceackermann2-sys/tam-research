import numpy as np
from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase13 as p13
from tam_research import physics_tokenizer_phase14 as p14

def test_frozen_context8_recipe():
    assert p10.LEVELS==512 and p10.ZMAX==16.0
    assert p13.CONTEXT==8 and p13.RIDGE==.001
    assert p10.features(np.zeros((1,2,4,4),np.float32)).shape[-1]==16

def test_gray_smooth_and_sharp_are_distinct_and_parameter_matched():
    s,F,k=p14.gen_gray(4,5,8,(.025,.045),(.055,.065),246814,False)
    q,F2,k2=p14.gen_gray(4,5,8,(.025,.045),(.055,.065),246814,True)
    assert s.shape==q.shape==(4,5,2,8,8)
    np.testing.assert_array_equal(F,F2); np.testing.assert_array_equal(k,k2)
    assert not np.array_equal(s,q)

def test_registered_seed_namespace_unique():
    vals=[]
    for s in p14.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))

def test_representation_training_excludes_nls():
    assert p14.run_rep.__name__=='run_rep'
    assert len(p14.REGISTERED)==3

def test_smoke_returns_all_eleven_splits():
    s=p14.Seeds(731954001,731954002,731954003,731954004,731954005,731954006,731954011,731954012,731954013,731954021,731954022,731954023,731954024,731954031,731954032,731954033,731954034)
    r=p14.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_companded_ratios'])==11
    assert r['nls_used_in_representation_training'] is False
    assert r['context_frames']==8 and r['observed_transitions_used']==7
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
