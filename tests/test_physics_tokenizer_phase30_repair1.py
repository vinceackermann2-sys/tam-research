import pickle, tempfile, numpy as np
from pathlib import Path
from tam_research import physics_tokenizer_phase30_repair1 as r

def test_fresh_seed_namespace_unique():
    vals=[]
    for rep in (1,2,3): vals.extend(vars(r.seeds_for_rep(rep)).values())
    assert len(vals)==87 and len(vals)==len(set(vals))

def test_scientific_config_is_phase30_exact():
    assert r.p30.MEDIAN_GAIN_MIN==0.10
    assert r.p30.CLEAN_PREFIX_TRANSITIONS==(1,2,3,4)
    assert r.p30.HELDOUT_TRANSITIONS==(5,6)
    assert r.p30.CLEAN_ALL_TRANSITIONS==(1,2,3,4,5,6)

def test_family_partition_covers_23_splits():
    s=r.p30._seedset(999302500)
    raw=r.p30.build_eval(s,8,2)
    counts={f:sum(r.split_family(k)==f for k in raw) for f in r.FAMILIES}
    assert counts=={'wave':3,'gray_scott':4,'nls':4,'advection':4,'burgers':4,'hamilton_jacobi':4}

def test_smoke_prepare_and_family_eval(tmp_path):
    prep=tmp_path/'p.pkl'; out=tmp_path/'o.json'
    s=r.p30._seedset(999302700)
    state,dm,ds,basis=r.p30.build_all(s,8,6,(16,4),2,1000); raw=r.p30.build_eval(s,8,2)
    prep.write_bytes(pickle.dumps({'replicate':0,'seed_base':999302700,'seeds':vars(s),'state':state,'dm':dm,'ds':ds,'basis':basis,'raw':raw,'scientific_config':'smoke'}))
    z=r.eval_family(str(prep),'wave',str(out))
    assert z['total_cells']==3 and all(np.isfinite(v) for v in z['primary_ratios'].values())
