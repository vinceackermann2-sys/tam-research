import numpy as np
from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase12 as p12
from tam_research import physics_tokenizer_phase13 as p13

def test_frozen_representation_and_features():
    assert p10.LEVELS==512 and p10.ZMAX==16.0
    assert p10.features(np.zeros((1,2,4,4),np.float32)).shape[-1]==16

def test_context_is_exactly_eight_and_uses_seven_transitions():
    assert p13.CONTEXT==8
    raw,_,_=p12.gen_nls(2,11,8,(.04,.08),(.10,.25),1234567,3)
    m=p13.fit_maps_context8(raw[:,:8])
    assert m.shape==(2,2,16)

def test_phase13_registered_seed_namespace_unique():
    vals=[]
    for s in p13.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))

def test_nls_not_used_in_representation_fit_path():
    # fit_frozen_representation is inherited unchanged from Phase 12 and accepts only wave/gray train seeds.
    assert p13.to_phase12_seeds(p13.REGISTERED[0]).wave==641390001

def test_smoke_returns_all_four_splits():
    s=p13.Seeds(939132001,939132002,939132003,939132004,939132005,939132006,939132011,939132012,939132013,939132014)
    r=p13.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert set(r['primary_companded_ratios'])=={'nls_id_h8','nls_spectral_ood_h3','nls_parameter_ood_h3','nls_combined_ood_h3'}
    assert r['context_frames']==8 and r['observed_transitions_used']==7
    assert r['nls_used_in_representation_training'] is False
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
