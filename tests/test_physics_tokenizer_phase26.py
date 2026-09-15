import numpy as np
from tam_research import physics_tokenizer_phase26 as p26

def test_primary_library_is_exact_phase25_interaction62():
    z=np.zeros((2,2,8,8),np.float32); b=p26.p25.p22.fourier_basis()
    assert p26.p25.interaction62_features(z,b).shape[-1]==62

def test_registered_seed_namespace_unique():
    vals=[]
    for s in p26.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))

def test_phase25_evaluation_path_is_reused():
    assert p26.run_rep.__code__.co_names.count('eval_split') >= 1

def test_tiny_smoke_returns_four_primary_and_control_splits():
    s=p26.Seeds(999262001,999262002,999262003,999262004,999262005,999262006,999262011,999262012,999262013,999262014)
    r=p26.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_token62_ratios'])==4
    assert len(r['secondary_token26_ratios'])==4
    assert all(np.isfinite(v) for v in r['primary_token62_ratios'].values())
