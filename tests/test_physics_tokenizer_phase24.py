import numpy as np
from tam_research import physics_tokenizer_phase24 as p24


def test_subset_lattice_complete():
    assert len(p24.OMISSION_SUBSETS)==16
    assert {r:sum(len(o)==r for o in p24.OMISSION_SUBSETS) for r in range(5)}=={0:1,1:4,2:6,3:4,4:1}


def test_feature_dims_follow_rank():
    b=p24.p22.fourier_basis(); z=np.zeros((2,2,8,8),np.float32)
    for o in p24.OMISSION_SUBSETS:
        assert p24.selected_features(z,b,o).shape[-1]==8+2*(9-len(o))


def test_only_preregistered_sines_are_omitted():
    for o in p24.OMISSION_SUBSETS:
        assert set(o)<=set(p24.SINE_POOL)
    assert set(p24.SINE_POOL)=={'sin_x','sin_y','sin_diag_plus','sin_diag_minus'}


def test_registered_seed_namespace_unique():
    vals=[]
    for s in p24.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))


def test_tiny_smoke_has_all_variants_and_splits():
    s=p24.Seeds(999242001,999242002,999242003,999242004,999242005,999242006,999242011,999242012,999242013,999242021,999242022,999242023,999242024,999242031,999242032,999242033,999242034,999242041,999242042,999242043,999242044)
    r=p24.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['variants'])==16
    assert all(len(v['primary_companded_ratios'])==15 for v in r['variants'].values())
    assert all(np.isfinite(x) for v in r['variants'].values() for x in v['primary_companded_ratios'].values())
