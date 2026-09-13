import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase11 as p11


def test_phase10_compander_is_frozen():
    assert p10.LEVELS==512 and p10.ZMAX==16.0
    inv=p11.validate_invariants()
    assert inv['nominal_bits']==18 and inv['prefix_causal'] is True


def test_gray_generators_cover_smooth_and_sharp_without_parameter_leak():
    s,F,k=p11.gen_gray(5,6,8,(.025,.045),(.055,.065),246810,False)
    q,F2,k2=p11.gen_gray(5,6,8,(.025,.045),(.055,.065),246810,True)
    assert s.shape==q.shape==(5,6,2,8,8)
    np.testing.assert_array_equal(F,F2); np.testing.assert_array_equal(k,k2)
    assert not np.array_equal(s,q)


def test_registered_seed_namespace_unique_within_phase11():
    values=[]
    for s in p11.REGISTERED: values.extend(vars(s).values())
    assert len(values)==len(set(values))


def test_smoke_returns_all_seven_named_splits():
    seeds=p11.Seeds(731943101,731943102,731943103,731943104,731943105,731943106,731943111,731943112,731943113,731943121,731943122,731943123,731943124)
    r=p11.run_rep(seeds,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert set(r['splits'])=={'wave_id_h8','wave_spectral_ood_h3','wave_combined_ood_h3','gray_id_h8','gray_sharp_ood_h3','gray_parameter_ood_h3','gray_combined_ood_h3'}
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
