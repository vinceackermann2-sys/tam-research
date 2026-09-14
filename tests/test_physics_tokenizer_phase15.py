import numpy as np
from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase13 as p13
from tam_research import physics_tokenizer_phase15 as p15

def test_frozen_recipe_and_directional_extension_dims():
    z=np.zeros((1,2,8,8),np.float32)
    assert p10.LEVELS==512 and p10.ZMAX==16.0
    assert p13.CONTEXT==8 and p13.RIDGE==.001
    assert p10.features(z).shape[-1]==16
    assert p15.features_directional(z).shape[-1]==20

def test_advection_generator_deterministic_and_finite():
    a,s,ang=p15.gen_adv(3,6,8,(.18,.30),(-.20,.20),424215,3)
    b,s2,ang2=p15.gen_adv(3,6,8,(.18,.30),(-.20,.20),424215,3)
    np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(s,s2); np.testing.assert_array_equal(ang,ang2)
    assert np.isfinite(a).all()

def test_directional_features_close_representability_gap_on_probe():
    x,_,_=p15.gen_adv(4,8,8,(.30,.35),(.70,.80),424216,4)
    r16=p15.context_one_step_ratio(x,p10.features,16)
    r20=p15.context_one_step_ratio(x,p15.features_directional,20)
    assert r20 < r16*1e-2

def test_registered_seed_namespace_unique():
    vals=[]
    for s in p15.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))

def test_smoke_returns_four_splits_and_excludes_advection_training():
    s=p15.Seeds(995153001,995153002,995153003,995153004,995153005,995153006,995153011,995153012,995153013,995153014)
    r=p15.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert set(r['primary_companded_ratios'])=={'adv_id_h8','adv_spectral_ood_h3','adv_speed_ood_h3','adv_combined_ood_h3'}
    assert r['advection_used_in_representation_training'] is False
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
