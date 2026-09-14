import numpy as np
from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase12 as p12

def test_frozen_phase10_codec_and_identifier():
    assert p10.LEVELS == 512 and p10.ZMAX == 16.0
    assert p10.features(np.zeros((1,2,4,4),np.float32)).shape[-1] == 16

def test_nls_is_deterministic_and_finite():
    a,al,g=p12.gen_nls(3,6,8,(.04,.08),(.10,.25),4242,3)
    b,al2,g2=p12.gen_nls(3,6,8,(.04,.08),(.10,.25),4242,3)
    np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(al,al2); np.testing.assert_array_equal(g,g2)
    assert np.isfinite(a).all()

def test_nls_exact_step_is_in_frozen_feature_span():
    assert p12.representability_error() < 1e-12

def test_registered_seed_namespace_unique_within_phase12():
    vals=[]
    for s in p12.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==len(set(vals))

def test_smoke_has_four_frozen_nls_splits():
    s=p12.Seeds(991240101,991240102,991240103,991240104,991240105,991240106,991240111,991240112,991240113,991240114)
    r=p12.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert set(r['splits']) == {'nls_id_h8','nls_spectral_ood_h3','nls_parameter_ood_h3','nls_combined_ood_h3'}
    assert r['nls_used_in_representation_training'] is False
    assert all(np.isfinite(v) for v in r['primary_companded_ratios'].values())
