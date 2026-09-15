import numpy as np
from tam_research import physics_tokenizer_phase27 as p27


def test_primary_library_is_phase26_interaction62():
    z=np.zeros((2,2,8,8),np.float32); b=p27.p25.p22.fourier_basis()
    assert p27.p25.interaction62_features(z,b).shape[-1]==62
    assert p27.p25.base26_features(z,b).shape[-1]==26


def test_registered_seed_namespace_unique_and_complete():
    vals=[]
    for s in p27.REGISTERED: vals.extend(vars(s).values())
    assert len(vals)==75
    assert len(vals)==len(set(vals))


def test_representation_training_excludes_three_unseen_families():
    names=p27.build_all.__code__.co_names
    assert 'gen_nls' not in names and 'gen_adv' not in names and 'gen_burgers' not in names


def test_family_name_covers_all_five():
    assert p27.family_name('wave_id_h8')=='wave'
    assert p27.family_name('gray_id_h8')=='gray_scott'
    assert p27.family_name('nls_id_h8')=='nls'
    assert p27.family_name('adv_id_h8')=='advection'
    assert p27.family_name('burgers_id_h8')=='burgers'


def test_tiny_smoke_has_nineteen_primary_splits():
    s=p27.Seeds(999272001,999272002,999272003,999272004,999272005,999272006,
        999272011,999272012,999272013,
        999272021,999272022,999272023,999272024,
        999272031,999272032,999272033,999272034,
        999272041,999272042,999272043,999272044,
        999272051,999272052,999272053,999272054)
    r=p27.run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500)
    assert len(r['primary_ratios'])==19
    assert len(r['secondary_control26_ratios'])==19
    assert sum(v['total_cells'] for v in r['per_family'].values())==19
    assert all(np.isfinite(v) for v in r['primary_ratios'].values())
