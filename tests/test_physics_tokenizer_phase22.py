import numpy as np
from tam_research import physics_tokenizer_phase22 as p22


def test_fourier_basis_is_orthonormal_and_complete():
    b = p22.fourier_basis()
    assert b.shape == (9, 9)
    np.testing.assert_allclose(b @ b.T, np.eye(9), atol=1e-6)


def test_full_basis_reconstructs_patch_and_lodo_dims():
    rng = np.random.default_rng(2201)
    z = rng.normal(size=(3, 2, 8, 8)).astype(np.float32)
    b = p22.fourier_basis()
    for ch in range(2):
        P = p22.patch9(z, ch)
        Y = np.einsum('bhwp,kp->bhwk', P, b, optimize=True)
        rec = np.einsum('bhwk,kp->bhwp', Y, b, optimize=True)
        np.testing.assert_allclose(P, rec, rtol=1e-5, atol=1e-5)
    assert p22.selected_features(z, b, None).shape[-1] == 26
    for i in range(9):
        assert p22.selected_features(z, b, i).shape[-1] == 24


def test_mode_names_and_registered_seeds_unique():
    assert len(p22.MODE_NAMES) == 9
    assert len(set(p22.MODE_NAMES)) == 9
    vals = []
    for s in p22.REGISTERED:
        vals.extend(vars(s).values())
    assert len(vals) == len(set(vals))


def test_representation_training_excludes_unseen_families():
    names = p22.build_all.__code__.co_names
    assert 'build_codecs' in names
    assert 'gen_nls' not in names and 'gen_adv' not in names


def test_tiny_smoke_has_control_and_all_nine_lodo_variants():
    s = p22.Seeds(
        999222001,999222002,999222003,999222004,999222005,999222006,
        999222011,999222012,999222013,
        999222021,999222022,999222023,999222024,
        999222031,999222032,999222033,999222034,
        999222041,999222042,999222043,999222044)
    r = p22.run_rep(s, grid=8, train_n=10, eval_n=4, code_sizes=(16,4), iters=2, max_points=1500)
    assert len(r['full_control']['primary_companded_ratios']) == 15
    assert set(r['leave_one_out']) == set(p22.MODE_NAMES)
    assert all(len(v['primary_companded_ratios']) == 15 for v in r['leave_one_out'].values())
    assert all(np.isfinite(x) for x in r['full_control']['primary_companded_ratios'].values())
