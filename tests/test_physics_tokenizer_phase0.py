import numpy as np
import torch

from tam_research.physics_tokenizer_phase0 import (
    AutoEncoder,
    TokenDynamics,
    fit_kmeans,
    generate_sequences,
    nearest_codes,
)


def test_wave_generator_is_deterministic_and_finite():
    a, ca = generate_sequences(3, 5, 8, (0.6, 1.0), 0.12, 123)
    b, cb = generate_sequences(3, 5, 8, (0.6, 1.0), 0.12, 123)
    assert a.shape == (3, 5, 2, 8, 8)
    assert ca.shape == (3,)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ca, cb)


def test_core_tokenizer_shapes_and_code_bounds():
    torch.manual_seed(5)
    model = AutoEncoder(latent_dim=6)
    x = torch.randn(2, 2, 8, 8)
    z = model.encode(x)
    assert z.shape == (2, 6, 8, 8)

    flat = z.detach().permute(0, 2, 3, 1).reshape(-1, 6).numpy()
    codebook = fit_kmeans(flat, k=8, seed=5, iters=2, max_points=1000)
    ids, q = nearest_codes(z, codebook)
    assert codebook.shape == (8, 6)
    assert ids.shape == (2, 8, 8)
    assert q.shape == z.shape
    assert int(ids.min()) >= 0
    assert int(ids.max()) < 8
    assert model.decode(q).shape == x.shape


def test_spatial_token_dynamics_output_shape():
    model = TokenDynamics(k=16, context=3, spatial=64, d_model=8)
    x = torch.randint(0, 16, (4, 3, 64))
    logits = model(x)
    assert logits.shape == (4, 64, 16)
