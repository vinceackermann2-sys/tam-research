import numpy as np

from tam_research.physics_tokenizer_gray_scott import (
    FEATURE_DIM,
    generate_gray_scott,
    local_polynomial_features,
    periodic_laplacian,
)


def test_gray_scott_generator_is_finite_and_shaped():
    x, feed, kill = generate_gray_scott(
        3, 5, 8, (0.025, 0.045), (0.055, 0.065), seed=123, substeps=2
    )
    assert x.shape == (3, 5, 2, 8, 8)
    assert feed.shape == (3,)
    assert kill.shape == (3,)
    assert np.isfinite(x).all()


def test_local_features_and_periodic_laplacian():
    state = np.zeros((2, 2, 8, 8), dtype=np.float32)
    state[:, 0] = 1.0
    feat = local_polynomial_features(state)
    assert feat.shape == (2, 8, 8, FEATURE_DIM)
    assert np.allclose(periodic_laplacian(np.ones((8, 8), np.float32)), 0.0)
