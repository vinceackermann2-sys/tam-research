import numpy as np

from tam_research.physics_tokenizer_innovation import (
    encode_innov_seq,
    features,
    fit_maps,
    q_delta,
)


def _tiny_codecs():
    state_mean = np.array([0.0, 0.0], np.float32)
    state_std = np.array([1.0, 1.0], np.float32)
    state_coarse = np.array([[0.0, 0.0], [1.0, 1.0]], np.float32)
    state_residual = np.array([[0.0, 0.0]], np.float32)
    delta_mean = np.array([0.0, 0.0], np.float32)
    delta_std = np.array([1.0, 1.0], np.float32)
    delta_coarse = np.array([[0.0, 0.0], [0.5, -0.5]], np.float32)
    delta_residual = np.array([[0.0, 0.0]], np.float32)
    return state_mean, state_std, state_coarse, state_residual, delta_mean, delta_std, delta_coarse, delta_residual


def test_matched_nominal_bit_budget():
    assert int(np.log2(2048)) + int(np.log2(128)) == 18


def test_static_zero_innovation_reconstructs_static_sequence():
    codecs = _tiny_codecs()
    raw = np.zeros((2, 4, 2, 3, 3), np.float32)
    decoded = encode_innov_seq(raw, *codecs)
    np.testing.assert_array_equal(decoded, raw)


def test_causal_prefix_is_invariant_to_future_change():
    codecs = _tiny_codecs()
    raw = np.zeros((2, 4, 2, 3, 3), np.float32)
    changed = raw.copy()
    changed[:, 3] = 7.0
    original_decoded = encode_innov_seq(raw, *codecs)
    changed_decoded = encode_innov_seq(changed, *codecs)
    np.testing.assert_array_equal(original_decoded[:, :3], changed_decoded[:, :3])


def test_innovation_projection_is_previous_plus_decoded_delta():
    *_, delta_mean, delta_std, delta_coarse, delta_residual = _tiny_codecs()
    previous = np.ones((1, 2, 3, 3), np.float32)
    proposed = previous.copy()
    decoded_delta = q_delta(proposed - previous, delta_mean, delta_std, delta_coarse, delta_residual)
    projected = previous + decoded_delta
    np.testing.assert_array_equal(projected, previous)


def test_phase7_feature_shape_and_context_map_shape():
    rng = np.random.default_rng(27182818)
    state = rng.normal(size=(3, 2, 4, 4)).astype(np.float32)
    assert features(state).shape == (3, 4, 4, 16)
    context = rng.normal(size=(3, 3, 2, 4, 4)).astype(np.float32)
    assert fit_maps(context).shape == (3, 2, 16)
