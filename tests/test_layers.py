from __future__ import annotations

import numpy as np

from vocca.model.layers import apply_rope, gelu, linear, rms_norm, rope_cache, silu


def test_rms_norm_unit_scale():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    out = rms_norm(x, np.ones(2, dtype=np.float32))
    # RMS of [3,4] = 3.5355; normalized vector has RMS ~1
    assert np.isclose(np.sqrt(np.mean(out**2)), 1.0, atol=1e-3)


def test_gelu_monotone_positive():
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    out = gelu(x)
    assert out[0] < out[1] < out[2]
    assert np.isclose(out[1], 0.0, atol=1e-6)


def test_silu_zero_at_zero():
    assert np.isclose(silu(np.array([0.0]))[0], 0.0)


def test_linear_matches_manual():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    w = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    out = linear(x, w)
    assert out.shape == (1, 3)
    assert np.allclose(out[0], [1.0, 2.0, 3.0])


def test_linear_with_bias():
    x = np.zeros((1, 2), dtype=np.float32)
    w = np.ones((2, 2), dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)
    out = linear(x, w, b)
    assert np.allclose(out[0], [1.0, 2.0])


def test_rope_cache_shapes():
    cos, sin = rope_cache(10, 8)
    assert cos.shape == (10, 8)
    assert sin.shape == (10, 8)


def test_rope_position_zero_is_identity():
    cos, sin = rope_cache(4, 8)
    x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
    out = apply_rope(x, cos[:1], sin[:1])
    assert np.allclose(out, x, atol=1e-6)  # 位置 0：cos=1, sin=0


def test_rope_preserves_norm():
    cos, sin = rope_cache(4, 8)
    x = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32)
    out = apply_rope(x, cos, sin)
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(out, axis=-1), atol=1e-5)
