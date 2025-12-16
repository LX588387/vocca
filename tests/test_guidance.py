from __future__ import annotations

import numpy as np
import pytest

from vocca.guidance import classifier_free_guidance


def test_scale_one_returns_cond():
    cond = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    uncond = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert np.array_equal(classifier_free_guidance(cond, uncond, 1.0), cond)


def test_extrapolation():
    cond = np.array([2.0], dtype=np.float32)
    uncond = np.array([1.0], dtype=np.float32)
    out = classifier_free_guidance(cond, uncond, 2.0)
    assert out[0] == pytest.approx(3.0)  # 1 + 2*(2-1)


def test_scale_zero_ignores_condition():
    cond = np.array([5.0], dtype=np.float32)
    uncond = np.array([1.0], dtype=np.float32)
    out = classifier_free_guidance(cond, uncond, 0.0)
    assert out[0] == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        classifier_free_guidance(np.zeros(3), np.zeros(2), 1.5)
