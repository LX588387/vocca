from __future__ import annotations

import numpy as np
import pytest

from vocca.control import ControlEncoder, ControlSpec


def test_output_shape_single_prefix():
    enc = ControlEncoder(dim=32, n_prefix=1)
    out = enc.encode(ControlSpec(emotion="happy"))
    assert out.shape == (1, 32)
    assert out.dtype == np.float32


def test_output_shape_multi_prefix():
    enc = ControlEncoder(dim=32, n_prefix=4)
    out = enc.encode(ControlSpec())
    assert out.shape == (4, 32)


def test_none_control_is_zero():
    enc = ControlEncoder(dim=16, n_prefix=2)
    out = enc.encode(None)
    assert out.shape == (2, 16)
    assert np.all(out == 0.0)


def test_deterministic():
    a = ControlEncoder(dim=32, seed=7).encode(ControlSpec(emotion="sad"))
    b = ControlEncoder(dim=32, seed=7).encode(ControlSpec(emotion="sad"))
    assert np.array_equal(a, b)


def test_different_emotion_differs():
    enc = ControlEncoder(dim=32)
    happy = enc.encode(ControlSpec(emotion="happy"))
    sad = enc.encode(ControlSpec(emotion="sad"))
    assert not np.allclose(happy, sad)


def test_intensity_scales_norm():
    enc = ControlEncoder(dim=32)
    low = np.linalg.norm(enc.encode(ControlSpec(emotion="happy", intensity=0.1)))
    high = np.linalg.norm(enc.encode(ControlSpec(emotion="happy", intensity=1.0)))
    assert high > low


@pytest.mark.parametrize(
    "bad", [{"dim": 0}, {"dim": 32, "n_prefix": 0}, {"dim": 30, "n_prefix": 4}]
)
def test_invalid_construction(bad):
    with pytest.raises(ValueError):
        ControlEncoder(**bad)


def test_prefix_len_property():
    assert ControlEncoder(dim=8, n_prefix=2).prefix_len == 2
