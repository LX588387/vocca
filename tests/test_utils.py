from __future__ import annotations

import numpy as np
import pytest

from vocca.utils import rng_from_seed, softmax, timer


def test_softmax_sums_to_one():
    out = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(out.sum() - 1.0) < 1e-6
    assert np.all(out > 0)


def test_softmax_temperature_zero_is_onehot():
    out = softmax(np.array([1.0, 5.0, 2.0]), temperature=0.0)
    assert out.tolist() == [0.0, 1.0, 0.0]


def test_softmax_negative_temperature_raises():
    with pytest.raises(ValueError):
        softmax(np.array([1.0, 2.0]), temperature=-1.0)


def test_softmax_low_temperature_sharpens():
    logits = np.array([1.0, 2.0])
    sharp = softmax(logits, temperature=0.1)
    soft = softmax(logits, temperature=2.0)
    assert sharp[1] > soft[1]


def test_rng_reproducible():
    a = rng_from_seed(3).random(5)
    b = rng_from_seed(3).random(5)
    assert np.array_equal(a, b)


def test_timer_measures_positive():
    with timer() as t:
        sum(range(1000))
    assert t.seconds >= 0.0
