from __future__ import annotations

import numpy as np

from vocca.sampling import (
    apply_repetition_penalty,
    greedy,
    sample_token,
    top_k_filter,
    top_p_filter,
)


def test_greedy_picks_argmax():
    assert greedy(np.array([0.1, 5.0, 2.0])) == 1


def test_repetition_penalty_lowers_seen():
    logits = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    out = apply_repetition_penalty(logits, np.array([0]), penalty=2.0)
    assert out[0] < out[1]


def test_repetition_penalty_identity():
    logits = np.array([1.0, 2.0], dtype=np.float32)
    out = apply_repetition_penalty(logits, np.array([0]), penalty=1.0)
    assert np.array_equal(out, logits)


def test_repetition_penalty_negative_logit():
    logits = np.array([-2.0, 1.0], dtype=np.float32)
    out = apply_repetition_penalty(logits, np.array([0]), penalty=2.0)
    assert out[0] < -2.0  # 负 logit 乘惩罚后更小


def test_top_k_filter_masks():
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = top_k_filter(logits, 2)
    assert np.isneginf(out[0]) and np.isneginf(out[1])
    assert out[2] == 3.0 and out[3] == 4.0


def test_top_k_zero_noop():
    logits = np.array([1.0, 2.0], dtype=np.float32)
    assert np.array_equal(top_k_filter(logits, 0), logits)


def test_top_p_keeps_at_least_one():
    logits = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    out = top_p_filter(logits, 0.1)
    assert not np.isneginf(out[0])
    assert np.isneginf(out[1])


def test_top_p_full_noop():
    logits = np.array([1.0, 2.0], dtype=np.float32)
    assert np.array_equal(top_p_filter(logits, 1.0), logits)


def test_sample_token_deterministic_with_seed():
    logits = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    a = sample_token(np.array(logits), np.random.default_rng(0), temperature=1.0)
    b = sample_token(np.array(logits), np.random.default_rng(0), temperature=1.0)
    assert a == b


def test_sample_token_zero_temperature_greedy():
    logits = np.array([1.0, 9.0, 2.0], dtype=np.float32)
    assert sample_token(logits, np.random.default_rng(0), temperature=0.0) == 1


def test_sample_token_respects_top_k():
    logits = np.array([0.0, 0.0, 0.0, 100.0], dtype=np.float32)
    tok = sample_token(logits, np.random.default_rng(1), temperature=1.0, top_k=1)
    assert tok == 3
