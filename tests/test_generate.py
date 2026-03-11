from __future__ import annotations

import numpy as np

from vocca.config import GenerationConfig
from vocca.control import ControlSpec
from vocca.generate import generate_tokens, iter_tokens


def test_generate_tokens_deterministic(tiny_lm):
    text = np.array([1, 2, 3], dtype=np.int64)
    cfg = GenerationConfig(max_new_tokens=20, seed=42)
    a = generate_tokens(tiny_lm, text, None, cfg)
    b = generate_tokens(tiny_lm, text, None, cfg)
    assert np.array_equal(a.tokens, b.tokens)


def test_generated_tokens_in_codebook(tiny_lm):
    text = np.array([1, 2], dtype=np.int64)
    cfg = GenerationConfig(max_new_tokens=30, seed=0)
    stream = generate_tokens(tiny_lm, text, None, cfg)
    assert np.all(stream.tokens < tiny_lm.config.codebook_size)
    assert np.all(stream.tokens >= 0)


def test_seed_changes_output(tiny_lm):
    text = np.array([1, 2], dtype=np.int64)
    a = generate_tokens(tiny_lm, text, None, GenerationConfig(max_new_tokens=20, seed=1))
    b = generate_tokens(tiny_lm, text, None, GenerationConfig(max_new_tokens=20, seed=2))
    assert not np.array_equal(a.tokens, b.tokens)


def test_cfg_changes_output(tiny_lm):
    text = np.array([1, 2], dtype=np.int64)
    control = ControlSpec(emotion="happy")
    plain = generate_tokens(
        tiny_lm, text, control, GenerationConfig(max_new_tokens=20, seed=0, guidance_scale=1.0)
    )
    guided = generate_tokens(
        tiny_lm, text, control, GenerationConfig(max_new_tokens=20, seed=0, guidance_scale=3.0)
    )
    assert not np.array_equal(plain.tokens, guided.tokens)


def test_iter_tokens_yields_ints(tiny_lm):
    text = np.array([1], dtype=np.int64)
    toks = list(iter_tokens(tiny_lm, text, None, GenerationConfig(max_new_tokens=5, seed=0)))
    assert all(isinstance(t, int) for t in toks)


def test_greedy_generation(tiny_lm):
    text = np.array([1, 2], dtype=np.int64)
    cfg = GenerationConfig(max_new_tokens=10, temperature=0.0)
    a = generate_tokens(tiny_lm, text, None, cfg)
    b = generate_tokens(tiny_lm, text, None, cfg)
    assert np.array_equal(a.tokens, b.tokens)  # 贪心也确定
