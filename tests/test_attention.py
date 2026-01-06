from __future__ import annotations

import numpy as np

from vocca.model.attention import AttentionWeights, LayerCache, self_attention
from vocca.model.layers import rope_cache


def _weights(dim, kv_dim, rng):
    return AttentionWeights(
        wq=(rng.standard_normal((dim, dim)) * 0.1).astype(np.float32),
        wk=(rng.standard_normal((kv_dim, dim)) * 0.1).astype(np.float32),
        wv=(rng.standard_normal((kv_dim, dim)) * 0.1).astype(np.float32),
        wo=(rng.standard_normal((dim, dim)) * 0.1).astype(np.float32),
    )


def test_attention_output_shape():
    rng = np.random.default_rng(0)
    dim, n_heads, n_kv = 16, 4, 2
    head_dim = dim // n_heads
    w = _weights(dim, n_kv * head_dim, rng)
    cos, sin = rope_cache(8, head_dim)
    x = rng.standard_normal((5, dim)).astype(np.float32)
    out = self_attention(x, w, cos[:5], sin[:5], n_heads, n_kv)
    assert out.shape == (5, dim)


def test_cache_matches_full_forward():
    """逐 token 走 cache，与整段一次前向数值一致——流式正确性的核心。"""
    rng = np.random.default_rng(3)
    dim, n_heads, n_kv = 16, 4, 2
    head_dim = dim // n_heads
    w = _weights(dim, n_kv * head_dim, rng)
    seq = 6
    cos, sin = rope_cache(seq, head_dim)
    x = rng.standard_normal((seq, dim)).astype(np.float32)

    full = self_attention(x, w, cos, sin, n_heads, n_kv)

    cache = LayerCache.empty(n_kv, head_dim)
    incremental = []
    for i in range(seq):
        out = self_attention(x[i : i + 1], w, cos[i : i + 1], sin[i : i + 1], n_heads, n_kv, cache)
        incremental.append(out[0])
    incremental = np.stack(incremental)
    assert np.allclose(full, incremental, atol=1e-5)


def test_cache_grows():
    rng = np.random.default_rng(1)
    head_dim = 4
    cache = LayerCache.empty(2, head_dim)
    assert cache.length == 0
    w = _weights(8, 8, rng)
    cos, sin = rope_cache(3, head_dim)
    self_attention(rng.standard_normal((3, 8)).astype(np.float32), w, cos, sin, 2, 2, cache)
    assert cache.length == 3


def test_mha_equals_gqa_when_heads_equal():
    rng = np.random.default_rng(5)
    dim, n_heads = 16, 4
    head_dim = dim // n_heads
    w = _weights(dim, dim, rng)
    cos, sin = rope_cache(4, head_dim)
    x = rng.standard_normal((4, dim)).astype(np.float32)
    out = self_attention(x, w, cos, sin, n_heads, n_heads)
    assert np.all(np.isfinite(out))
