from __future__ import annotations

import numpy as np

from vocca.config import ModelConfig
from vocca.model import init_weights
from vocca.model.layers import rope_cache
from vocca.model.lm import new_cache
from vocca.model.transformer import decode, transformer_block


def test_transformer_block_shape():
    cfg = ModelConfig(dim=16, n_heads=4, n_kv_heads=2, n_layers=1)
    w = init_weights(cfg)
    cos, sin = rope_cache(5, cfg.head_dim)
    x = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32)
    out = transformer_block(x, w.blocks[0], cos, sin, cfg.n_heads, cfg.n_kv_heads)
    assert out.shape == (5, 16)


def test_decode_shape():
    cfg = ModelConfig(dim=16, n_heads=4, n_kv_heads=2, n_layers=3)
    w = init_weights(cfg)
    cos, sin = rope_cache(4, cfg.head_dim)
    x = np.random.default_rng(1).standard_normal((4, 16)).astype(np.float32)
    out = decode(x, w, cos, sin)
    assert out.shape == (4, 16)


def test_decode_cached_matches_full():
    cfg = ModelConfig(dim=16, n_heads=4, n_kv_heads=2, n_layers=2)
    w = init_weights(cfg)
    seq = 5
    cos, sin = rope_cache(seq, cfg.head_dim)
    x = np.random.default_rng(2).standard_normal((seq, 16)).astype(np.float32)
    full = decode(x, w, cos, sin)

    caches = new_cache(cfg)
    rows = []
    for i in range(seq):
        rows.append(decode(x[i : i + 1], w, cos[i : i + 1], sin[i : i + 1], caches)[0])
    assert np.allclose(full, np.stack(rows), atol=1e-5)
