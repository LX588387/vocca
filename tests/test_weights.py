from __future__ import annotations

import numpy as np
import pytest

from vocca.config import ModelConfig
from vocca.exceptions import ShapeError
from vocca.model import init_weights, load_weights, save_weights


def test_init_deterministic():
    cfg = ModelConfig(dim=16, n_heads=4, n_layers=2)
    a = init_weights(cfg)
    b = init_weights(cfg)
    assert np.array_equal(a.text_embed, b.text_embed)


def test_shapes():
    cfg = ModelConfig(dim=16, n_heads=4, n_kv_heads=2, n_layers=2, text_vocab=50, codebook_size=40)
    w = init_weights(cfg)
    assert w.text_embed.shape == (50, 16)
    assert w.audio_embed.shape == (42, 16)
    assert w.head.shape == (42, 16)
    assert len(w.blocks) == 2
    assert w.blocks[0].attn.wk.shape == (2 * 4, 16)  # kv_dim x dim


def test_save_load_roundtrip(tmp_path):
    cfg = ModelConfig(dim=16, n_heads=4, n_kv_heads=2, n_layers=2)
    w = init_weights(cfg)
    path = tmp_path / "w.npz"
    save_weights(path, w)
    loaded = load_weights(path, cfg)
    assert np.array_equal(w.head, loaded.head)
    assert np.array_equal(w.blocks[1].w_gate, loaded.blocks[1].w_gate)


def test_load_shape_mismatch(tmp_path):
    cfg = ModelConfig(dim=16, n_heads=4, n_layers=1)
    save_weights(tmp_path / "w.npz", init_weights(cfg))
    other = ModelConfig(dim=16, n_heads=4, n_layers=1, text_vocab=999)
    with pytest.raises(ShapeError):
        load_weights(tmp_path / "w.npz", other)


def test_num_parameters_counts():
    cfg = ModelConfig(dim=16, n_heads=4, n_layers=1)
    assert init_weights(cfg).num_parameters() > 0
