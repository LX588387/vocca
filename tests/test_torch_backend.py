from __future__ import annotations

import numpy as np
import pytest

from vocca.control import ControlSpec
from vocca.integrations import torch_available
from vocca.model import AcousticLM

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")


def test_torch_matches_numpy_forward():
    from vocca.config import ModelConfig
    from vocca.integrations import forward_logits_torch

    cfg = ModelConfig(dim=32, n_heads=4, n_kv_heads=2, n_layers=2, text_vocab=64, codebook_size=32)
    lm = AcousticLM(cfg)
    text = np.array([1, 2, 3], dtype=np.int64)
    audio = np.array([cfg.audio_bos, 4, 9], dtype=np.int64)
    control = ControlSpec(emotion="happy")

    numpy_logits = lm.forward(text, audio, control)
    torch_logits = forward_logits_torch(lm, text, audio, control)
    assert np.allclose(numpy_logits, torch_logits, atol=1e-4)


def test_torch_available_true():
    assert torch_available() is True
