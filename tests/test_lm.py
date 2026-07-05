from __future__ import annotations

import numpy as np
import pytest

from vocca.control import ControlSpec
from vocca.exceptions import ShapeError


def test_forward_shape(tiny_lm):
    text = np.array([1, 2, 3], dtype=np.int64)
    audio = np.array([tiny_lm.config.audio_bos, 4, 5], dtype=np.int64)
    logits = tiny_lm.forward(text, audio)
    assert logits.shape == (3, tiny_lm.config.audio_vocab)


def test_prefill_step_matches_forward(tiny_lm):
    """prefill + step 与整段 forward 逐位置一致。"""
    text = np.array([1, 2, 3, 4], dtype=np.int64)
    control = ControlSpec(emotion="happy")
    audio = np.array([tiny_lm.config.audio_bos, 7, 11, 15], dtype=np.int64)
    full = tiny_lm.forward(text, audio, control)

    cache = tiny_lm.new_cache()
    l0 = tiny_lm.prefill(text, cache, control)
    assert np.allclose(l0, full[0], atol=1e-5)
    for i in range(1, len(audio)):
        li = tiny_lm.step(int(audio[i]), cache)
        assert np.allclose(li, full[i], atol=1e-5)


def test_control_changes_logits(tiny_lm):
    text = np.array([1, 2], dtype=np.int64)
    audio = np.array([tiny_lm.config.audio_bos], dtype=np.int64)
    happy = tiny_lm.forward(text, audio, ControlSpec(emotion="happy"))
    sad = tiny_lm.forward(text, audio, ControlSpec(emotion="sad"))
    assert not np.allclose(happy, sad)


def test_none_control_zero_prefix(tiny_lm):
    text = np.array([1, 2], dtype=np.int64)
    audio = np.array([tiny_lm.config.audio_bos], dtype=np.int64)
    logits = tiny_lm.forward(text, audio, None)
    assert logits.shape == (1, tiny_lm.config.audio_vocab)


def test_bad_text_id_raises(tiny_lm):
    with pytest.raises(ShapeError):
        tiny_lm.forward(np.array([9999], dtype=np.int64), np.array([0], dtype=np.int64))


def test_num_parameters_positive(tiny_lm):
    assert tiny_lm.num_parameters > 0
