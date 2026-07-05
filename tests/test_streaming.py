from __future__ import annotations

import numpy as np

from vocca.control import ControlSpec


def test_stream_concat_equals_generate(generator):
    """流式各块拼接 == 整段生成（相同种子）——核心一致性保证。"""
    prompt = "你好世界"
    control = ControlSpec(emotion="happy", style="news")
    full = generator.generate(prompt, control=control)
    chunks = list(generator.stream(prompt, control=control, chunk_tokens=4))
    streamed = np.concatenate([c.waveform for c in chunks])
    assert np.array_equal(streamed, full.audio.waveform)


def test_last_chunk_flagged(generator):
    chunks = list(generator.stream("测试", chunk_tokens=4))
    assert chunks[-1].is_final
    assert sum(c.is_final for c in chunks) == 1


def test_chunk_indices_monotonic(generator):
    chunks = list(generator.stream("测试文本", chunk_tokens=4))
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_token_counts_sum(generator):
    prompt = "你好世界"
    full = generator.generate(prompt)
    chunks = list(generator.stream(prompt, chunk_tokens=3))
    assert sum(c.n_tokens for c in chunks) == len(full.tokens)
