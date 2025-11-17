from __future__ import annotations

import numpy as np

from vocca.types import AudioChunk, AudioClip


def test_audio_clip_duration_and_len():
    clip = AudioClip(waveform=np.zeros(16000, dtype=np.float32), sample_rate=8000)
    assert len(clip) == 16000
    assert clip.duration == 2.0
    assert clip.peak() == 0.0


def test_audio_clip_peak():
    wav = np.array([0.0, -0.7, 0.3], dtype=np.float32)
    clip = AudioClip(waveform=wav, sample_rate=8000)
    assert abs(clip.peak() - 0.7) < 1e-6


def test_audio_clip_empty_peak():
    assert AudioClip(np.zeros(0, dtype=np.float32), 8000).peak() == 0.0


def test_audio_chunk_len():
    chunk = AudioChunk(np.zeros(100, dtype=np.float32), index=0, is_final=True, n_tokens=5)
    assert len(chunk) == 100
    assert chunk.is_final and chunk.n_tokens == 5
