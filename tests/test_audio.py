from __future__ import annotations

import numpy as np
import pytest

from vocca.audio import (
    load_wav,
    log_mel_spectrogram,
    mel_filterbank,
    mel_spectrogram,
    save_wav,
    stft,
)
from vocca.exceptions import AudioIOError


def _tone(freq=220.0, sr=8000, secs=0.5):
    t = np.arange(int(sr * secs)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_wav_roundtrip(tmp_path):
    wav = _tone()
    path = tmp_path / "a.wav"
    save_wav(path, wav, 8000)
    loaded, sr = load_wav(path)
    assert sr == 8000
    assert len(loaded) == len(wav)
    assert np.max(np.abs(loaded - wav)) < 1e-3  # 16-bit 量化误差


def test_save_clips_out_of_range(tmp_path):
    path = tmp_path / "clip.wav"
    save_wav(path, np.array([2.0, -2.0], dtype=np.float32), 8000)
    loaded, _ = load_wav(path)
    assert loaded.max() <= 1.0 and loaded.min() >= -1.0


def test_save_bad_sample_rate(tmp_path):
    with pytest.raises(AudioIOError):
        save_wav(tmp_path / "x.wav", np.zeros(10, dtype=np.float32), 0)


def test_load_missing_file(tmp_path):
    with pytest.raises(AudioIOError):
        load_wav(tmp_path / "nope.wav")


def test_stft_shape():
    wav = _tone()
    spec = stft(wav, n_fft=256, hop=64)
    assert spec.shape[1] == 256 // 2 + 1


def test_mel_filterbank_shape():
    fb = mel_filterbank(8000, n_fft=256, n_mels=40)
    assert fb.shape == (40, 129)
    assert np.all(fb >= 0)


def test_mel_spectrogram_shape():
    mel = mel_spectrogram(_tone(), 8000, n_fft=256, hop=64, n_mels=40)
    assert mel.shape[1] == 40


def test_log_mel_finite():
    lm = log_mel_spectrogram(_tone(), 8000, n_fft=256, hop=64, n_mels=40)
    assert np.all(np.isfinite(lm))
