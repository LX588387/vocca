from __future__ import annotations

import numpy as np
import pytest

from vocca.codec import GriffinLim, MockCodec, OscillatorCodec, build_codec, register_codec
from vocca.config import CodecConfig
from vocca.exceptions import RegistryError


def test_oscillator_decode_length():
    codec = OscillatorCodec(sample_rate=8000, frame_size=160, codebook_size=32)
    tokens = np.array([1, 5, 10, 3], dtype=np.int64)
    wav = codec.decode(tokens)
    assert len(wav) == 4 * 160
    assert codec.num_samples(4) == 640


def test_oscillator_audible():
    codec = OscillatorCodec(codebook_size=32)
    wav = codec.decode(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    assert np.max(np.abs(wav)) > 0.05  # 不是静音


def test_oscillator_streaming_equals_full():
    """任意分块解码再拼接 == 整段解码（逐样本）。"""
    codec = OscillatorCodec(sample_rate=8000, frame_size=100, codebook_size=32)
    tokens = np.array([3, 7, 1, 9, 4, 2, 8, 5], dtype=np.int64)
    full = codec.decode(tokens)

    state = codec.initial_state()
    parts = []
    for group in ([3, 7], [1], [9, 4, 2], [8, 5]):
        wav, state = codec.decode_chunk(np.array(group, dtype=np.int64), state)
        parts.append(wav)
    streamed = np.concatenate(parts)
    assert np.array_equal(full, streamed)


def test_oscillator_empty():
    codec = OscillatorCodec()
    wav, state = codec.decode_chunk(np.array([], dtype=np.int64), codec.initial_state())
    assert len(wav) == 0


def test_oscillator_special_tokens_are_silent():
    codec = OscillatorCodec(codebook_size=32)
    wav = codec.decode(np.array([32, 33], dtype=np.int64))  # BOS/EOS 区
    assert np.max(np.abs(wav)) < 1e-6


def test_oscillator_encode_roundtrip_shape():
    codec = OscillatorCodec(sample_rate=8000, frame_size=160, codebook_size=32)
    wav = codec.decode(np.array([5, 10, 15], dtype=np.int64))
    tokens = codec.encode(wav)
    assert len(tokens) == 3


def test_mock_codec_levels():
    codec = MockCodec(frame_size=10, codebook_size=100)
    wav = codec.decode(np.array([0, 50, 99], dtype=np.int64))
    assert len(wav) == 30
    assert np.isclose(wav[0], -0.5)


def test_mock_encode_roundtrip():
    codec = MockCodec(frame_size=10, codebook_size=100)
    tokens = np.array([10, 40, 80], dtype=np.int64)
    recovered = codec.encode(codec.decode(tokens))
    assert np.allclose(recovered, tokens, atol=1)


def test_griffin_lim_reduces_error():
    from vocca.audio import stft

    t = np.arange(4000) / 8000
    wav = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    mag = np.abs(stft(wav, n_fft=256, hop=64))
    gl_few = GriffinLim(n_fft=256, hop=64, n_iter=1)
    gl_many = GriffinLim(n_fft=256, hop=64, n_iter=30)
    err_few = gl_few.spectral_convergence(gl_few(mag), mag)
    err_many = gl_many.spectral_convergence(gl_many(mag), mag)
    assert err_many <= err_few


def test_build_codec_and_register():
    codec = build_codec(CodecConfig(name="oscillator", codebook_size=32))
    assert isinstance(codec, OscillatorCodec)

    register_codec("mock2", MockCodec)
    assert isinstance(build_codec(CodecConfig(name="mock2")), MockCodec)


def test_build_unknown_codec():
    with pytest.raises(RegistryError):
        build_codec(CodecConfig(name="nope"))
