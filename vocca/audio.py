"""音频 I/O 与基础 DSP，纯 numpy + 标准库 ``wave``。

提供 16-bit PCM WAV 的读写、STFT/ISTFT、mel 滤波器组与 log-mel 谱。刻意不依赖
``librosa`` / ``scipy``，让核心链路零重依赖；需要更专业的重采样 / 编解码时再
装可选的 ``soundfile`` 附加依赖。
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .exceptions import AudioIOError
from .types import FeatureArray, PathLike, Waveform

__all__ = [
    "save_wav",
    "load_wav",
    "stft",
    "istft",
    "mel_filterbank",
    "mel_spectrogram",
    "log_mel_spectrogram",
    "hz_to_mel",
    "mel_to_hz",
]

_INT16_MAX = 32767.0


def save_wav(path: PathLike, waveform: Waveform, sample_rate: int) -> None:
    """把 [-1, 1] 的单声道 float 波形写成 16-bit PCM WAV。"""
    if sample_rate <= 0:
        raise AudioIOError("sample_rate 必须为正")
    wav = np.asarray(waveform, dtype=np.float32).ravel()
    clipped = np.clip(wav, -1.0, 1.0)
    pcm = (clipped * _INT16_MAX).astype("<i2")
    try:
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(int(sample_rate))
            fh.writeframes(pcm.tobytes())
    except (OSError, wave.Error) as exc:
        raise AudioIOError(f"写 WAV 失败：{exc}") from exc


def load_wav(path: PathLike) -> tuple[Waveform, int]:
    """读 16-bit PCM WAV，返回 ``(波形 float32, 采样率)``。多声道会取平均降为单声道。"""
    if not Path(path).exists():
        raise AudioIOError(f"文件不存在：{path}")
    try:
        with wave.open(str(path), "rb") as fh:
            n_channels = fh.getnchannels()
            sample_rate = fh.getframerate()
            width = fh.getsampwidth()
            frames = fh.readframes(fh.getnframes())
    except (OSError, wave.Error) as exc:
        raise AudioIOError(f"读 WAV 失败：{exc}") from exc
    if width != 2:
        raise AudioIOError(f"仅支持 16-bit PCM，收到 {width * 8}-bit")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / _INT16_MAX
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), int(sample_rate)


def _hann(n: int) -> Waveform:
    if n == 1:
        return np.ones(1, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)


def stft(x: Waveform, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """短时傅里叶变换，返回复数谱 ``(帧数, n_fft//2 + 1)``。"""
    x = np.asarray(x, dtype=np.float32)
    window = _hann(n_fft)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] * window for i in range(n_frames)], axis=0)
    return np.fft.rfft(frames, axis=-1)


def istft(spec: np.ndarray, n_fft: int = 1024, hop: int = 256) -> Waveform:
    """ISTFT，带重叠相加与窗能量归一化，近似还原 :func:`stft` 的输入。"""
    window = _hann(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=-1)
    n_frames = frames.shape[0]
    length = (n_frames - 1) * hop + n_fft
    out = np.zeros(length, dtype=np.float32)
    norm = np.zeros(length, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        out[start : start + n_fft] += frames[i] * window
        norm[start : start + n_fft] += window**2
    norm[norm < 1e-8] = 1.0
    return (out / norm).astype(np.float32)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """HTK 风格的 Hz → mel。"""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """mel → Hz。"""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """构造三角 mel 滤波器组 ``(n_mels, n_fft//2 + 1)``。"""
    fmax = fmax if fmax is not None else sample_rate / 2
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        rising = (freqs - left) / max(center - left, 1e-8)
        falling = (right - freqs) / max(right - center, 1e-8)
        fb[m - 1] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


def mel_spectrogram(
    x: Waveform,
    sample_rate: int,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 80,
) -> FeatureArray:
    """幅度谱经 mel 滤波得到的 mel 频谱 ``(帧数, n_mels)``。"""
    mag = np.abs(stft(x, n_fft=n_fft, hop=hop))
    fb = mel_filterbank(sample_rate, n_fft=n_fft, n_mels=n_mels)
    return (mag @ fb.T).astype(np.float32)


def log_mel_spectrogram(
    x: Waveform,
    sample_rate: int,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 80,
    eps: float = 1e-6,
) -> FeatureArray:
    """log-mel 谱：对 :func:`mel_spectrogram` 取自然对数。"""
    return np.log(mel_spectrogram(x, sample_rate, n_fft, hop, n_mels) + eps).astype(np.float32)
