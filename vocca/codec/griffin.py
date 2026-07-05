"""Griffin-Lim 声码器：从幅度谱迭代恢复相位，重建波形。

这是一个**真实**的经典算法（Griffin & Lim, 1984）：给定幅度谱，随机初始化
相位，反复在「时域重建 → 重新 STFT → 保幅度换相位」之间迭代，逐步逼近一段
与目标幅度谱吻合的波形。它不依赖神经网络，常用作 mel/幅度声码器的基线，也
方便在没有 GPU 的环境里验证「谱 → 波形」这条路。
"""

from __future__ import annotations

import numpy as np

from ..audio import istft, stft
from ..types import Waveform
from ..utils import rng_from_seed

__all__ = ["GriffinLim"]


class GriffinLim:
    """迭代式幅度谱声码器。

    Args:
        n_fft: FFT 点数。
        hop: 帧移。
        n_iter: 迭代次数，越多相位越收敛。
        seed: 初始随机相位的种子，保证可复现。
    """

    def __init__(self, n_fft: int = 1024, hop: int = 256, n_iter: int = 32, seed: int = 0) -> None:
        if n_iter <= 0:
            raise ValueError("n_iter 必须为正")
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.n_iter = int(n_iter)
        self.seed = int(seed)

    def __call__(self, magnitude: np.ndarray) -> Waveform:
        """从幅度谱 ``(帧数, n_fft//2+1)`` 重建波形。"""
        mag = np.asarray(magnitude, dtype=np.float64)
        if mag.ndim != 2:
            raise ValueError("magnitude 必须是二维 (帧数, 频点)")
        rng = rng_from_seed(self.seed)
        angles = np.exp(2j * np.pi * rng.random(mag.shape))
        spec = mag * angles
        wav = istft(spec, n_fft=self.n_fft, hop=self.hop)
        for _ in range(self.n_iter):
            rebuilt = stft(wav, n_fft=self.n_fft, hop=self.hop)
            phase = np.exp(1j * np.angle(rebuilt))
            # 帧数可能因边界相差 1，对齐到较短的一方。
            n = min(mag.shape[0], phase.shape[0])
            spec = mag[:n] * phase[:n]
            wav = istft(spec, n_fft=self.n_fft, hop=self.hop)
        peak = np.max(np.abs(wav)) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        return wav.astype(np.float32)

    def spectral_convergence(self, wav: Waveform, target_mag: np.ndarray) -> float:
        """衡量重建波形的幅度谱与目标的相对误差（越小越好）。"""
        rebuilt = np.abs(stft(np.asarray(wav), n_fft=self.n_fft, hop=self.hop))
        n = min(rebuilt.shape[0], target_mag.shape[0])
        num = np.linalg.norm(target_mag[:n] - rebuilt[:n])
        den = np.linalg.norm(target_mag[:n]) + 1e-8
        return float(num / den)
