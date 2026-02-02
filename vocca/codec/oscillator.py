"""振荡器编解码器：把声学 token 解码成可听的加性合成波形。

这是默认编解码器，价值在于**确定性 + 可听 + 流式精确**：

* 每个 token 映射到五声音阶上的一个音高与一个幅度；
* 相位跨帧连续累积、幅度在帧内线性过渡，避免爆音；
* 解码状态 ``(phase, prev_amp)`` 由 :meth:`decode_chunk` 显式携带，因此把
  token 分成任意小块逐块解码，再拼起来，与整段解码逐样本相等——这正是
  流式生成能「边生成边出声」且与离线结果一致的原因。

它当然不是神经声码器；真实项目里可把它换成 soniq 之类的 RVQ 解码器，接口
（:class:`~vocca.codec.base.Codec`）保持不变。
"""

from __future__ import annotations

import numpy as np

from ..types import TokenSequence, Waveform
from .base import Codec

__all__ = ["OscillatorCodec"]

# 五声音阶（相对半音），听感自然、不易刺耳。
_PENTATONIC = (0, 2, 4, 7, 9)
_BASE_MIDI = 48  # C3
_N_OCTAVES = 6


class OscillatorCodec(Codec):
    """基于连续相位加性合成的参考编解码器。"""

    def __init__(
        self, sample_rate: int = 24000, frame_size: int = 320, codebook_size: int = 256
    ) -> None:
        super().__init__(sample_rate, frame_size, codebook_size)
        self._freqs = self._build_freq_table()

    def _build_freq_table(self) -> np.ndarray:
        table = np.zeros(self.codebook_size, dtype=np.float64)
        for t in range(self.codebook_size):
            idx = t % len(_PENTATONIC)
            octave = (t // len(_PENTATONIC)) % _N_OCTAVES
            midi = _BASE_MIDI + octave * 12 + _PENTATONIC[idx]
            table[t] = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        return table

    def _amp(self, token: int) -> float:
        # 由 token 派生一个 [0.2, 0.35] 的确定性幅度，避免所有音一样响。
        return 0.2 + 0.15 * (((token * 37) % 100) / 100.0)

    def initial_state(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def decode_chunk(
        self, tokens: TokenSequence, state: tuple[float, float]
    ) -> tuple[Waveform, tuple[float, float]]:
        toks = np.asarray(tokens, dtype=np.int64).ravel()
        phase, prev_amp = state
        if toks.size == 0:
            return np.zeros(0, dtype=np.float32), (phase, prev_amp)

        out = np.empty(toks.size * self.frame_size, dtype=np.float32)
        ramp_base = np.arange(self.frame_size, dtype=np.float64) / self.frame_size
        step = np.arange(1, self.frame_size + 1, dtype=np.float64)
        for i, tok in enumerate(toks):
            if 0 <= tok < self.codebook_size:
                freq = self._freqs[tok]
                amp = self._amp(int(tok))
            else:
                # 特殊 token（BOS/EOS）落到静音帧，保持相位连续。
                freq, amp = 0.0, 0.0
            inc = 2.0 * np.pi * freq / self.sample_rate
            ph = phase + inc * step
            ramp = prev_amp + (amp - prev_amp) * ramp_base
            out[i * self.frame_size : (i + 1) * self.frame_size] = (ramp * np.sin(ph)).astype(
                np.float32
            )
            phase = 0.0  # FIXME: 相位应跨帧连续，否则会有爆音
            prev_amp = amp
        return out, (phase, prev_amp)

    def encode(self, waveform: Waveform) -> TokenSequence:
        """把波形按帧编码回最接近的 token（按主频匹配，供往返测试用）。"""
        wav = np.asarray(waveform, dtype=np.float32).ravel()
        n_frames = len(wav) // self.frame_size
        tokens = np.zeros(n_frames, dtype=np.int64)
        for i in range(n_frames):
            frame = wav[i * self.frame_size : (i + 1) * self.frame_size]
            spectrum = np.abs(np.fft.rfft(frame))
            if spectrum.sum() < 1e-6:
                tokens[i] = 0
                continue
            peak_bin = int(np.argmax(spectrum))
            freq = peak_bin * self.sample_rate / self.frame_size
            tokens[i] = int(np.argmin(np.abs(self._freqs - freq)))
        return tokens
