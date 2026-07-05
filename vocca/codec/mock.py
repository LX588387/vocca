"""最简编解码器：token → 定值直流帧。

没有任何合成，纯粹把每个 token 映射成一段常数电平（``token / codebook`` 归一化
后减 0.5）。它不好听，但绝对确定、无三角函数、便于在单测里精确断言样本值，
也用作 :class:`~vocca.codec.base.Codec` 接口的最小示例。
"""

from __future__ import annotations

import numpy as np

from ..types import TokenSequence, Waveform
from .base import Codec

__all__ = ["MockCodec"]


class MockCodec(Codec):
    """确定性直流编解码器，主要供测试使用。"""

    def initial_state(self) -> None:
        return None

    def _level(self, token: int) -> float:
        if not 0 <= token < self.codebook_size:
            return 0.0
        return token / self.codebook_size - 0.5

    def decode_chunk(self, tokens: TokenSequence, state: None) -> tuple[Waveform, None]:
        toks = np.asarray(tokens, dtype=np.int64).ravel()
        levels = np.array([self._level(int(t)) for t in toks], dtype=np.float32)
        wav = np.repeat(levels, self.frame_size)
        return wav, None

    def encode(self, waveform: Waveform) -> TokenSequence:
        wav = np.asarray(waveform, dtype=np.float32).ravel()
        n_frames = len(wav) // self.frame_size
        tokens = np.zeros(n_frames, dtype=np.int64)
        for i in range(n_frames):
            level = float(np.mean(wav[i * self.frame_size : (i + 1) * self.frame_size]))
            tokens[i] = int(
                np.clip(round((level + 0.5) * self.codebook_size), 0, self.codebook_size - 1)
            )
        return tokens
