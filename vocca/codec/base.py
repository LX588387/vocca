"""编解码器抽象：声学 token ↔ 波形。

生成侧只需要 ``decode``（token → 波形）；``encode`` 可选，便于做往返测试或
把提示音频转成 token。为支持**流式且与整段解码数值一致**，解码统一走
``decode_chunk(tokens, state)``：它接收上一块留下的状态（相位、重叠尾巴等），
返回本块波形与新状态。整段 ``decode`` 只是对全部 token 调用一次 ``decode_chunk``，
因此「分块解码再拼接」与「一次性解码」逐样本相等。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..types import TokenSequence, Waveform

__all__ = ["Codec"]


class Codec(ABC):
    """所有编解码器的基类。"""

    def __init__(
        self, sample_rate: int = 24000, frame_size: int = 320, codebook_size: int = 256
    ) -> None:
        if sample_rate <= 0 or frame_size <= 0 or codebook_size <= 0:
            raise ValueError("sample_rate / frame_size / codebook_size 必须为正")
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.codebook_size = int(codebook_size)

    @abstractmethod
    def initial_state(self) -> Any:
        """返回流式解码的初始状态。"""

    @abstractmethod
    def decode_chunk(self, tokens: TokenSequence, state: Any) -> tuple[Waveform, Any]:
        """解码一小段 token，返回 ``(波形, 新状态)``。"""

    def decode(self, tokens: TokenSequence) -> Waveform:
        """整段解码：等价于用初始状态一次性 ``decode_chunk``。"""
        wav, _ = self.decode_chunk(np.asarray(tokens, dtype=np.int64), self.initial_state())
        return wav

    def encode(self, waveform: Waveform) -> TokenSequence:  # pragma: no cover - 可选能力
        """可选：把波形编码回 token。默认未实现。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 encode")

    @property
    def frame_rate(self) -> float:
        """每秒 token 帧数。"""
        return self.sample_rate / self.frame_size

    def num_samples(self, n_tokens: int) -> int:
        """``n_tokens`` 个 token 解码后的样本数。"""
        return n_tokens * self.frame_size
