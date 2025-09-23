"""框架内部通用的类型别名与轻量数据结构。

放在一个没有重依赖的模块里，其它模块可以自由 import 而不担心循环依赖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

# 单声道波形：一维 float32 数组，取值范围约定在 [-1, 1]。
Waveform = npt.NDArray[np.float32]

# 特征帧序列，形状 ``(帧数, 特征维度)``，例如 log-mel。
FeatureArray = npt.NDArray[np.float32]

# 隐藏状态 / 嵌入序列，形状 ``(序列长度, 隐藏维度)``。
Hidden = npt.NDArray[np.float32]

# 离散声学 token 序列，一维 int64，来自编解码器（RVQ / 声码器码本）。
TokenSequence = npt.NDArray[np.int64]

# logits，形状 ``(词表大小,)`` 或 ``(序列长度, 词表大小)``。
Logits = npt.NDArray[np.float32]

PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class AudioClip:
    """一段带采样率的单声道音频。

    Attributes:
        waveform: 单声道波形，一维数组。
        sample_rate: 采样率（Hz）。
    """

    waveform: Waveform
    sample_rate: int

    @property
    def duration(self) -> float:
        """时长（秒）。"""
        return float(len(self.waveform)) / float(self.sample_rate)

    def __len__(self) -> int:
        return len(self.waveform)

    def peak(self) -> float:
        """波形绝对值的峰值，常用于快速判断是否为静音。"""
        if len(self.waveform) == 0:
            return 0.0
        return float(np.max(np.abs(self.waveform)))


@dataclass(frozen=True)
class AudioChunk:
    """流式生成时逐块产出的音频片段。

    Attributes:
        waveform: 这一块的波形。
        index: 从 0 开始的块序号。
        is_final: 是否为最后一块。
        n_tokens: 这一块由多少个声学 token 解码而来。
    """

    waveform: Waveform
    index: int
    is_final: bool
    n_tokens: int

    def __len__(self) -> int:
        return len(self.waveform)
