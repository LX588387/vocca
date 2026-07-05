"""把 :class:`~vocca.control.spec.ControlSpec` 编码成条件向量。

声学模型不认识「happy」这种字符串，需要把控制条件映射到一段固定维度的
实数向量。这里用**确定性哈希嵌入**：情感 / 风格 / 音色各查一张由随机
种子固定下来的嵌入表，连续字段（强度 / 语速 / 音高 / 能量）归一化后拼在
向量前部。整体零训练、完全可复现——它的作用是把控制信号稳定地注入模型，
而不是学出「最好的」情感表示。

得到的条件向量随后会：

* 作为**前缀**参与自回归注意力（见 :mod:`vocca.model.lm`）；
* 在分类器无关引导（CFG）里被替换为「空条件」以估计条件方向。
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..types import Hidden
from .spec import EMOTIONS, STYLES, ControlSpec

__all__ = ["ControlEncoder"]

# 连续字段的归一化区间，需与 ControlSpec 的取值范围一致。
_CONT_RANGES = {
    "intensity": (0.0, 1.0),
    "speed": (0.5, 2.0),
    "pitch": (-12.0, 12.0),
    "energy": (0.25, 4.0),
}
_N_CONT = len(_CONT_RANGES)


class ControlEncoder:
    """确定性的控制条件编码器。

    Args:
        dim: 输出条件向量的维度，需与模型隐藏维度一致。
        seed: 决定哈希嵌入表的随机种子。
        n_prefix: 条件在前缀里占用的 token 数；>1 时把向量切成多段。
    """

    def __init__(self, dim: int, seed: int = 0, n_prefix: int = 1) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正")
        if n_prefix <= 0:
            raise ValueError("n_prefix 必须为正")
        if dim % n_prefix != 0:
            raise ValueError("dim 必须能被 n_prefix 整除")
        self.dim = int(dim)
        self.seed = int(seed)
        self.n_prefix = int(n_prefix)
        self._table_size = 8192
        rng = np.random.default_rng(seed)
        self._table = (rng.standard_normal((self._table_size, dim)) * 0.02).astype(np.float32)
        # 给三类离散字段各留一个命名空间，避免碰撞。
        self._emotion_base = 1000
        self._style_base = 2000
        self._speaker_base = 3000

    @property
    def prefix_len(self) -> int:
        """条件前缀占用的 token 数。"""
        return self.n_prefix

    def _row(self, base: int, key: str) -> Hidden:
        digest = hashlib.sha1(f"{base}:{key}".encode()).hexdigest()
        idx = int(digest, 16) % self._table_size
        return self._table[idx]

    def encode(self, control: ControlSpec | None) -> Hidden:
        """把控制条件编码成形状 ``(n_prefix, dim)`` 的条件向量。

        ``control=None`` 表示「空条件」，返回全零向量——CFG 的无条件分支
        正是靠它实现的。
        """
        if control is None:
            return np.zeros((self.n_prefix, self.dim), dtype=np.float32)

        vec = np.zeros(self.dim, dtype=np.float32)
        vec += self._row(self._emotion_base, control.emotion)
        vec += self._row(self._style_base, control.style)
        vec += self._row(self._speaker_base, control.speaker)

        # 连续字段：归一化到 [0, 1] 后写进向量前 _N_CONT 维，并乘上强度做门控。
        cont = self._continuous(control)
        vec[:_N_CONT] += cont * (0.5 + control.intensity)

        # 情感强度整体缩放，让「更强的情感」在向量范数上也体现出来。
        vec *= 0.5 + control.intensity

        # 情感 / 风格的序号附加一个小扰动，增强可分性。
        vec[_N_CONT + EMOTIONS.index(control.emotion) % (self.dim - _N_CONT)] += 0.1
        vec[_N_CONT + STYLES.index(control.style) % (self.dim - _N_CONT)] += 0.1

        if self.n_prefix == 1:
            return vec.reshape(1, self.dim)
        return self._split(vec)

    def _split(self, vec: Hidden) -> Hidden:
        """把单个条件向量摊到 ``n_prefix`` 个前缀 token 上。

        每个前缀 token 主要携带向量的一段，同时叠加一份缩小的全局信息，
        这样注意力既能看到细分维度，也不至于丢掉整体条件。
        """
        seg = self.dim // self.n_prefix
        out = np.zeros((self.n_prefix, self.dim), dtype=np.float32)
        for i in range(self.n_prefix):
            out[i, i * seg : (i + 1) * seg] = vec[i * seg : (i + 1) * seg]
            out[i] += vec * 0.1
        return out

    @staticmethod
    def _continuous(control: ControlSpec) -> Hidden:
        values = []
        for name, (lo, hi) in _CONT_RANGES.items():
            raw = float(getattr(control, name))
            values.append((raw - lo) / (hi - lo))
        return np.asarray(values, dtype=np.float32)
