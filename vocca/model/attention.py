"""带 KV cache 的多头自注意力（含分组查询 GQA），纯 numpy。

设计目标是**增量解码与整段前向数值一致**：位置编码用绝对位置、因果掩码
按绝对位置构造，因此逐 token 走 cache 得到的每个位置输出，与一次性喂入整
段序列完全相同。这条性质由 ``tests/test_attention.py`` 严格校验，也是流式
生成正确性的基石。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Hidden
from .layers import apply_rope, linear

__all__ = ["LayerCache", "AttentionWeights", "self_attention"]


@dataclass
class AttentionWeights:
    """一层自注意力的四个投影矩阵，均为 ``(out, in)`` 排布。"""

    wq: Hidden
    wk: Hidden
    wv: Hidden
    wo: Hidden


@dataclass
class LayerCache:
    """单层的 KV 缓存，随解码增量追加。"""

    k: Hidden  # (n_kv_heads, cur_len, head_dim)
    v: Hidden

    @property
    def length(self) -> int:
        return self.k.shape[1]

    @classmethod
    def empty(cls, n_kv_heads: int, head_dim: int) -> LayerCache:
        z = np.zeros((n_kv_heads, 0, head_dim), dtype=np.float32)
        return cls(k=z, v=z.copy())

    def append(self, k: Hidden, v: Hidden) -> None:
        self.k = np.concatenate([self.k, k], axis=1)
        self.v = np.concatenate([self.v, v], axis=1)


def _split_heads(x: Hidden, n_heads: int, head_dim: int) -> Hidden:
    # (seq, n_heads*head_dim) -> (n_heads, seq, head_dim)
    seq = x.shape[0]
    return x.reshape(seq, n_heads, head_dim).transpose(1, 0, 2)


def _repeat_kv(x: Hidden, n_rep: int) -> Hidden:
    # (n_kv_heads, seq, head_dim) -> (n_kv_heads*n_rep, seq, head_dim)
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=0)


def self_attention(
    x: Hidden,
    weights: AttentionWeights,
    cos: Hidden,
    sin: Hidden,
    n_heads: int,
    n_kv_heads: int,
    cache: LayerCache | None = None,
) -> Hidden:
    """因果多头自注意力。

    Args:
        x: 输入隐藏状态 ``(seq, dim)``。
        weights: 本层投影权重。
        cos, sin: 与 ``x`` 各 token **绝对位置**对齐的 RoPE 表 ``(seq, head_dim)``。
        n_heads: 查询头数。
        n_kv_heads: 键 / 值头数（GQA；等于 ``n_heads`` 时退化为标准 MHA）。
        cache: 可选 KV 缓存；传入则把本段 K/V 追加进去并对全长做注意力。

    Returns:
        注意力输出 ``(seq, dim)``。
    """
    seq, dim = x.shape
    head_dim = dim // n_heads

    q = _split_heads(linear(x, weights.wq), n_heads, head_dim)
    k = _split_heads(linear(x, weights.wk), n_kv_heads, head_dim)
    v = _split_heads(linear(x, weights.wv), n_kv_heads, head_dim)

    # RoPE 施加在「当前这段」的绝对位置上。
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    past_len = cache.length if cache is not None else 0
    if cache is not None:
        cache.append(k, v)
        k, v = cache.k, cache.v

    # GQA：把 kv 头复制到 q 头数。
    n_rep = n_heads // n_kv_heads
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    scale = 1.0 / np.sqrt(head_dim)
    scores = np.einsum("hqd,hkd->hqk", q, k) * scale  # (n_heads, seq, total)

    # 因果掩码：绝对位置 (past_len + i) 只能看 j <= past_len + i。
    total = k.shape[1]
    qpos = past_len + np.arange(seq)[:, None]
    kpos = np.arange(total)[None, :]
    mask = kpos > qpos  # True 处需要屏蔽
    scores = np.where(mask[None, :, :], -np.inf, scores)

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights_ = np.exp(scores)
    weights_ /= np.sum(weights_, axis=-1, keepdims=True)

    out = np.einsum("hqk,hkd->hqd", weights_.astype(np.float32), v)  # (n_heads, seq, head_dim)
    out = out.transpose(1, 0, 2).reshape(seq, dim)
    return linear(out, weights.wo)
