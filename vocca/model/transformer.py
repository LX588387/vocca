"""Transformer 解码器：块前向 + 整栈前向，支持逐层 KV cache。

前馈用 SwiGLU，注意力前后各接一个 RMSNorm，残差连接。整栈前向对外只暴露
两个函数：:func:`transformer_block` 与 :func:`decode`，权重与 cache 都由调用
方（:mod:`vocca.model.lm`）持有，保持无状态、易测。
"""

from __future__ import annotations

import numpy as np

from ..types import Hidden
from .attention import LayerCache, self_attention
from .layers import linear, rms_norm, silu
from .weights import BlockWeights, ModelWeights

__all__ = ["transformer_block", "decode"]


def _swiglu(x: Hidden, block: BlockWeights) -> Hidden:
    gate = silu(linear(x, block.w_gate))
    up = linear(x, block.w_up)
    return linear(gate * up, block.w_down)


def transformer_block(
    x: Hidden,
    block: BlockWeights,
    cos: Hidden,
    sin: Hidden,
    n_heads: int,
    n_kv_heads: int,
    cache: LayerCache | None = None,
) -> Hidden:
    """单个解码器块：pre-norm 注意力 + pre-norm SwiGLU 前馈。"""
    h = x + self_attention(
        rms_norm(x, block.attn_norm), block.attn, cos, sin, n_heads, n_kv_heads, cache
    )
    out = h + _swiglu(rms_norm(h, block.ffn_norm), block)
    return out.astype(np.float32)


def decode(
    hidden: Hidden,
    weights: ModelWeights,
    cos: Hidden,
    sin: Hidden,
    caches: list[LayerCache] | None = None,
) -> Hidden:
    """整个解码器前向，返回经过 final norm 的隐藏状态 ``(seq, dim)``。

    ``caches`` 为逐层 :class:`LayerCache` 列表；``None`` 表示整段一次性前向
    （训练 / 校验用），传入则做增量解码。
    """
    cfg = weights.config
    x = hidden
    for i, block in enumerate(weights.blocks):
        layer_cache = caches[i] if caches is not None else None
        x = transformer_block(x, block, cos, sin, cfg.n_heads, cfg.n_kv_heads, layer_cache)
    return rms_norm(x, weights.final_norm)
