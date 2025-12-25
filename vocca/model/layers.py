"""Transformer 用到的基础算子，纯 numpy 实现。

包含 RMSNorm、旋转位置编码（RoPE）、GELU 与线性层封装。这些函数都是
无状态的纯函数（权重从外部传入），方便 :mod:`vocca.model.attention` 与
:mod:`vocca.model.transformer` 复用，也便于单测逐个校验。
"""

from __future__ import annotations

import numpy as np

from ..types import Hidden

__all__ = ["rms_norm", "gelu", "silu", "linear", "rope_cache", "apply_rope"]


def rms_norm(x: Hidden, weight: Hidden, eps: float = 1e-6) -> Hidden:
    """RMSNorm：按最后一维做均方根归一化，再逐通道缩放。

    相比 LayerNorm 少了去均值一步，是近年解码器（LLaMA 系）的常见选择。
    """
    x = np.asarray(x, dtype=np.float32)
    ms = np.mean(np.square(x), axis=-1, keepdims=True)
    normed = x / np.sqrt(ms + eps)
    return (normed * weight).astype(np.float32)


def gelu(x: Hidden) -> Hidden:
    """GELU 激活（tanh 近似）。"""
    x = np.asarray(x, dtype=np.float32)
    c = np.sqrt(2.0 / np.pi).astype(np.float32)
    inner = c * (x + 0.044715 * np.power(x, 3))
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(np.float32)


def silu(x: Hidden) -> Hidden:
    """SiLU / Swish 激活：``x * sigmoid(x)``，SwiGLU 前馈网络的门控用。"""
    x = np.asarray(x, dtype=np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def linear(x: Hidden, weight: Hidden, bias: Hidden | None = None) -> Hidden:
    """全连接：``x @ weightᵀ (+ bias)``。

    ``weight`` 形状 ``(out, in)``，与 PyTorch ``nn.Linear`` 的约定一致，
    方便未来直接搬运权重。
    """
    out = np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T
    if bias is not None:
        out = out + bias
    return out.astype(np.float32)


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[Hidden, Hidden]:
    """预计算 RoPE 的 cos / sin 表，形状均为 ``(seq_len, head_dim)``。

    偶数维为一组旋转对，这里把每个频率复制到相邻两维，方便后续按元素相乘。
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim 必须为偶数才能用 RoPE")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(seq_len, dtype=np.float64)[:, None]
    ang = pos * inv_freq[None, :]  # (seq_len, half)
    cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1)
    return cos.astype(np.float32), sin.astype(np.float32)


def apply_rope(x: Hidden, cos: Hidden, sin: Hidden) -> Hidden:
    """对形状 ``(..., seq, head_dim)`` 的张量施加旋转位置编码。

    ``cos`` / ``sin`` 需与 ``x`` 的 ``(seq, head_dim)`` 对齐（可广播头维度）。
    """
    x = np.asarray(x, dtype=np.float32)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return (x * cos + rotated * sin).astype(np.float32)
