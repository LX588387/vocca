"""从 logits 采样声学 token 的各种策略。

实现温度、top-k、top-p（核采样）与重复惩罚，并把它们组合进
:func:`sample_token`。所有函数都不修改入参、只依赖传入的 ``rng``，因此在
相同种子下完全可复现。
"""

from __future__ import annotations

import numpy as np

from .types import Logits, TokenSequence
from .utils import softmax

__all__ = [
    "apply_repetition_penalty",
    "top_k_filter",
    "top_p_filter",
    "sample_token",
    "greedy",
]


def greedy(logits: Logits) -> int:
    """直接取 argmax，等价于温度趋近 0。"""
    return int(np.argmax(np.asarray(logits)))


def apply_repetition_penalty(logits: Logits, history: TokenSequence, penalty: float) -> Logits:
    """对已出现过的 token 施加重复惩罚（CTRL 论文的做法）。

    正 logit 除以 penalty、负 logit 乘以 penalty，从而压低其概率。
    ``penalty == 1.0`` 时原样返回。
    """
    if penalty == 1.0 or len(history) == 0:
        return logits
    out = np.array(logits, dtype=np.float32, copy=True)
    seen = np.unique(np.asarray(history, dtype=np.int64))
    seen = seen[(seen >= 0) & (seen < out.shape[-1])]
    pos = out[seen] > 0
    out[seen[pos]] /= penalty
    out[seen[~pos]] *= penalty
    return out


def top_k_filter(logits: Logits, k: int) -> Logits:
    """只保留概率最高的 k 个 logit，其余置为 ``-inf``。``k<=0`` 表示不过滤。"""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    out = np.array(logits, dtype=np.float32, copy=True)
    kth = np.partition(out, -k)[-k]
    out[out < kth] = -np.inf
    return out


def top_p_filter(logits: Logits, p: float) -> Logits:
    """核采样：保留累积概率达到 p 的最小 token 集合，其余置为 ``-inf``。"""
    if p >= 1.0:
        return logits
    out = np.array(logits, dtype=np.float32, copy=True)
    order = np.argsort(out)[::-1]
    probs = softmax(out[order])
    cumulative = np.cumsum(probs)
    # 至少保留第一个 token；截断处第一个越过阈值的仍保留。
    keep = cumulative <= p
    keep[0] = True
    removed = order[~keep]
    out[removed] = -np.inf
    return out


def sample_token(
    logits: Logits,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    history: TokenSequence | None = None,
    repetition_penalty: float = 1.0,
) -> int:
    """按给定策略从 logits 采样一个 token id。

    温度为 0 时退化为贪心。过滤顺序：重复惩罚 → top-k → top-p → 采样。
    """
    logits = np.asarray(logits, dtype=np.float32)
    if history is not None:
        logits = apply_repetition_penalty(logits, history, repetition_penalty)
    if temperature <= 0:
        return greedy(logits)
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))
