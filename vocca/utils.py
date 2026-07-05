"""零散的小工具：随机数种子、计时、数值稳定的 softmax。

刻意保持轻量，只依赖 numpy。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from .types import Logits


def rng_from_seed(seed: int | None) -> np.random.Generator:
    """从整数种子构造一个 ``numpy`` 随机数生成器。

    ``seed=None`` 时使用系统熵源，得到不可复现但相互独立的序列。
    """
    return np.random.default_rng(seed)


def softmax(logits: Logits, axis: int = -1, temperature: float = 1.0) -> Logits:
    """数值稳定的 softmax，可选温度缩放。

    ``temperature`` 越小分布越尖锐；等于 0 时退化为 one-hot（取 argmax）。
    """
    arr = np.asarray(logits, dtype=np.float64)
    if temperature < 0:
        raise ValueError("temperature 不能为负")
    if temperature == 0:
        out = np.zeros_like(arr)
        idx = np.argmax(arr, axis=axis)
        np.put_along_axis(out, np.expand_dims(idx, axis), 1.0, axis=axis)
        return out.astype(np.float32)
    arr = arr / temperature
    arr = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(arr)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return (exp / denom).astype(np.float32)


@contextmanager
def timer() -> Iterator[_Elapsed]:
    """上下文管理器，测量一段代码的墙钟耗时（秒）。

    ::

        with timer() as t:
            do_work()
        print(t.seconds)
    """
    handle = _Elapsed()
    start = time.perf_counter()
    try:
        yield handle
    finally:
        handle.seconds = time.perf_counter() - start


class _Elapsed:
    """:func:`timer` 用来回填耗时的小容器。"""

    __slots__ = ("seconds",)

    def __init__(self) -> None:
        self.seconds = 0.0
