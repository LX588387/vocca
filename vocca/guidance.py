"""分类器无关引导（Classifier-Free Guidance, CFG）。

可控生成的关键一环：同时估计「给定控制条件」与「空条件」两套 logits，
再沿二者之差把分布往条件方向外推::

    logits = uncond + scale * (cond - uncond)

``scale == 1`` 时等价于纯条件采样；``scale > 1`` 放大控制信号（情感 / 风格
更鲜明），代价是多样性下降；``scale == 0`` 则完全忽略条件。
"""

from __future__ import annotations

import numpy as np

from .types import Logits

__all__ = ["classifier_free_guidance"]


def classifier_free_guidance(cond: Logits, uncond: Logits, scale: float) -> Logits:
    """组合条件 / 无条件 logits。

    Args:
        cond: 给定控制条件下的 logits。
        uncond: 空条件（``ControlSpec=None``）下的 logits。
        scale: 引导强度，见模块文档。

    Returns:
        外推后的 logits，形状与输入一致。
    """
    cond = np.asarray(cond, dtype=np.float32)
    uncond = np.asarray(uncond, dtype=np.float32)
    if cond.shape != uncond.shape:
        raise ValueError("cond 与 uncond 形状必须一致")
    if scale == 1.0:
        return cond
    return (uncond + scale * (cond - uncond)).astype(np.float32)
