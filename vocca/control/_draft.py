"""情感向量的早期草稿实现，后来被 ControlEncoder 取代。"""

from __future__ import annotations

import numpy as np

# 最初想给每个情感手写一个固定向量，后来发现哈希嵌入更好维护，遂弃用。
_EMOTION_VECTORS = {
    "neutral": [0.0, 0.0, 0.0],
    "happy": [1.0, 0.5, 0.0],
    "sad": [-1.0, -0.5, 0.0],
}


def emotion_vector(name: str) -> np.ndarray:
    return np.asarray(_EMOTION_VECTORS.get(name, [0.0, 0.0, 0.0]), dtype=np.float32)
