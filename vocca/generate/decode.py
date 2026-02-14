"""自回归解码：把文本 + 控制条件变成一串声学 token。

早期版本，只有贪心 / 温度采样，还没有 KV cache 与分类器无关引导。
"""

from __future__ import annotations

import numpy as np

from ..config import GenerationConfig
from ..control import ControlSpec
from ..model import AcousticLM
from ..sampling import sample_token
from ..types import TokenSequence
from ..utils import rng_from_seed


def generate_tokens(
    lm: AcousticLM,
    text_ids: TokenSequence,
    control: ControlSpec | None = None,
    cfg: GenerationConfig | None = None,
) -> TokenSequence:
    """朴素解码循环（无 cache）：每步都对全序列重新前向。"""
    cfg = cfg or GenerationConfig()
    rng = rng_from_seed(cfg.seed)
    audio = [lm.config.audio_bos]
    out: list[int] = []
    for _ in range(cfg.max_new_tokens):
        logits = lm.forward(text_ids, np.asarray(audio, dtype=np.int64), control)[-1]
        token = sample_token(logits, rng, temperature=cfg.temperature)
        if token == lm.config.audio_eos:
            break
        audio.append(token)
        if 0 <= token < lm.config.codebook_size:
            out.append(token)
    return np.asarray(out, dtype=np.int64)
