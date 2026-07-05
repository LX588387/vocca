"""自回归解码循环：把文本 + 控制条件变成一串声学 token。

这是生成的核心，被 :class:`~vocca.generate.generator.VoiceGenerator` 与
:class:`~vocca.generate.streaming.StreamingGenerator` 共用。要点：

* **KV cache**：``prefill`` 建好前缀 + 文本的缓存，之后每步只前向一个 token；
* **分类器无关引导（CFG）**：``guidance_scale != 1`` 时并行维护「条件」与
  「空条件」两套 cache，逐步用 :func:`~vocca.guidance.classifier_free_guidance`
  组合 logits；
* **采样**：温度 / top-k / top-p / 重复惩罚，种子固定即可复现；
* 以生成器（``yield``）形式产出 token，天然支持流式——调用方要整段就
  ``list(...)``，要流式就分块消费。
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np

from ..config import GenerationConfig
from ..control import ControlSpec
from ..guidance import classifier_free_guidance
from ..model import AcousticLM
from ..sampling import sample_token
from ..types import Logits, TokenSequence
from ..utils import rng_from_seed

__all__ = ["iter_tokens", "generate_tokens", "TokenStream"]


@dataclass
class TokenStream:
    """整段生成的结果。"""

    tokens: TokenSequence
    stopped_by_eos: bool
    steps: int = field(default=0)


def _combined_logits(
    lm: AcousticLM,
    text_ids: TokenSequence,
    control: ControlSpec | None,
    cfg: GenerationConfig,
    cond_cache: list,
    uncond_cache: list | None,
    prev_token: int | None,
) -> Logits:
    """取当前步的（可能经过 CFG 的）logits。

    ``prev_token is None`` 表示这是首步，走 ``prefill``；否则走 ``step``。
    """
    if prev_token is None:
        cond = lm.prefill(text_ids, cond_cache, control)
        if uncond_cache is not None:
            uncond = lm.prefill(text_ids, uncond_cache, None)
            return classifier_free_guidance(cond, uncond, cfg.guidance_scale)
        return cond
    cond = lm.step(prev_token, cond_cache)
    if uncond_cache is not None:
        uncond = lm.step(prev_token, uncond_cache)
        return classifier_free_guidance(cond, uncond, cfg.guidance_scale)
    return cond


def iter_tokens(
    lm: AcousticLM,
    text_ids: TokenSequence,
    control: ControlSpec | None = None,
    cfg: GenerationConfig | None = None,
) -> Generator[int, None, TokenStream]:
    """逐个产出声学 token（不含特殊 token），直到 EOS 或达到步数上限。

    生成器结束时（``StopIteration``）会返回一个 :class:`TokenStream`，其
    ``tokens`` 字段为空、但 ``steps`` / ``stopped_by_eos`` 记录了本次解码的
    真实统计，供 :func:`generate_tokens` 手动迭代时捕获。
    """
    cfg = cfg or GenerationConfig()
    cfg.validate()
    rng = rng_from_seed(cfg.seed)
    use_cfg = cfg.guidance_scale != 1.0 and control is not None
    cond_cache = lm.new_cache()
    uncond_cache = lm.new_cache() if use_cfg else None

    logits = _combined_logits(lm, text_ids, control, cfg, cond_cache, uncond_cache, None)
    history: list[int] = []
    eos = lm.config.audio_eos
    codebook = lm.config.codebook_size
    steps = 0
    stopped_by_eos = False

    for _ in range(cfg.max_new_tokens):
        token = sample_token(
            logits,
            rng,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            history=np.asarray(history, dtype=np.int64),
            repetition_penalty=cfg.repetition_penalty,
        )
        if cfg.stop_on_eos and token == eos:
            stopped_by_eos = True
            break
        steps += 1
        if 0 <= token < codebook:
            yield token
        history.append(token)
        logits = _combined_logits(lm, text_ids, control, cfg, cond_cache, uncond_cache, token)

    return TokenStream(
        tokens=np.zeros(0, dtype=np.int64), stopped_by_eos=stopped_by_eos, steps=steps
    )


def generate_tokens(
    lm: AcousticLM,
    text_ids: TokenSequence,
    control: ControlSpec | None = None,
    cfg: GenerationConfig | None = None,
) -> TokenStream:
    """跑完整个解码循环，收集所有声学 token。"""
    tokens: list[int] = []
    gen = iter_tokens(lm, text_ids, control, cfg)
    summary = TokenStream(tokens=np.zeros(0, dtype=np.int64), stopped_by_eos=False)
    try:
        while True:
            tokens.append(next(gen))
    except StopIteration as stop:
        if isinstance(stop.value, TokenStream):
            summary = stop.value
    return TokenStream(
        tokens=np.asarray(tokens, dtype=np.int64),
        stopped_by_eos=summary.stopped_by_eos,
        steps=summary.steps,
    )
