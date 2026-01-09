"""声学语言模型：把文本 + 控制条件映射成声学 token 的自回归分布。

序列布局（从左到右，因果注意力）::

    [ 控制前缀 c_1 … c_p ] [ 文本 token t_1 … t_m ] [ 声学 token a_1 … a_n ]

控制前缀由 :class:`~vocca.control.encoder.ControlEncoder` 生成并直接作为隐藏
状态注入；文本 / 声学 token 各查一张嵌入表。输出头只预测**声学**词表，因为
生成阶段我们只关心 a_i。模型对外提供三个入口：

* :meth:`forward`：整段前向，返回每个声学位置的 logits（训练 / 校验用）；
* :meth:`prefill`：喂入前缀 + 文本 + ``<audio_bos>``，建好 cache，返回首个声学
  token 的 logits；
* :meth:`step`：喂入上一步采样出的声学 token，增量前向，返回下一步 logits。

:meth:`prefill` / :meth:`step` 走 KV cache，与 :meth:`forward` 数值一致。
"""

from __future__ import annotations

import numpy as np

from ..config import ModelConfig
from ..control import ControlEncoder, ControlSpec
from ..exceptions import ShapeError
from ..types import Hidden, Logits, TokenSequence
from .attention import LayerCache
from .layers import linear, rope_cache
from .transformer import decode
from .weights import ModelWeights, init_weights

__all__ = ["AcousticLM", "new_cache"]

Cache = list  # 逐层 LayerCache 的别名，语义上等价于 KV cache


class AcousticLM:
    """文本 / 提示驱动的声学 token 生成模型（纯 numpy 参考实现）。

    Args:
        config: 模型结构配置。
        weights: 预置权重；``None`` 时按配置随机初始化。
        control_encoder: 控制条件编码器；``None`` 时按配置构造。
    """

    def __init__(
        self,
        config: ModelConfig,
        weights: ModelWeights | None = None,
        control_encoder: ControlEncoder | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.weights = weights if weights is not None else init_weights(config)
        self.control = control_encoder or ControlEncoder(
            config.dim, seed=config.seed, n_prefix=max(1, config.control_prefix)
        )
        if self.control.dim != config.dim:
            raise ShapeError("control encoder 维度须与模型 dim 一致")
        cos, sin = rope_cache(config.max_seq_len, config.head_dim, config.rope_base)
        self._cos, self._sin = cos, sin

    # -- 缓存 ------------------------------------------------------------
    def new_cache(self) -> Cache:
        """新建一套逐层空 KV cache。"""
        return new_cache(self.config)

    # -- 嵌入 ------------------------------------------------------------
    def _control_prefix(self, control: ControlSpec | None) -> Hidden:
        if self.config.control_prefix == 0:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        return self.control.encode(control)

    def _embed_text(self, text_ids: TokenSequence) -> Hidden:
        ids = np.asarray(text_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.config.text_vocab):
            raise ShapeError("文本 token id 超出词表范围")
        return self.weights.text_embed[ids]

    def _embed_audio(self, audio_ids: TokenSequence) -> Hidden:
        ids = np.asarray(audio_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.config.audio_vocab):
            raise ShapeError("声学 token id 超出词表范围")
        return self.weights.audio_embed[ids]

    def _rope_slice(self, start: int, length: int) -> tuple[Hidden, Hidden]:
        end = start + length
        if end > self.config.max_seq_len:
            raise ShapeError(f"序列长度 {end} 超过 max_seq_len={self.config.max_seq_len}")
        return self._cos[start:end], self._sin[start:end]

    def _head(self, hidden: Hidden) -> Logits:
        return linear(hidden, self.weights.head)

    # -- 前向 ------------------------------------------------------------
    def forward(
        self,
        text_ids: TokenSequence,
        audio_ids: TokenSequence,
        control: ControlSpec | None = None,
    ) -> Logits:
        """整段前向，返回**声学段每个位置**的 logits ``(len(audio_ids), audio_vocab)``。

        主要用于训练与一致性校验；生成请用 :meth:`prefill` / :meth:`step`。
        """
        prefix = self._control_prefix(control)
        text = self._embed_text(text_ids)
        audio = self._embed_audio(audio_ids)
        hidden = np.concatenate([prefix, text, audio], axis=0)
        cos, sin = self._rope_slice(0, hidden.shape[0])
        out = decode(hidden, self.weights, cos, sin, caches=None)
        logits = self._head(out)
        return logits[-audio.shape[0] :] if audio.shape[0] else logits[0:0]

    def prefill(
        self,
        text_ids: TokenSequence,
        cache: Cache,
        control: ControlSpec | None = None,
    ) -> Logits:
        """建 cache 并返回首个声学 token 的 logits ``(audio_vocab,)``。"""
        prefix = self._control_prefix(control)
        text = self._embed_text(text_ids)
        bos = self._embed_audio(np.asarray([self.config.audio_bos], dtype=np.int64))
        hidden = np.concatenate([prefix, text, bos], axis=0)
        cos, sin = self._rope_slice(0, hidden.shape[0])
        out = decode(hidden, self.weights, cos, sin, caches=cache)
        return self._head(out[-1:])[0]

    def step(self, audio_id: int, cache: Cache) -> Logits:
        """喂入一个声学 token，增量前向，返回下一步 logits ``(audio_vocab,)``。"""
        start = cache[0].length
        emb = self._embed_audio(np.asarray([audio_id], dtype=np.int64))
        cos, sin = self._rope_slice(start, 1)
        out = decode(emb, self.weights, cos, sin, caches=cache)
        return self._head(out[-1:])[0]

    # -- 便捷属性 --------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return self.weights.num_parameters()


def new_cache(config: ModelConfig) -> Cache:
    """按配置新建逐层空 KV cache。"""
    return [LayerCache.empty(config.n_kv_heads, config.head_dim) for _ in range(config.n_layers)]
