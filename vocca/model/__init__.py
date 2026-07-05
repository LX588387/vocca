"""声学语言模型子系统：层、注意力、解码栈、权重与自回归接口。"""

from __future__ import annotations

from .attention import AttentionWeights, LayerCache, self_attention
from .layers import apply_rope, gelu, linear, rms_norm, rope_cache, silu
from .lm import AcousticLM, new_cache
from .transformer import decode, transformer_block
from .weights import (
    BlockWeights,
    ModelWeights,
    init_weights,
    load_weights,
    save_weights,
)

__all__ = [
    "AcousticLM",
    "new_cache",
    "ModelWeights",
    "BlockWeights",
    "init_weights",
    "save_weights",
    "load_weights",
    "decode",
    "transformer_block",
    "self_attention",
    "AttentionWeights",
    "LayerCache",
    "rms_norm",
    "gelu",
    "silu",
    "linear",
    "rope_cache",
    "apply_rope",
]
