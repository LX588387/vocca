"""生成子系统：解码循环、高层生成器与流式接口。"""

from __future__ import annotations

from .generator import GenerationResult, VoiceGenerator
from .loop import TokenStream, generate_tokens, iter_tokens
from .streaming import StreamingGenerator

__all__ = [
    "VoiceGenerator",
    "GenerationResult",
    "StreamingGenerator",
    "iter_tokens",
    "generate_tokens",
    "TokenStream",
]
