"""编解码器子系统：声学 token ↔ 波形。"""

from __future__ import annotations

from ..config import CodecConfig
from ..registry import Registry
from .base import Codec
from .griffin import GriffinLim
from .mock import MockCodec
from .oscillator import OscillatorCodec

__all__ = [
    "Codec",
    "OscillatorCodec",
    "MockCodec",
    "GriffinLim",
    "build_codec",
    "register_codec",
    "CODEC_REGISTRY",
]

CODEC_REGISTRY: Registry[type[Codec]] = Registry("codec")
CODEC_REGISTRY.register("oscillator", OscillatorCodec)
CODEC_REGISTRY.register("mock", MockCodec)


def register_codec(name: str, cls: type[Codec], *, overwrite: bool = False) -> None:
    """注册一个自定义编解码器类，之后可用其名字通过 :func:`build_codec` 构造。"""
    CODEC_REGISTRY.register(name, cls, overwrite=overwrite)


def build_codec(config: CodecConfig) -> Codec:
    """按 :class:`~vocca.config.CodecConfig` 构造编解码器。"""
    cls = CODEC_REGISTRY.get(config.name)
    return cls(
        sample_rate=config.sample_rate,
        frame_size=config.frame_size,
        codebook_size=config.codebook_size,
    )
