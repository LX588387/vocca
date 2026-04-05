"""vocca —— 文本 / 提示驱动的可控声音生成框架。

核心链路：文本 + 控制条件 → 声学 token → 波形。此文件在开发中逐步补全导出。
"""

from __future__ import annotations

from .config import CodecConfig, GenerationConfig, ModelConfig, VoccaConfig
from .control import ControlSpec, parse_prompt, preset
from .exceptions import (
    ConfigError,
    ControlError,
    RegistryError,
    ShapeError,
    VoccaError,
)
from .types import AudioChunk, AudioClip
from .version import __version__

__all__ = [
    "__version__",
    "ControlSpec",
    "parse_prompt",
    "preset",
    "VoccaConfig",
    "ModelConfig",
    "CodecConfig",
    "GenerationConfig",
    "AudioClip",
    "AudioChunk",
    "VoccaError",
    "ConfigError",
    "ControlError",
    "RegistryError",
    "ShapeError",
]
