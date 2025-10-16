"""控制条件子系统：情感 / 风格 / 音色等可控属性。"""

from __future__ import annotations

from .encoder import ControlEncoder
from .presets import available_presets, preset, register_preset
from .prompt import ParsedPrompt, parse_prompt
from .spec import EMOTIONS, STYLES, ControlSpec

__all__ = [
    "ControlSpec",
    "EMOTIONS",
    "STYLES",
    "ControlEncoder",
    "parse_prompt",
    "ParsedPrompt",
    "preset",
    "register_preset",
    "available_presets",
]
