"""内置的控制预设，以及一个可扩展的预设注册表。

预设就是给常用的 :class:`~vocca.control.spec.ControlSpec` 组合起个名字，
让用户直接 ``preset("news_anchor")`` 而不必逐字段填写。
"""

from __future__ import annotations

from ..exceptions import RegistryError
from .spec import ControlSpec

__all__ = ["preset", "register_preset", "available_presets"]

_PRESETS: dict[str, ControlSpec] = {
    "default": ControlSpec(),
    "news_anchor": ControlSpec(emotion="neutral", style="news", intensity=0.4, speed=1.05),
    "storyteller": ControlSpec(emotion="calm", style="narration", intensity=0.7, speed=0.95),
    "excited": ControlSpec(emotion="happy", style="conversational", intensity=0.95, pitch=2.0),
    "sad_poem": ControlSpec(emotion="sad", style="poetry", intensity=0.85, speed=0.9, pitch=-2.0),
    "angry_shout": ControlSpec(emotion="angry", style="shout", intensity=1.0, energy=2.0),
    "gentle_whisper": ControlSpec(
        emotion="calm", style="whisper", intensity=0.3, energy=0.5, speed=0.9
    ),
    "hotline": ControlSpec(emotion="happy", style="customer_service", intensity=0.5),
}


def preset(name: str) -> ControlSpec:
    """按名字取一个预设的 :class:`ControlSpec`。

    找不到时抛 :class:`~vocca.exceptions.RegistryError`，并在报错里列出可选项。
    """
    try:
        return _PRESETS[name]
    except KeyError as exc:
        options = ", ".join(sorted(_PRESETS))
        raise RegistryError(f"未知预设 '{name}'，可选：{options}") from exc


def register_preset(name: str, spec: ControlSpec, *, overwrite: bool = False) -> None:
    """注册一个自定义预设。

    默认禁止覆盖同名预设，除非显式 ``overwrite=True``。
    """
    if not overwrite and name in _PRESETS:
        raise RegistryError(f"预设 '{name}' 已存在（如需覆盖请传 overwrite=True）")
    if not isinstance(spec, ControlSpec):
        raise TypeError("spec 必须是 ControlSpec")
    _PRESETS[name] = spec


def available_presets() -> tuple[str, ...]:
    """返回所有已注册预设名，按字典序排列。"""
    return tuple(sorted(_PRESETS))
