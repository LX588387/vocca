"""控制条件的声明式描述。

:class:`ControlSpec` 用一组人类可读的字段（情感、风格、音色、语速、
音高、能量）描述「想要什么样的声音」，再由 :mod:`vocca.control.encoder`
编码成送进声学模型的条件向量。字段都做了范围校验，非法值会在构造时
立即抛 :class:`ControlError`，而不是等到生成阶段才炸。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..exceptions import ControlError

__all__ = ["ControlSpec", "EMOTIONS", "STYLES"]

# 支持的离散情感标签。保持精简，覆盖常见语音表达。
EMOTIONS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "calm",
)

# 支持的说话 / 音频风格标签。
STYLES = (
    "default",
    "news",
    "conversational",
    "narration",
    "customer_service",
    "poetry",
    "whisper",
    "shout",
)


@dataclass(frozen=True)
class ControlSpec:
    """一次生成的可控属性。

    Attributes:
        emotion: 情感标签，取值见 :data:`EMOTIONS`。
        style: 风格标签，取值见 :data:`STYLES`。
        speaker: 音色 / 说话人标识；任意字符串，编码时哈希成音色向量。
        intensity: 情感强度，``0.0``（几乎中性）到 ``1.0``（最强）。
        speed: 语速倍率，``1.0`` 为常速，范围 ``[0.5, 2.0]``。
        pitch: 音高偏移（半音），范围 ``[-12, 12]``。
        energy: 整体能量 / 响度，``1.0`` 为基准，范围 ``[0.25, 4.0]``。
        extra: 透传给下游编码器的自定义键值，不做校验。
    """

    emotion: str = "neutral"
    style: str = "default"
    speaker: str = "default"
    intensity: float = 0.6
    speed: float = 1.0
    pitch: float = 0.0
    energy: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.emotion not in EMOTIONS:
            raise ControlError(f"未知情感 '{self.emotion}'，可选：{', '.join(EMOTIONS)}")
        if self.style not in STYLES:
            raise ControlError(f"未知风格 '{self.style}'，可选：{', '.join(STYLES)}")
        _check_range("intensity", self.intensity, 0.0, 1.0)
        _check_range("speed", self.speed, 0.5, 2.0)
        _check_range("pitch", self.pitch, -12.0, 12.0)
        _check_range("energy", self.energy, 0.25, 4.0)

    def with_(self, **changes: Any) -> ControlSpec:
        """返回一个替换了部分字段的新 :class:`ControlSpec`（不可变对象）。"""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """序列化成普通 dict，便于写 YAML / JSON。"""
        return {
            "emotion": self.emotion,
            "style": self.style,
            "speaker": self.speaker,
            "intensity": self.intensity,
            "speed": self.speed,
            "pitch": self.pitch,
            "energy": self.energy,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlSpec:
        """:meth:`to_dict` 的逆操作，忽略未知键。"""
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not isinstance(value, (int, float)):
        raise ControlError(f"{name} 必须是数值，收到 {type(value).__name__}")
    if not lo <= float(value) <= hi:
        raise ControlError(f"{name}={value} 超出范围 [{lo}, {hi}]")
