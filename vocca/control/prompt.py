"""提示词解析：从文本里抽取内联控制标签。

支持在文本中用 ``[key:value]`` 形式内联指定控制条件，例如::

    [emotion:happy][speed:1.2] 今天天气真好[pitch:2] 我们出去走走吧

解析后得到去掉标签的纯文本，以及一个在给定基线上叠加了这些标签的
:class:`~vocca.control.spec.ControlSpec`。标签就近生效——但为了保持接口
简单，这里采用「整句统一」的语义：所有标签合并后作用于整段文本。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..exceptions import ControlError
from .spec import ControlSpec

__all__ = ["parse_prompt", "ParsedPrompt"]

# 匹配 [key:value]，key 为字母，value 允许字母数字、点、负号、下划线。
_TAG = re.compile(r"\[([a-zA-Z_]+)\s*:\s*([-\w.]+)\]")

# 数值型字段名 → 类型转换器。
_NUMERIC = {
    "intensity": float,
    "speed": float,
    "pitch": float,
    "energy": float,
}
_STRING = {"emotion", "style", "speaker"}


@dataclass(frozen=True)
class ParsedPrompt:
    """:func:`parse_prompt` 的结果。

    Attributes:
        text: 去掉控制标签后的纯文本。
        control: 合并了内联标签的控制条件。
        tags: 解析到的原始 ``(key, value)`` 列表，便于调试。
    """

    text: str
    control: ControlSpec
    tags: tuple[tuple[str, str], ...]


def parse_prompt(prompt: str, base: ControlSpec | None = None) -> ParsedPrompt:
    """解析带内联控制标签的提示词。

    Args:
        prompt: 可能包含 ``[key:value]`` 标签的文本。
        base: 基线控制条件；标签会在其之上覆盖。默认为 :class:`ControlSpec` 缺省值。

    Returns:
        :class:`ParsedPrompt`。

    Raises:
        ControlError: 出现未知标签键，或数值无法解析。
    """
    base = base or ControlSpec()
    tags: list[tuple[str, str]] = []
    overrides: dict[str, object] = {}

    for match in _TAG.finditer(prompt):
        key, raw = match.group(1).lower(), match.group(2)
        tags.append((key, raw))
        if key in _NUMERIC:
            try:
                overrides[key] = _NUMERIC[key](raw)
            except ValueError as exc:
                raise ControlError(f"标签 [{key}:{raw}] 的值不是合法数值") from exc
        elif key in _STRING:
            overrides[key] = raw
        else:
            raise ControlError(f"未知控制标签 '{key}'")

    text = _TAG.sub("", prompt).strip()
    # 合并可能触发 ControlSpec 的范围校验，非法组合会在此抛错。
    control = base.with_(**overrides) if overrides else base
    return ParsedPrompt(text=text, control=control, tags=tuple(tags))
