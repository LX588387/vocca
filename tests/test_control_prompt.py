from __future__ import annotations

import pytest

from vocca.control import ControlSpec, parse_prompt
from vocca.exceptions import ControlError


def test_no_tags_passthrough():
    parsed = parse_prompt("你好世界")
    assert parsed.text == "你好世界"
    assert parsed.control == ControlSpec()
    assert parsed.tags == ()


def test_single_tag_strips_and_applies():
    parsed = parse_prompt("[emotion:happy] 早上好")
    assert parsed.text == "早上好"
    assert parsed.control.emotion == "happy"


def test_numeric_tags():
    parsed = parse_prompt("[speed:1.2][pitch:-2] 测试")
    assert parsed.control.speed == 1.2
    assert parsed.control.pitch == -2.0


def test_multiple_tags_in_middle():
    parsed = parse_prompt("前[emotion:sad]中[style:poetry]后")
    assert parsed.text == "前中后"
    assert parsed.control.emotion == "sad"
    assert parsed.control.style == "poetry"


def test_base_control_is_overridden():
    base = ControlSpec(emotion="angry", speed=1.5)
    parsed = parse_prompt("[emotion:happy] hi", base=base)
    assert parsed.control.emotion == "happy"
    assert parsed.control.speed == 1.5  # 未被标签覆盖，保留 base


def test_unknown_tag_raises():
    with pytest.raises(ControlError):
        parse_prompt("[tempo:fast] x")


def test_bad_numeric_raises():
    with pytest.raises(ControlError):
        parse_prompt("[speed:quick] x")


def test_tags_recorded():
    parsed = parse_prompt("[emotion:calm][speed:0.8] x")
    assert ("emotion", "calm") in parsed.tags
    assert ("speed", "0.8") in parsed.tags
