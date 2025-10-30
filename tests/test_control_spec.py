from __future__ import annotations

import pytest

from vocca.control import ControlSpec
from vocca.exceptions import ControlError


def test_default_spec_valid():
    spec = ControlSpec()
    assert spec.emotion == "neutral"
    assert spec.style == "default"


def test_invalid_emotion():
    with pytest.raises(ControlError):
        ControlSpec(emotion="ecstatic")


def test_invalid_style():
    with pytest.raises(ControlError):
        ControlSpec(style="rap")


@pytest.mark.parametrize(
    "field,value",
    [("intensity", 1.5), ("speed", 3.0), ("pitch", 20.0), ("energy", 10.0)],
)
def test_out_of_range(field, value):
    with pytest.raises(ControlError):
        ControlSpec(**{field: value})


def test_non_numeric_field():
    with pytest.raises(ControlError):
        ControlSpec(speed="fast")  # type: ignore[arg-type]


def test_with_returns_new_object():
    a = ControlSpec(emotion="happy")
    b = a.with_(emotion="sad")
    assert a.emotion == "happy"
    assert b.emotion == "sad"


def test_dict_roundtrip():
    spec = ControlSpec(emotion="angry", style="shout", intensity=0.9, extra={"k": 1})
    data = spec.to_dict()
    restored = ControlSpec.from_dict(data)
    assert restored == spec


def test_from_dict_ignores_unknown_keys():
    spec = ControlSpec.from_dict({"emotion": "calm", "bogus": 42})
    assert spec.emotion == "calm"
