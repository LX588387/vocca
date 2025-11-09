from __future__ import annotations

import pytest

from vocca.control import ControlSpec, available_presets, preset, register_preset
from vocca.exceptions import RegistryError


def test_builtin_presets_exist():
    names = available_presets()
    assert "news_anchor" in names
    assert "default" in names
    assert names == tuple(sorted(names))


def test_preset_returns_spec():
    spec = preset("excited")
    assert isinstance(spec, ControlSpec)
    assert spec.emotion == "happy"


def test_unknown_preset_raises():
    with pytest.raises(RegistryError):
        preset("does_not_exist")


def test_register_and_use():
    register_preset("robot_test", ControlSpec(style="news", speaker="robot"))
    assert "robot_test" in available_presets()
    assert preset("robot_test").speaker == "robot"


def test_register_duplicate_raises():
    register_preset("dup_test", ControlSpec(), overwrite=False)
    with pytest.raises(RegistryError):
        register_preset("dup_test", ControlSpec())
    register_preset("dup_test", ControlSpec(emotion="sad"), overwrite=True)
    assert preset("dup_test").emotion == "sad"


def test_register_wrong_type():
    with pytest.raises(TypeError):
        register_preset("bad_type", object())  # type: ignore[arg-type]
