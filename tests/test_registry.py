from __future__ import annotations

import pytest

from vocca.exceptions import RegistryError
from vocca.registry import Registry


def test_register_and_get():
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    assert reg.get("a") == 1
    assert "a" in reg
    assert len(reg) == 1


def test_duplicate_raises():
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(RegistryError):
        reg.register("a", 2)
    reg.register("a", 2, overwrite=True)
    assert reg.get("a") == 2


def test_get_unknown_raises():
    reg: Registry[int] = Registry("thing")
    with pytest.raises(RegistryError):
        reg.get("missing")


def test_names_sorted():
    reg: Registry[int] = Registry("thing")
    reg.register("b", 2)
    reg.register("a", 1)
    assert reg.names() == ("a", "b")


def test_decorator():
    reg: Registry[type] = Registry("cls")

    @reg.decorator("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo
