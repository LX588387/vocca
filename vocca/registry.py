"""一个极简的泛型组件注册表。

用来给编解码器（以及未来的其它可插拔组件）按名字注册 / 查找，让用户不改
框架源码就能接入自定义实现::

    from vocca.registry import Registry

    codecs = Registry("codec")
    codecs.register("my", MyCodec)
    cls = codecs.get("my")
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

from .exceptions import RegistryError

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """按名字注册 / 取用一类组件。"""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T, *, overwrite: bool = False) -> T:
        """注册一个组件；默认禁止覆盖同名项。"""
        if not overwrite and name in self._items:
            raise RegistryError(f"{self.kind} '{name}' 已注册（覆盖请传 overwrite=True）")
        self._items[name] = item
        return item

    def decorator(self, name: str) -> Callable[[T], T]:
        """当装饰器用：``@registry.decorator("name")``。"""

        def wrap(item: T) -> T:
            self.register(name, item)
            return item

        return wrap

    def get(self, name: str) -> T:
        """按名字取组件，找不到时抛 :class:`RegistryError`。"""
        try:
            return self._items[name]
        except KeyError as exc:
            options = ", ".join(sorted(self._items)) or "（空）"
            raise RegistryError(f"未知 {self.kind} '{name}'，可选：{options}") from exc

    def names(self) -> tuple[str, ...]:
        """所有已注册名字，按字典序。"""
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)
