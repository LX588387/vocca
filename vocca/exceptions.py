"""vocca 使用的异常类型。

对外抛出的异常都继承自 :class:`VoccaError`，调用方既可以用一个
``except VoccaError`` 兜住框架内部的全部错误，也能按需区分具体子类。
"""

from __future__ import annotations


class VoccaError(Exception):
    """vocca 所有异常的基类。"""


class ConfigError(VoccaError):
    """配置不合法或缺失必填字段时抛出。"""


class RegistryError(VoccaError):
    """在注册表里找不到组件，或重复注册同名组件时抛出。"""


class ShapeError(VoccaError):
    """张量 / 数组形状不符合预期时抛出。

    自回归解码对形状很敏感（KV cache、条件前缀拼接），单独拎出来能给出
    更友好的报错。
    """


class ControlError(VoccaError):
    """控制条件（情感 / 风格 / 音色等）不合法时抛出。"""


class CodecError(VoccaError):
    """声码器 / 编解码器在 token 与波形之间转换失败时抛出。"""


class GenerationError(VoccaError):
    """自回归生成过程中出错时抛出（例如超出最大步数仍未收敛）。"""


class AudioIOError(VoccaError):
    """读写音频文件失败时抛出。"""
