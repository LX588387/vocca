"""便捷入口：一行完成「文本 → 音频文件」。

:class:`~vocca.generate.generator.VoiceGenerator` 已经是完整管线，这里再提供
几个更省事的函数，覆盖最常见的一次性调用场景。
"""

from __future__ import annotations

from .codec import Codec
from .config import VoccaConfig
from .control import ControlSpec, preset
from .generate import GenerationResult, VoiceGenerator
from .types import PathLike

__all__ = ["synthesize", "synthesize_to_file", "load_generator"]


def load_generator(
    config: VoccaConfig | PathLike | None = None, codec: Codec | None = None
) -> VoiceGenerator:
    """构造一个 :class:`VoiceGenerator`。

    ``config`` 可以是 :class:`VoccaConfig`、YAML 路径或 ``None``（默认配置）。
    """
    if config is None:
        cfg = VoccaConfig()
    elif isinstance(config, VoccaConfig):
        cfg = config
    else:
        cfg = VoccaConfig.from_yaml(config)
    return VoiceGenerator(cfg, codec=codec)


def synthesize(
    prompt: str,
    control: ControlSpec | str | None = None,
    generator: VoiceGenerator | None = None,
) -> GenerationResult:
    """把提示词合成为音频并返回 :class:`GenerationResult`。

    ``control`` 传字符串时按预设名解析（见 :func:`vocca.control.preset`）。
    """
    gen = generator or VoiceGenerator()
    spec = preset(control) if isinstance(control, str) else control
    return gen.generate(prompt, control=spec)


def synthesize_to_file(
    prompt: str,
    path: PathLike,
    control: ControlSpec | str | None = None,
    generator: VoiceGenerator | None = None,
) -> GenerationResult:
    """合成并直接写成 WAV，返回结果对象。"""
    gen = generator or VoiceGenerator()
    result = synthesize(prompt, control=control, generator=gen)
    gen.save(result, path)
    return result
