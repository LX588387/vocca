"""vocca —— 文本 / 提示驱动的可控声音生成框架。

一条「文本 + 控制条件 → 声学 token → 波形」的自回归管线：控制条件（情感 /
风格 / 音色 / 语速 / 音高 / 能量）经编码后作为前缀注入声学语言模型，配合
分类器无关引导（CFG）让控制更鲜明；解码走 KV cache，并支持**流式**边生成
边出声。核心纯 numpy、零重依赖，torch 为可选后端。

常用入口::

    from vocca import VoiceGenerator, ControlSpec

    gen = VoiceGenerator()
    result = gen.generate("你好，世界", control=ControlSpec(emotion="happy"))
    gen.save(result, "hello.wav")

    # 流式：
    for chunk in gen.stream("[style:news] 晚间新闻现在开始"):
        play(chunk.waveform)
"""

from __future__ import annotations

from .audio import load_wav, log_mel_spectrogram, mel_spectrogram, save_wav
from .codec import Codec, GriffinLim, MockCodec, OscillatorCodec, build_codec, register_codec
from .config import CodecConfig, GenerationConfig, ModelConfig, VoccaConfig
from .control import (
    ControlEncoder,
    ControlSpec,
    ParsedPrompt,
    available_presets,
    parse_prompt,
    preset,
    register_preset,
)
from .exceptions import (
    AudioIOError,
    CodecError,
    ConfigError,
    ControlError,
    GenerationError,
    RegistryError,
    ShapeError,
    VoccaError,
)
from .generate import (
    GenerationResult,
    StreamingGenerator,
    VoiceGenerator,
    generate_tokens,
    iter_tokens,
)
from .guidance import classifier_free_guidance
from .model import AcousticLM, ModelWeights, init_weights, load_weights, save_weights
from .pipeline import load_generator, synthesize, synthesize_to_file
from .sampling import sample_token
from .text import ByteTokenizer, TextTokenizer
from .types import AudioChunk, AudioClip
from .version import __version__

__all__ = [
    "__version__",
    # 生成
    "VoiceGenerator",
    "StreamingGenerator",
    "GenerationResult",
    "iter_tokens",
    "generate_tokens",
    "synthesize",
    "synthesize_to_file",
    "load_generator",
    # 模型
    "AcousticLM",
    "ModelWeights",
    "init_weights",
    "save_weights",
    "load_weights",
    "classifier_free_guidance",
    "sample_token",
    # 控制
    "ControlSpec",
    "ControlEncoder",
    "parse_prompt",
    "ParsedPrompt",
    "preset",
    "register_preset",
    "available_presets",
    # 编解码
    "Codec",
    "OscillatorCodec",
    "MockCodec",
    "GriffinLim",
    "build_codec",
    "register_codec",
    # 文本
    "TextTokenizer",
    "ByteTokenizer",
    # 音频
    "load_wav",
    "save_wav",
    "mel_spectrogram",
    "log_mel_spectrogram",
    # 配置
    "VoccaConfig",
    "ModelConfig",
    "CodecConfig",
    "GenerationConfig",
    # 类型
    "AudioClip",
    "AudioChunk",
    # 异常
    "VoccaError",
    "ConfigError",
    "RegistryError",
    "ShapeError",
    "ControlError",
    "CodecError",
    "GenerationError",
    "AudioIOError",
]
