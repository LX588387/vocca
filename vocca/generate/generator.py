"""高层生成入口：:class:`VoiceGenerator`。

把文本前端、声学语言模型与编解码器串成一条「文本 / 提示 → 波形」的管线，
既支持整段合成（:meth:`generate`），也支持流式（:meth:`stream`）。控制条件
可以显式传 :class:`~vocca.control.spec.ControlSpec`，也可以写在文本里的内联
标签（``[emotion:happy]`` 等），后者优先级更高。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..codec import Codec, build_codec
from ..config import VoccaConfig
from ..control import ControlSpec, parse_prompt
from ..model import AcousticLM
from ..model.weights import ModelWeights
from ..text import ByteTokenizer
from ..types import AudioChunk, AudioClip, PathLike, TokenSequence
from .loop import generate_tokens
from .streaming import StreamingGenerator

__all__ = ["VoiceGenerator", "GenerationResult"]


@dataclass
class GenerationResult:
    """一次整段生成的结果。

    Attributes:
        audio: 合成出的音频片段。
        tokens: 生成的声学 token 序列。
        control: 实际生效的控制条件（合并内联标签后的结果）。
        text: 去掉控制标签后的纯文本。
        stopped_by_eos: 是否因采样到 ``<audio_eos>`` 而提前停止。
    """

    audio: AudioClip
    tokens: TokenSequence
    control: ControlSpec
    text: str
    stopped_by_eos: bool

    @property
    def duration(self) -> float:
        return self.audio.duration


class VoiceGenerator:
    """文本 / 提示驱动的可控声音生成器。

    Args:
        config: 完整管线配置；``None`` 时用默认配置（小模型，便于快速试跑）。
        weights: 声学模型权重；``None`` 时按配置随机初始化。
        codec: 自定义编解码器；``None`` 时按 ``config.codec`` 构造。
    """

    def __init__(
        self,
        config: VoccaConfig | None = None,
        weights: ModelWeights | None = None,
        codec: Codec | None = None,
    ) -> None:
        self.config = (config or VoccaConfig()).validate()
        self.tokenizer = ByteTokenizer()
        self.lm = AcousticLM(self.config.model, weights)
        self.codec = codec or build_codec(self.config.codec)
        if self.tokenizer.vocab_size > self.config.model.text_vocab:
            raise ValueError("文本词表大于模型 text_vocab，无法编码")

    # -- 文本 / 控制预处理 ----------------------------------------------
    def _prepare(self, prompt: str, control: ControlSpec | None) -> tuple[str, ControlSpec]:
        parsed = parse_prompt(prompt, base=control or ControlSpec())
        return parsed.text, parsed.control

    def _encode_text(self, text: str) -> TokenSequence:
        return self.tokenizer.encode(text)

    # -- 整段生成 --------------------------------------------------------
    def generate(self, prompt: str, control: ControlSpec | None = None) -> GenerationResult:
        """把提示词合成为一段音频。"""
        text, effective = self._prepare(prompt, control)
        text_ids = self._encode_text(text)
        stream = generate_tokens(self.lm, text_ids, effective, self.config.generation)
        wav = self.codec.decode(stream.tokens)
        clip = AudioClip(waveform=wav, sample_rate=self.config.codec.sample_rate)
        return GenerationResult(
            audio=clip,
            tokens=stream.tokens,
            control=effective,
            text=text,
            stopped_by_eos=stream.stopped_by_eos,
        )

    # -- 流式生成 --------------------------------------------------------
    def stream(
        self,
        prompt: str,
        control: ControlSpec | None = None,
        chunk_tokens: int = 16,
    ) -> Iterator[AudioChunk]:
        """流式合成：逐块产出 :class:`~vocca.types.AudioChunk`。"""
        text, effective = self._prepare(prompt, control)
        text_ids = self._encode_text(text)
        streamer = StreamingGenerator(self.lm, self.codec, chunk_tokens=chunk_tokens)
        yield from streamer.stream(text_ids, effective, self.config.generation)

    # -- 便捷方法 --------------------------------------------------------
    def save(self, result: GenerationResult, path: PathLike) -> None:
        """把生成结果写成 WAV 文件。"""
        from ..audio import save_wav

        save_wav(path, result.audio.waveform, result.audio.sample_rate)

    @property
    def sample_rate(self) -> int:
        return self.config.codec.sample_rate
