"""配置对象与 YAML 读写。

用普通 dataclass 描述模型 / 编解码器 / 生成三块配置，再由
:class:`VoccaConfig` 打包。刻意不引入 hydra 之类的重框架；
``from_yaml`` / ``to_yaml`` 负责磁盘与对象之间的转换。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError
from .types import PathLike

__all__ = [
    "ModelConfig",
    "CodecConfig",
    "GenerationConfig",
    "VoccaConfig",
]


@dataclass
class ModelConfig:
    """声学语言模型的结构超参。

    默认值刻意偏小，让纯 numpy 的参考实现在 CI 上也能秒级跑完；真实
    权重可通过 :func:`vocca.model.weights.load_weights` 覆盖。
    """

    text_vocab: int = 256
    codebook_size: int = 256
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    ffn_mult: float = 2.6667
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    control_prefix: int = 1
    seed: int = 0

    @property
    def audio_vocab(self) -> int:
        """声学词表 = 码本大小 + 2 个特殊 token（BOS / EOS）。"""
        return self.codebook_size + 2

    @property
    def audio_bos(self) -> int:
        return self.codebook_size

    @property
    def audio_eos(self) -> int:
        return self.codebook_size + 1

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def validate(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ConfigError(f"dim({self.dim}) 必须能被 n_heads({self.n_heads}) 整除")
        if self.n_heads % self.n_kv_heads != 0:
            raise ConfigError(
                f"n_heads({self.n_heads}) 必须是 n_kv_heads({self.n_kv_heads}) 的整数倍"
            )
        for name in ("text_vocab", "codebook_size", "dim", "n_layers", "n_heads"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} 必须为正整数")
        if self.control_prefix < 0:
            raise ConfigError("control_prefix 不能为负")


@dataclass
class CodecConfig:
    """波形 ↔ 声学 token 的编解码器配置。"""

    name: str = "oscillator"
    sample_rate: int = 24000
    frame_size: int = 320
    codebook_size: int = 256
    options: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.sample_rate <= 0 or self.frame_size <= 0:
            raise ConfigError("sample_rate 与 frame_size 必须为正")

    @property
    def frame_rate(self) -> float:
        """每秒多少个 token 帧。"""
        return self.sample_rate / self.frame_size


@dataclass
class GenerationConfig:
    """自回归生成 / 采样 / 引导的运行时配置。"""

    max_new_tokens: int = 256
    temperature: float = 0.9
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    guidance_scale: float = 1.5
    seed: int = 0
    stop_on_eos: bool = True

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ConfigError("max_new_tokens 必须为正")
        if self.temperature < 0:
            raise ConfigError("temperature 不能为负")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigError("top_p 必须落在 (0, 1]")
        if self.top_k < 0:
            raise ConfigError("top_k 不能为负")
        if self.repetition_penalty <= 0:
            raise ConfigError("repetition_penalty 必须为正")


@dataclass
class VoccaConfig:
    """一条完整生成管线的配置。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def validate(self) -> VoccaConfig:
        self.model.validate()
        self.codec.validate()
        self.generation.validate()
        if self.codec.codebook_size != self.model.codebook_size:
            raise ConfigError(
                "codec.codebook_size 与 model.codebook_size 必须一致"
                f"（{self.codec.codebook_size} != {self.model.codebook_size}）"
            )
        return self

    # -- YAML ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "codec": asdict(self.codec),
            "generation": asdict(self.generation),
        }

    def to_yaml(self, path: PathLike) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoccaConfig:
        try:
            return cls(
                model=ModelConfig(**data.get("model", {})),
                codec=CodecConfig(**data.get("codec", {})),
                generation=GenerationConfig(**data.get("generation", {})),
            )
        except TypeError as exc:  # 未知字段等
            raise ConfigError(f"配置字段不合法：{exc}") from exc

    @classmethod
    def from_yaml(cls, path: PathLike) -> VoccaConfig:
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ConfigError("配置文件顶层必须是映射（mapping）")
        return cls.from_dict(data)
