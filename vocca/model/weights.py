"""模型权重容器与初始化 / 存取。

参考实现默认用一个固定种子随机初始化权重——它**不是**训练好的模型，
输出的声学 token 是「结构化噪声」，但完全可复现。真实权重可以用
:func:`save_weights` / :func:`load_weights` 以 ``.npz`` 形式落盘与加载，
键名与这里的字段一一对应，方便从 PyTorch 训练侧导出。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ModelConfig
from ..exceptions import ShapeError
from ..types import Hidden, PathLike
from .attention import AttentionWeights

__all__ = ["BlockWeights", "ModelWeights", "init_weights", "save_weights", "load_weights"]


@dataclass
class BlockWeights:
    """单个 Transformer 块的权重。"""

    attn: AttentionWeights
    attn_norm: Hidden
    ffn_norm: Hidden
    w_gate: Hidden  # SwiGLU 门控 (hidden, dim)
    w_up: Hidden  # (hidden, dim)
    w_down: Hidden  # (dim, hidden)


@dataclass
class ModelWeights:
    """整个声学语言模型的权重。"""

    config: ModelConfig
    text_embed: Hidden  # (text_vocab, dim)
    audio_embed: Hidden  # (audio_vocab, dim)
    blocks: list[BlockWeights]
    final_norm: Hidden  # (dim,)
    head: Hidden  # (audio_vocab, dim)

    def num_parameters(self) -> int:
        """统计参数总量，便于打印模型规模。"""
        total = self.text_embed.size + self.audio_embed.size + self.final_norm.size
        total += self.head.size
        for b in self.blocks:
            total += b.attn.wq.size + b.attn.wk.size + b.attn.wv.size + b.attn.wo.size
            total += b.attn_norm.size + b.ffn_norm.size
            total += b.w_gate.size + b.w_up.size + b.w_down.size
        return int(total)


def _ffn_hidden(cfg: ModelConfig) -> int:
    # 取 8 的整数倍，贴近真实实现的对齐习惯。
    raw = int(cfg.dim * cfg.ffn_mult)
    return max(8, (raw + 7) // 8 * 8)


def init_weights(cfg: ModelConfig, seed: int | None = None) -> ModelWeights:
    """按配置随机初始化一套权重（确定性，取决于 ``seed``）。"""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    dim = cfg.dim
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    hidden = _ffn_hidden(cfg)

    def randn(*shape: int, scale: float = 0.02) -> Hidden:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    blocks = []
    for _ in range(cfg.n_layers):
        attn = AttentionWeights(
            wq=randn(dim, dim),
            wk=randn(kv_dim, dim),
            wv=randn(kv_dim, dim),
            wo=randn(dim, dim),
        )
        blocks.append(
            BlockWeights(
                attn=attn,
                attn_norm=np.ones(dim, dtype=np.float32),
                ffn_norm=np.ones(dim, dtype=np.float32),
                w_gate=randn(hidden, dim),
                w_up=randn(hidden, dim),
                w_down=randn(dim, hidden),
            )
        )

    return ModelWeights(
        config=cfg,
        text_embed=randn(cfg.text_vocab, dim),
        audio_embed=randn(cfg.audio_vocab, dim),
        blocks=blocks,
        final_norm=np.ones(dim, dtype=np.float32),
        head=randn(cfg.audio_vocab, dim),
    )


def save_weights(path: PathLike, weights: ModelWeights) -> None:
    """把权重打平存成 ``.npz``。"""
    flat: dict[str, np.ndarray] = {
        "text_embed": weights.text_embed,
        "audio_embed": weights.audio_embed,
        "final_norm": weights.final_norm,
        "head": weights.head,
    }
    for i, b in enumerate(weights.blocks):
        flat[f"blocks.{i}.attn.wq"] = b.attn.wq
        flat[f"blocks.{i}.attn.wk"] = b.attn.wk
        flat[f"blocks.{i}.attn.wv"] = b.attn.wv
        flat[f"blocks.{i}.attn.wo"] = b.attn.wo
        flat[f"blocks.{i}.attn_norm"] = b.attn_norm
        flat[f"blocks.{i}.ffn_norm"] = b.ffn_norm
        flat[f"blocks.{i}.w_gate"] = b.w_gate
        flat[f"blocks.{i}.w_up"] = b.w_up
        flat[f"blocks.{i}.w_down"] = b.w_down
    np.savez(Path(path), **flat)  # type: ignore[arg-type]


def load_weights(path: PathLike, cfg: ModelConfig) -> ModelWeights:
    """从 ``.npz`` 加载权重，并按 ``cfg`` 校验形状。"""
    cfg.validate()
    data = np.load(Path(path))
    try:
        blocks = []
        for i in range(cfg.n_layers):
            attn = AttentionWeights(
                wq=data[f"blocks.{i}.attn.wq"],
                wk=data[f"blocks.{i}.attn.wk"],
                wv=data[f"blocks.{i}.attn.wv"],
                wo=data[f"blocks.{i}.attn.wo"],
            )
            blocks.append(
                BlockWeights(
                    attn=attn,
                    attn_norm=data[f"blocks.{i}.attn_norm"],
                    ffn_norm=data[f"blocks.{i}.ffn_norm"],
                    w_gate=data[f"blocks.{i}.w_gate"],
                    w_up=data[f"blocks.{i}.w_up"],
                    w_down=data[f"blocks.{i}.w_down"],
                )
            )
        weights = ModelWeights(
            config=cfg,
            text_embed=data["text_embed"],
            audio_embed=data["audio_embed"],
            blocks=blocks,
            final_norm=data["final_norm"],
            head=data["head"],
        )
    except KeyError as exc:
        raise ShapeError(f"权重文件缺少键 {exc}") from exc

    if weights.text_embed.shape != (cfg.text_vocab, cfg.dim):
        raise ShapeError("text_embed 形状与配置不符")
    if weights.audio_embed.shape != (cfg.audio_vocab, cfg.dim):
        raise ShapeError("audio_embed 形状与配置不符")
    return weights
