"""可选的 torch 后端：用 PyTorch 复算一遍前向，交叉验证 numpy 参考实现。

核心库不依赖 torch；本模块只在装了 ``torch`` 时才可用（``pip install vocca[torch]``）。
它把 :class:`~vocca.model.weights.ModelWeights` 搬到 torch 张量上，用**相同的**
数学重跑一次整段前向，从而在 ``tests/test_torch_backend.py`` 里断言两套实现
逐元素接近。这既证明 numpy 实现的正确性，也给未来接 GPU 训练留了接口。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..control import ControlSpec
from ..model.lm import AcousticLM
from ..types import Logits, TokenSequence

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["torch_available", "forward_logits_torch"]


def torch_available() -> bool:
    """当前环境是否装了 torch。"""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:  # pragma: no cover - 取决于环境
        return False


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    import torch

    ms = torch.mean(x * x, dim=-1, keepdim=True)
    return x / torch.sqrt(ms + eps) * w


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    import torch

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


def forward_logits_torch(
    lm: AcousticLM,
    text_ids: TokenSequence,
    audio_ids: TokenSequence,
    control: ControlSpec | None = None,
) -> Logits:
    """用 torch 复算 :meth:`AcousticLM.forward`，返回声学段 logits。"""
    import torch

    cfg = lm.config
    w = lm.weights
    tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731

    prefix = torch.tensor(lm._control_prefix(control), dtype=torch.float32)
    text = tt(w.text_embed[np.asarray(text_ids, dtype=np.int64)])
    audio = tt(w.audio_embed[np.asarray(audio_ids, dtype=np.int64)])
    hidden = torch.cat([prefix, text, audio], dim=0)
    seq = hidden.shape[0]

    cos = tt(lm._cos[:seq]).unsqueeze(0)
    sin = tt(lm._sin[:seq]).unsqueeze(0)
    head_dim = cfg.head_dim
    n_heads, n_kv = cfg.n_heads, cfg.n_kv_heads
    mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)

    x = hidden
    for block in w.blocks:
        h = _rms_norm(x, tt(block.attn_norm))
        q = (h @ tt(block.attn.wq).T).view(seq, n_heads, head_dim).transpose(0, 1)
        k = (h @ tt(block.attn.wk).T).view(seq, n_kv, head_dim).transpose(0, 1)
        v = (h @ tt(block.attn.wv).T).view(seq, n_kv, head_dim).transpose(0, 1)
        q = _rope(q, cos, sin)
        k = _rope(k, cos, sin)
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
        scores = (q @ k.transpose(-1, -2)) / (head_dim**0.5) + mask
        attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(0, 1).reshape(seq, cfg.dim) @ tt(block.attn.wo).T
        x = x + attn
        h2 = _rms_norm(x, tt(block.ffn_norm))
        gate = torch.nn.functional.silu(h2 @ tt(block.w_gate).T)
        up = h2 @ tt(block.w_up).T
        x = x + (gate * up) @ tt(block.w_down).T

    x = _rms_norm(x, tt(w.final_norm))
    logits = x @ tt(w.head).T
    n_audio = len(np.asarray(audio_ids))
    return logits[-n_audio:].detach().cpu().numpy().astype(np.float32)
