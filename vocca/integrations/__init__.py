"""可选集成（torch 等）。核心库不依赖它们。"""

from __future__ import annotations

from .torch_backend import forward_logits_torch, torch_available

__all__ = ["torch_available", "forward_logits_torch"]
