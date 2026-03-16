"""``vocca`` 命令行入口。

子命令：

* ``generate``：把文本合成为 WAV，可指定情感 / 风格 / 预设，可流式；
* ``presets``：列出内置控制预设；
* ``info``：打印默认模型规模与配置概览。

刻意只用标准库 ``argparse``，不引入额外 CLI 框架。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .config import GenerationConfig, VoccaConfig
from .control import ControlSpec, available_presets, preset
from .exceptions import VoccaError
from .generate import VoiceGenerator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vocca", description="可控声音生成框架")
    parser.add_argument("--version", action="version", version=f"vocca {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="把文本合成为音频")
    gen.add_argument("text", help="要合成的文本，可含内联标签如 [emotion:happy]")
    gen.add_argument("-o", "--output", default="out.wav", help="输出 WAV 路径")
    gen.add_argument("--preset", help="使用命名预设（见 vocca presets）")
    gen.add_argument("--emotion", help="情感标签")
    gen.add_argument("--style", help="风格标签")
    gen.add_argument("--speaker", help="音色 / 说话人标识")
    gen.add_argument("--intensity", type=float, help="情感强度 [0,1]")
    gen.add_argument("--speed", type=float, help="语速倍率 [0.5,2]")
    gen.add_argument("--guidance", type=float, default=1.5, help="CFG 引导强度")
    gen.add_argument("--max-tokens", type=int, default=128, help="最多生成的声学 token 数")
    gen.add_argument("--seed", type=int, default=0, help="采样随机种子")
    gen.add_argument("--stream", action="store_true", help="流式合成（分块出声）")
    gen.add_argument("--chunk", type=int, default=16, help="流式每块 token 数")

    sub.add_parser("presets", help="列出内置控制预设")
    sub.add_parser("info", help="打印默认模型规模与配置")
    return parser


def _control_from_args(args: argparse.Namespace) -> ControlSpec:
    base = preset(args.preset) if args.preset else ControlSpec()
    changes = {}
    for field in ("emotion", "style", "speaker", "intensity", "speed"):
        value = getattr(args, field)
        if value is not None:
            changes[field] = value
    return base.with_(**changes) if changes else base


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = VoccaConfig(
        generation=GenerationConfig(
            max_new_tokens=args.max_tokens, guidance_scale=args.guidance, seed=args.seed
        )
    )
    gen = VoiceGenerator(cfg)
    control = _control_from_args(args)

    if args.stream:
        from .audio import save_wav

        chunks = list(gen.stream(args.text, control=control, chunk_tokens=args.chunk))
        import numpy as np

        parts = [c.waveform for c in chunks] if chunks else [np.zeros(0, dtype=np.float32)]
        wav = np.concatenate(parts).astype(np.float32)
        save_wav(args.output, wav, gen.sample_rate)
        print(f"流式生成 {len(chunks)} 块，共 {len(wav)} 采样点 → {args.output}")
    else:
        result = gen.generate(args.text, control=control)
        gen.save(result, args.output)
        print(
            f"生成 {len(result.tokens)} 个声学 token，"
            f"时长 {result.duration:.2f}s → {args.output}"
        )
    return 0


def _cmd_presets(_: argparse.Namespace) -> int:
    for name in available_presets():
        spec = preset(name)
        print(f"{name:16s} emotion={spec.emotion:9s} style={spec.style}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    gen = VoiceGenerator()
    m = gen.config.model
    print(f"vocca {__version__}")
    print(f"模型参数量: {gen.lm.num_parameters:,}")
    print(f"结构: dim={m.dim} layers={m.n_layers} heads={m.n_heads} kv={m.n_kv_heads}")
    print(f"词表: text={m.text_vocab} audio={m.audio_vocab} (codebook={m.codebook_size})")
    print(f"编解码器: {gen.config.codec.name} @ {gen.sample_rate}Hz")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "generate": _cmd_generate,
        "presets": _cmd_presets,
        "info": _cmd_info,
    }
    try:
        return handlers[args.command](args)
    except VoccaError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
