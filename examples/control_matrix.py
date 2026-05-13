"""控制矩阵示例：同一句话在不同情感 / 风格下生成，比较结果差异。

演示「可控」是真的在改变输出——不同控制条件得到不同的声学 token。

运行::

    python examples/control_matrix.py
"""

from __future__ import annotations

from vocca import ControlSpec, VoiceGenerator


def main() -> None:
    gen = VoiceGenerator()
    text = "今天的天气怎么样"

    combos = [
        ControlSpec(emotion="neutral", style="news"),
        ControlSpec(emotion="happy", style="conversational", intensity=0.9),
        ControlSpec(emotion="sad", style="poetry", speed=0.9, pitch=-2.0),
        ControlSpec(emotion="angry", style="shout", energy=2.0),
    ]

    print(f"文本：{text}\n")
    for spec in combos:
        result = gen.generate(text, control=spec)
        head = result.tokens[:8].tolist()
        print(f"{spec.emotion:8s}/{spec.style:14s} → 前 8 个 token: {head}")


if __name__ == "__main__":
    main()
