"""最简示例：把一句话合成为 WAV。

运行::

    python examples/quickstart.py
"""

from __future__ import annotations

from vocca import ControlSpec, VoiceGenerator


def main() -> None:
    gen = VoiceGenerator()
    result = gen.generate("你好，世界", control=ControlSpec(emotion="happy", style="news"))
    gen.save(result, "hello.wav")
    print(f"生成 {len(result.tokens)} 个声学 token，时长 {result.duration:.2f}s → hello.wav")
    print(f"实际控制：emotion={result.control.emotion} style={result.control.style}")


if __name__ == "__main__":
    main()
