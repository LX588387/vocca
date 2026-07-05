"""流式生成示例：边生成边收集，并验证与整段结果一致。

运行::

    python examples/streaming_demo.py
"""

from __future__ import annotations

import numpy as np

from vocca import VoiceGenerator


def main() -> None:
    gen = VoiceGenerator()
    prompt = "这是一段用于演示流式推理的较长文本，逐块产出音频。"

    chunks = []
    total_tokens = 0
    for chunk in gen.stream(prompt, chunk_tokens=16):
        total_tokens += chunk.n_tokens
        chunks.append(chunk.waveform)
        print(
            f"块 #{chunk.index}: {chunk.n_tokens} tokens, {len(chunk)} 采样点, final={chunk.is_final}"
        )

    streamed = np.concatenate(chunks)
    offline = gen.generate(prompt).audio.waveform
    print(f"\n共 {total_tokens} 个声学 token，流式波形长度 {len(streamed)}")
    print(f"流式 == 整段离线：{np.array_equal(streamed, offline)}")


if __name__ == "__main__":
    main()
