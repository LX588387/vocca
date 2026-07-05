"""一个简单的生成吞吐基准：对比「流式」与「整段」的首块延迟。

这不是严谨的性能测试，只是给出量级感受。运行::

    python benchmarks/bench_generate.py
"""

from __future__ import annotations

from vocca import VoiceGenerator
from vocca.config import GenerationConfig, VoccaConfig
from vocca.utils import timer


def main() -> None:
    cfg = VoccaConfig(generation=GenerationConfig(max_new_tokens=128, seed=0))
    gen = VoiceGenerator(cfg)
    prompt = "用于基准测试的一段中文文本，长度适中。"

    # 整段：拿到全部音频前必须等生成完成。
    with timer() as t_full:
        gen.generate(prompt)
    print(f"整段生成耗时：{t_full.seconds * 1000:.1f} ms")

    # 流式：测「拿到第一块」的时间（首块延迟）。
    with timer() as t_first:
        stream = gen.stream(prompt, chunk_tokens=16)
        next(stream)
    print(f"流式首块延迟：{t_first.seconds * 1000:.1f} ms")

    for chunk in stream:  # 消费剩余，避免生成器悬挂
        if chunk.is_final:
            break

    print("\n首块延迟通常远小于整段耗时——这正是流式推理的意义。")


if __name__ == "__main__":
    main()
