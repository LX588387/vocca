"""提示标签示例：把控制条件写进文本里的 [key:value] 标签。

运行::

    python examples/prompt_tags.py
"""

from __future__ import annotations

from vocca import VoiceGenerator, parse_prompt


def main() -> None:
    gen = VoiceGenerator()

    prompt = "[emotion:sad][speed:0.9] 天色渐晚[pitch:-2]，故事到这里就结束了"
    parsed = parse_prompt(prompt)
    print(f"原始：{prompt}")
    print(f"纯文本：{parsed.text}")
    print(f"解析出的标签：{parsed.tags}")
    print(
        f"合并后的控制：emotion={parsed.control.emotion} speed={parsed.control.speed} pitch={parsed.control.pitch}"
    )

    result = gen.generate(prompt)
    gen.save(result, "tagged.wav")
    print(f"\n生成 {len(result.tokens)} 个 token → tagged.wav")


if __name__ == "__main__":
    main()
