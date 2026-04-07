# vocca

> 文本 / 提示驱动的可控声音生成框架：情感 / 风格 / 音色可控，支持流式推理。核心纯 numpy。

## 特性

- 多维可控：情感、风格、音色、语速、音高、能量。
- 自回归声学语言模型（Transformer 解码器 + KV cache）。
- 可插拔编解码器：token ↔ 波形。
- 零重依赖：核心仅 numpy + pyyaml。

## 安装

```bash
pip install vocca
pip install "vocca[dev]"   # 开发
```

## 快速上手

```python
from vocca import VoiceGenerator, ControlSpec

gen = VoiceGenerator()
result = gen.generate("你好，世界", control=ControlSpec(emotion="happy"))
gen.save(result, "hello.wav")
```

更多文档见 `docs/`。

## 许可证

MIT © Dai Yanan
