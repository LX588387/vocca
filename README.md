# vocca

> 文本 / 提示驱动的**可控声音生成**框架 —— 情感 / 风格 / 音色可控，KV cache + 分类器无关引导（CFG），支持**流式推理**。核心纯 numpy、零重依赖，torch 为可选后端。

## 特性

- 🎛️ 多维可控：情感、风格、音色 / 说话人、强度、语速、音高、能量。
- 🧭 分类器无关引导（CFG）：`guidance_scale` 放大控制信号。
- 🌊 流式推理：边生成边出声，分块解码与整段离线解码逐样本一致。
- ⚡ KV cache：增量解码与整段前向数值一致。
- 🧩 可插拔编解码器；🏷️ 内联提示标签；🐍 核心零重依赖。

## 安装

```bash
pip install vocca
pip install "vocca[torch]"   # 附带 torch 交叉验证
pip install "vocca[dev]"     # 开发工具链
```

## 快速上手

```python
from vocca import VoiceGenerator, ControlSpec

gen = VoiceGenerator()
result = gen.generate("你好，世界", control=ControlSpec(emotion="happy", style="news"))
gen.save(result, "hello.wav")

for chunk in gen.stream("[style:news] 晚间新闻现在开始"):
    play(chunk.waveform)
```

## 文档

见 `docs/`：architecture / usage / design-notes / api-reference。

## 许可证

MIT © Dai Yanan
