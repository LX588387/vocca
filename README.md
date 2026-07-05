# vocca

> 文本 / 提示驱动的**可控声音生成**框架 —— 情感 / 风格 / 音色可控，KV cache + 分类器无关引导（CFG），支持**流式推理**。核心纯 numpy、零重依赖，torch 为可选后端。

[![CI](https://github.com/LX588387/vocca/actions/workflows/ci.yml/badge.svg)](https://github.com/LX588387/vocca/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`vocca` 把「文本 + 控制条件 → 声学 token → 波形」这条生成链路拆成清晰、可替换的
组件，并用一套纯 numpy 的**参考实现**把它完整跑通：

```
提示词  ──►  文本前端  ──►┐
                          ├─►  声学语言模型（Transformer 解码器 + KV cache）──►  声学 token  ──►  编解码器  ──►  波形
控制条件 ──►  控制编码器 ──►┘         ▲
                                      └── 分类器无关引导（CFG）放大控制信号
```

它是我音频系列的第四块拼图，和前三块接口互补：

| 项目 | 角色 |
|------|------|
| [resona](https://github.com/LX588387/resona) | 自监督音频表示学习 |
| [soniq](https://github.com/LX588387/soniq) | 神经音频编解码器（连续音频 → 离散 token） |
| [auris](https://github.com/LX588387/auris) | 音频语言模型（**理解**：描述 / 问答 / 检索） |
| **vocca** | 可控声音语言模型（**生成**：文本 / 提示 → 语音与通用音频） |

> ⚠️ **关于「模型质量」**：核心实现默认用固定种子**随机初始化**权重，产出的声音是
> 结构化的、可复现的「合成音」，并非训练好的自然语音。vocca 的价值在于把可控生成
> 的**机制**（条件注入、CFG、采样、KV cache、流式解码、编解码器接口）做对、做透、可测；
> 真实权重可通过 `vocca.load_weights` 从 `.npz` 载入，编解码器可换成 soniq 那样的
> 神经声码器，接口保持不变。

## 特性

- 🎛️ **多维可控**：情感、风格、音色 / 说话人、强度、语速、音高、能量，声明式 `ControlSpec`。
- 🧭 **分类器无关引导（CFG）**：并行维护「条件 / 空条件」两套 KV cache，`guidance_scale` 放大控制信号。
- 🌊 **流式推理**：边生成边出声，首块延迟低；分块解码与整段离线解码**逐样本相等**。
- ⚡ **KV cache**：增量解码与整段前向数值一致（有单测严格校验），是流式正确性的基石。
- 🧩 **可插拔编解码器**：内置可听的振荡器编解码器 + Griffin-Lim 声码器，可注册自定义实现。
- 🏷️ **内联提示标签**：`[emotion:happy][speed:1.2] 文本…` 直接写在文本里。
- 🐍 **零重依赖**：核心仅 numpy + pyyaml；torch 可选，仅用于交叉验证 / 未来训练。

## 安装

```bash
pip install vocca            # 核心
pip install "vocca[torch]"   # 附带 torch 交叉验证
pip install "vocca[dev]"     # 开发工具链
```

## 快速上手

```python
from vocca import VoiceGenerator, ControlSpec

gen = VoiceGenerator()

# 整段合成
result = gen.generate("你好，世界", control=ControlSpec(emotion="happy", style="news"))
gen.save(result, "hello.wav")
print(result.duration, "秒，", len(result.tokens), "个声学 token")

# 内联控制标签
gen.generate("[emotion:sad][speed:0.9] 天色渐晚，故事到这里就结束了")

# 命名预设
from vocca import preset
gen.generate("晚间新闻现在开始", control=preset("news_anchor"))
```

### 流式推理

```python
for chunk in gen.stream("这是一段较长的文本，用来演示边生成边播放", chunk_tokens=16):
    play(chunk.waveform)          # 首块无需等整句生成完
    if chunk.is_final:
        break
```

### 命令行

```bash
vocca generate "你好世界" --emotion happy --style news -o hello.wav
vocca generate "长文本…" --stream --chunk 16 -o stream.wav
vocca presets          # 列出内置预设
vocca info             # 打印模型规模
```

## 文档

- [architecture.md](docs/architecture.md) —— 组件与数据流
- [usage.md](docs/usage.md) —— 使用指南与配方
- [design-notes.md](docs/design-notes.md) —— 设计取舍与「为什么这么做」
- [api-reference.md](docs/api-reference.md) —— 公开 API 速查

## 许可证

[MIT](LICENSE) © Dai Yanan
