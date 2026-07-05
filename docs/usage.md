# 使用指南

## 安装

```bash
pip install vocca            # 核心（numpy + pyyaml）
pip install "vocca[torch]"   # 附带 torch 交叉验证
pip install "vocca[dev]"     # 开发工具链
```

## 最简合成

```python
from vocca import VoiceGenerator, ControlSpec

gen = VoiceGenerator()
result = gen.generate("你好，世界", control=ControlSpec(emotion="happy"))
gen.save(result, "hello.wav")
print(result.duration, "秒", len(result.tokens), "个声学 token")
```

`GenerationResult` 里有：`audio`（`AudioClip`）、`tokens`、`control`（实际生效的
控制）、`text`（去标签后的文本）、`stopped_by_eos`。

## 控制条件

三种方式指定控制，优先级从低到高：

```python
# 1) 直接构造 ControlSpec
gen.generate("文本", control=ControlSpec(emotion="sad", style="poetry", speed=0.9))

# 2) 命名预设
from vocca import preset
gen.generate("晚间新闻现在开始", control=preset("news_anchor"))

# 3) 文本内联标签（会覆盖 base 控制里对应的字段）
gen.generate("[emotion:angry][energy:2.0] 这不可接受！")
```

可控字段与范围：

| 字段 | 取值 |
|------|------|
| `emotion` | neutral / happy / sad / angry / fear / surprise / disgust / calm |
| `style` | default / news / conversational / narration / customer_service / poetry / whisper / shout |
| `speaker` | 任意字符串（音色标识） |
| `intensity` | 0.0 – 1.0 |
| `speed` | 0.5 – 2.0 |
| `pitch` | -12 – 12（半音） |
| `energy` | 0.25 – 4.0 |

## 分类器无关引导（CFG）

`guidance_scale` 控制「有多听话」：

```python
from vocca import VoccaConfig
from vocca.config import GenerationConfig

cfg = VoccaConfig(generation=GenerationConfig(guidance_scale=3.0, seed=0))
gen = VoiceGenerator(cfg)
gen.generate("文本", control=ControlSpec(emotion="happy"))
```

- `1.0`：纯条件采样（不做引导）。
- `>1.0`：放大控制信号，情感 / 风格更鲜明，多样性下降。
- `0.0`：忽略控制。

## 流式推理

```python
buffers = []
for chunk in gen.stream("一段较长的文本，用于演示边生成边播放", chunk_tokens=16):
    buffers.append(chunk.waveform)   # 可直接送播放器
    if chunk.is_final:
        break
```

把各块拼起来，与 `gen.generate(...)`（相同种子）的整段结果**逐样本相等**。

## 采样参数

```python
GenerationConfig(
    max_new_tokens=256,
    temperature=0.9,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.1,
    guidance_scale=1.5,
    seed=0,
)
```

## 配置读写

```python
from vocca import VoccaConfig
cfg = VoccaConfig()
cfg.to_yaml("vocca.yaml")
cfg2 = VoccaConfig.from_yaml("vocca.yaml")
```

## 加载真实权重

参考实现默认随机初始化。要用训练好的权重：

```python
from vocca.config import ModelConfig
from vocca.model import load_weights, AcousticLM

model_cfg = ModelConfig(dim=512, n_layers=12, codebook_size=1024)
weights = load_weights("acoustic.npz", model_cfg)
lm = AcousticLM(model_cfg, weights)
```

## 命令行

```bash
vocca generate "你好世界" --emotion happy --style news -o hello.wav
vocca generate "长文本…" --preset storyteller --stream --chunk 16 -o s.wav
vocca presets
vocca info
```
