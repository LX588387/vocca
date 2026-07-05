# 架构

vocca 把「文本 / 提示 → 波形」拆成几个各司其职、可替换的组件。本文自顶向下
介绍数据如何在它们之间流动。

## 总览

```
                         ControlSpec (情感/风格/音色/语速/音高/能量)
                              │
                              ▼
提示词 ──► 文本前端 ──► 文本 token          控制编码器 ──► 条件前缀
   │        (ByteTokenizer)  │                  │
   │                         ▼                  ▼
   └───► 提示标签解析 ──►  ┌─────────────────────────────────┐
                          │        声学语言模型 (AcousticLM)   │
                          │  [条件前缀][文本 token][声学 token] │
                          │   RoPE + RMSNorm + SwiGLU + GQA     │
                          │        因果注意力 + KV cache         │
                          └───────────────┬─────────────────────┘
                                          │ logits (声学词表)
                    分类器无关引导 (CFG) ──┤
                                          ▼
                                   采样 (温度/top-k/top-p/重复惩罚)
                                          │ 声学 token
                                          ▼
                                   编解码器 (OscillatorCodec / ...)
                                          │ decode_chunk(tokens, state)
                                          ▼
                                        波形 ──► WAV
```

## 组件

### 文本前端（`vocca.text`）

- `ByteTokenizer`：UTF-8 字节级，词表恒为 256，对任意语言都不会「未登录」。生成
  管线默认用它。
- `TextTokenizer`：字符级，可 `fit` 出自定义词表，带 BOS/EOS/PAD/UNK。

### 控制子系统（`vocca.control`）

- `ControlSpec`：声明式的可控属性，构造时即做范围校验。
- `parse_prompt`：从文本里抽取 `[key:value]` 内联标签，叠加到基线控制上。
- `ControlEncoder`：把 `ControlSpec` 编码成条件向量（确定性哈希嵌入 + 连续字段
  归一化），作为**前缀**注入模型。
- `presets`：给常用组合起名字（`news_anchor`、`storyteller` 等）。

### 声学语言模型（`vocca.model`）

解码器-only Transformer，逐组件拆开：

- `layers`：RMSNorm、RoPE、SwiGLU、线性层。
- `attention`：带 KV cache 的因果多头注意力，支持分组查询（GQA）。**增量解码与
  整段前向数值一致**是它的核心契约。
- `transformer`：块前向与整栈前向。
- `lm.AcousticLM`：拼装嵌入 + 条件前缀，暴露 `forward` / `prefill` / `step`。
- `weights`：随机初始化与 `.npz` 存取。

### 采样与引导（`vocca.sampling` / `vocca.guidance`）

- 采样：温度、top-k、top-p、重复惩罚。
- CFG：并行跑「条件 / 空条件」两套 logits，按 `guidance_scale` 外推，放大控制信号。

### 编解码器（`vocca.codec`）

- `Codec`：抽象基类，核心是 `decode_chunk(tokens, state)`。
- `OscillatorCodec`：连续相位加性合成，可听、确定、流式精确。
- `GriffinLim`：迭代式幅度谱声码器。
- `MockCodec`：直流参考实现，测试用。

### 生成编排（`vocca.generate`）

- `loop`：核心自回归循环（cache + CFG + 采样），以生成器形式产出 token。
- `generator.VoiceGenerator`：高层入口，串起全部组件。
- `streaming.StreamingGenerator`：按 `chunk_tokens` 分块，边解码边 `yield`。

## 关键不变量

1. **cache 一致性**：`prefill + step` 逐位置等于 `forward`（见 `tests/test_lm.py`）。
2. **流式一致性**：分块解码拼接 == 整段解码，逐样本相等（见 `tests/test_streaming.py`）。
3. **可复现**：固定 `seed` 时，生成的 token 与波形完全确定。

这三条让「流式」不是近似，而是与离线结果严格一致的等价实现。
