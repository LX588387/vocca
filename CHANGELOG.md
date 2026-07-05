# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)，格式参考
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [Unreleased]

## [0.3.0] - 2026-07-05

### 新增
- 分类器无关引导（CFG）：并行维护条件 / 空条件两套 KV cache，`guidance_scale` 放大控制信号。
- 流式生成 `VoiceGenerator.stream` 与 `StreamingGenerator`，分块解码与整段离线解码逐样本一致。
- 内联提示标签解析（`[emotion:happy]` 等），随文本直接指定控制条件。
- 可选 torch 后端 `vocca.integrations`，交叉验证 numpy 前向。
- `vocca info` 子命令，打印模型规模与配置。

### 变更
- 编解码器统一到 `Codec.decode_chunk(tokens, state)` 接口，为流式一致性提供保证。
- 控制编码器支持多前缀 token（`control_prefix > 1`）。

## [0.2.0] - 2026-06-28

### 新增
- 声学语言模型 `AcousticLM`：RoPE + RMSNorm + SwiGLU 的解码器，支持 GQA 与 KV cache。
- 采样策略：温度、top-k、top-p（核采样）、重复惩罚。
- 振荡器编解码器 `OscillatorCodec`（可听）与 Griffin-Lim 声码器。
- 权重 `.npz` 存取，YAML 配置读写。

### 修复
- 修正 KV cache 增量前向与整段前向在边界位置的数值不一致。

## [0.1.0] - 2026-06-20

### 新增
- 项目骨架、`ControlSpec` 控制条件与内置预设。
- 字符级 / 字节级文本前端。
- WAV 读写与 STFT / mel 频谱等基础 DSP。

[Unreleased]: https://github.com/LX588387/vocca/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/LX588387/vocca/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/LX588387/vocca/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/LX588387/vocca/releases/tag/v0.1.0
