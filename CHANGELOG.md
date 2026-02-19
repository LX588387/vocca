# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
