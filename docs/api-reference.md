# API 速查

只列常用的公开接口；完整签名以源码 docstring 为准。

## 顶层

```python
import vocca
vocca.__version__
```

| 名称 | 说明 |
|------|------|
| `VoiceGenerator` | 高层生成器（整段 + 流式） |
| `StreamingGenerator` | 底层流式解码器 |
| `GenerationResult` | 整段生成结果 |
| `ControlSpec` | 控制条件 |
| `preset(name)` / `register_preset` / `available_presets` | 控制预设 |
| `parse_prompt(text, base=None)` | 解析内联控制标签 |
| `synthesize` / `synthesize_to_file` / `load_generator` | 便捷入口 |
| `AcousticLM` | 声学语言模型 |
| `OscillatorCodec` / `MockCodec` / `GriffinLim` / `build_codec` / `register_codec` | 编解码 |
| `ByteTokenizer` / `TextTokenizer` | 文本前端 |
| `save_wav` / `load_wav` / `mel_spectrogram` / `log_mel_spectrogram` | 音频 I/O 与 DSP |
| `VoccaConfig` / `ModelConfig` / `CodecConfig` / `GenerationConfig` | 配置 |
| `sample_token` / `classifier_free_guidance` | 采样 / 引导 |

## VoiceGenerator

```python
VoiceGenerator(config: VoccaConfig | None = None,
               weights: ModelWeights | None = None,
               codec: Codec | None = None)

.generate(prompt: str, control: ControlSpec | None = None) -> GenerationResult
.stream(prompt: str, control=None, chunk_tokens: int = 16) -> Iterator[AudioChunk]
.save(result: GenerationResult, path) -> None
.sample_rate -> int
```

## AcousticLM

```python
AcousticLM(config: ModelConfig, weights=None, control_encoder=None)

.forward(text_ids, audio_ids, control=None) -> Logits    # (len(audio_ids), audio_vocab)
.prefill(text_ids, cache, control=None) -> Logits         # (audio_vocab,)
.step(audio_id, cache) -> Logits                          # (audio_vocab,)
.new_cache() -> list[LayerCache]
.num_parameters -> int
```

## Codec

```python
class Codec:
    sample_rate: int
    frame_size: int
    codebook_size: int
    def initial_state(self) -> Any: ...
    def decode_chunk(self, tokens, state) -> tuple[Waveform, Any]: ...
    def decode(self, tokens) -> Waveform
    def encode(self, waveform) -> TokenSequence   # 可选
    frame_rate -> float
    num_samples(n_tokens) -> int
```

## ControlSpec

```python
ControlSpec(emotion="neutral", style="default", speaker="default",
            intensity=0.6, speed=1.0, pitch=0.0, energy=1.0, extra={})

.with_(**changes) -> ControlSpec
.to_dict() / ControlSpec.from_dict(data)
```

## 采样 / 引导

```python
sample_token(logits, rng, *, temperature=1.0, top_k=0, top_p=1.0,
             history=None, repetition_penalty=1.0) -> int
classifier_free_guidance(cond, uncond, scale) -> Logits
```
