from __future__ import annotations

from vocca.control import ControlSpec
from vocca.generate import GenerationResult
from vocca.pipeline import load_generator, synthesize, synthesize_to_file


def test_load_generator_default():
    gen = load_generator()
    assert gen.sample_rate > 0


def test_load_generator_from_yaml(tmp_path):
    from vocca.config import VoccaConfig

    path = tmp_path / "cfg.yaml"
    VoccaConfig().to_yaml(path)
    gen = load_generator(path)
    assert gen.config.model.dim == VoccaConfig().model.dim


def test_synthesize_with_spec():
    result = synthesize("你好", control=ControlSpec(emotion="happy"))
    assert isinstance(result, GenerationResult)
    assert result.control.emotion == "happy"


def test_synthesize_with_preset_name():
    result = synthesize("晚间新闻", control="news_anchor")
    assert result.control.style == "news"


def test_synthesize_to_file(tmp_path):
    from vocca.audio import load_wav

    path = tmp_path / "out.wav"
    result = synthesize_to_file("测试", path)
    assert path.exists()
    wav, sr = load_wav(path)
    assert len(wav) == len(result.audio.waveform)


def test_generate_applies_inline_tags():
    result = synthesize("[emotion:sad] 再见")
    assert result.control.emotion == "sad"
    assert result.text == "再见"
