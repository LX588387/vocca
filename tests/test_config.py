from __future__ import annotations

import pytest

from vocca.config import CodecConfig, GenerationConfig, ModelConfig, VoccaConfig
from vocca.exceptions import ConfigError


def test_model_config_derived_fields():
    cfg = ModelConfig(codebook_size=100, dim=64, n_heads=8)
    assert cfg.audio_vocab == 102
    assert cfg.audio_bos == 100
    assert cfg.audio_eos == 101
    assert cfg.head_dim == 8


def test_model_config_validate_dim_divisible():
    with pytest.raises(ConfigError):
        ModelConfig(dim=30, n_heads=4).validate()


def test_model_config_validate_gqa():
    with pytest.raises(ConfigError):
        ModelConfig(dim=32, n_heads=4, n_kv_heads=3).validate()


def test_generation_config_validate():
    with pytest.raises(ConfigError):
        GenerationConfig(top_p=1.5).validate()
    with pytest.raises(ConfigError):
        GenerationConfig(max_new_tokens=0).validate()
    with pytest.raises(ConfigError):
        GenerationConfig(temperature=-0.1).validate()


def test_codec_config_frame_rate():
    cfg = CodecConfig(sample_rate=24000, frame_size=240)
    assert cfg.frame_rate == 100.0


def test_vocca_config_codebook_mismatch():
    cfg = VoccaConfig(model=ModelConfig(codebook_size=64), codec=CodecConfig(codebook_size=32))
    with pytest.raises(ConfigError):
        cfg.validate()


def test_vocca_config_yaml_roundtrip(tmp_path):
    cfg = VoccaConfig()
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = VoccaConfig.from_yaml(path)
    assert loaded.model.dim == cfg.model.dim
    assert loaded.codec.name == cfg.codec.name


def test_vocca_config_from_dict_bad_field():
    with pytest.raises(ConfigError):
        VoccaConfig.from_dict({"model": {"nonexistent": 1}})


def test_vocca_config_from_yaml_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        VoccaConfig.from_yaml(path)
