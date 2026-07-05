from __future__ import annotations

import numpy as np
import pytest

from vocca.config import CodecConfig, GenerationConfig, ModelConfig, VoccaConfig
from vocca.generate import VoiceGenerator
from vocca.model import AcousticLM


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        text_vocab=64,
        codebook_size=32,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
        control_prefix=1,
        seed=0,
    )


@pytest.fixture
def tiny_lm(tiny_model_config: ModelConfig) -> AcousticLM:
    return AcousticLM(tiny_model_config)


@pytest.fixture
def tiny_config() -> VoccaConfig:
    return VoccaConfig(
        model=ModelConfig(
            text_vocab=256, codebook_size=32, dim=32, n_layers=2, n_heads=4, n_kv_heads=2
        ),
        codec=CodecConfig(sample_rate=8000, frame_size=160, codebook_size=32),
        generation=GenerationConfig(max_new_tokens=32, seed=1, guidance_scale=1.5),
    )


@pytest.fixture
def generator(tiny_config: VoccaConfig) -> VoiceGenerator:
    return VoiceGenerator(tiny_config)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
