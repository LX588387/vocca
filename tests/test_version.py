from __future__ import annotations

import re

import vocca


def test_version_string():
    assert isinstance(vocca.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", vocca.__version__)


def test_public_api_exports():
    for name in ("VoiceGenerator", "ControlSpec", "OscillatorCodec", "AcousticLM"):
        assert hasattr(vocca, name)
