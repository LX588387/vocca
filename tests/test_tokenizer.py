from __future__ import annotations

import numpy as np
import pytest

from vocca.exceptions import ConfigError
from vocca.text import ByteTokenizer, TextTokenizer


def test_char_tokenizer_roundtrip():
    tok = TextTokenizer(vocab=list("你好世界"))
    ids = tok.encode("你好")
    assert tok.decode(ids) == "你好"


def test_char_tokenizer_specials():
    tok = TextTokenizer(vocab=list("ab"))
    ids = tok.encode("a", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_char_tokenizer_unk():
    tok = TextTokenizer(vocab=list("ab"))
    ids = tok.encode("z", add_bos=False, add_eos=False)
    assert ids[0] == tok.unk_id


def test_char_tokenizer_fit_extends_vocab():
    tok = TextTokenizer()
    before = tok.vocab_size
    tok.fit(["新字"])
    assert tok.vocab_size == before + 2


def test_char_decode_out_of_range_raises():
    tok = TextTokenizer(vocab=list("ab"))
    with pytest.raises(ConfigError):
        tok.decode(np.array([999], dtype=np.int64), strip_special=False)


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    ids = tok.encode("你好，world")
    assert ids.max() < 256
    assert tok.decode(ids) == "你好，world"


def test_byte_tokenizer_vocab_fixed():
    assert ByteTokenizer.vocab_size == 256


def test_byte_tokenizer_empty():
    tok = ByteTokenizer()
    assert tok.decode(tok.encode("")) == ""


def test_byte_tokenizer_bad_id_raises():
    with pytest.raises(ConfigError):
        ByteTokenizer().decode(np.array([300], dtype=np.int64))
