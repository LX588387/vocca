"""文本前端：把字符串切成 token id 序列。

采用**字符级**分词（对中文天然友好，无需额外词典），并保留少量特殊
token（BOS/EOS/PAD/UNK）。词表可以显式给定，也可以从语料里 :meth:`fit`
出来；未登录字符回退到 ``<unk>``。字符级方案牺牲了压缩率，但零依赖、
完全确定、对多语言一视同仁，符合参考实现「先跑通链路」的定位。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..exceptions import ConfigError
from ..types import TokenSequence

__all__ = ["TextTokenizer", "ByteTokenizer"]

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
_SPECIAL = (PAD, BOS, EOS, UNK)


class TextTokenizer:
    """字符级文本分词器。

    Args:
        vocab: 字符列表（不含特殊 token）；``None`` 时留空，等待 :meth:`fit`。
    """

    def __init__(self, vocab: Iterable[str] | None = None) -> None:
        chars = list(vocab) if vocab is not None else []
        self._itos = list(_SPECIAL) + [c for c in chars if c not in _SPECIAL]
        self._stoi = {c: i for i, c in enumerate(self._itos)}

    # -- 词表构建 --------------------------------------------------------
    def fit(self, corpus: Iterable[str]) -> TextTokenizer:
        """从语料里收集字符，扩充词表（已存在的字符不重复添加）。"""
        for text in corpus:
            for ch in text:
                if ch not in self._stoi:
                    self._stoi[ch] = len(self._itos)
                    self._itos.append(ch)
        return self

    # -- 基本属性 --------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    @property
    def pad_id(self) -> int:
        return self._stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self._stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self._stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self._stoi[UNK]

    # -- 编解码 ----------------------------------------------------------
    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> TokenSequence:
        """把文本编码成 int64 的 token id 序列。"""
        ids = [self.bos_id] if add_bos else []
        ids.extend(self._stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return np.asarray(ids, dtype=np.int64)

    def decode(self, ids: TokenSequence, *, strip_special: bool = True) -> str:
        """把 token id 序列还原成文本。"""
        out = []
        specials = set(range(len(_SPECIAL)))
        for i in np.asarray(ids).tolist():
            if strip_special and i in specials:
                continue
            if not 0 <= i < len(self._itos):
                raise ConfigError(f"token id {i} 超出词表范围 [0, {len(self._itos)})")
            out.append(self._itos[i])
        return "".join(out)

    # -- 序列化 ----------------------------------------------------------
    def get_vocab(self) -> list[str]:
        """返回完整词表（含特殊 token），顺序即 id 顺序。"""
        return list(self._itos)


class ByteTokenizer:
    """UTF-8 字节级分词器：词表固定为 256，对任意文本都不会「未登录」。

    生成管线默认用它——中文一个字对应 3 个字节，虽然序列更长，但词表恒定、
    无需 :meth:`~TextTokenizer.fit`，也不会因为出现新字而越出文本嵌入表。
    """

    vocab_size = 256

    def encode(self, text: str) -> TokenSequence:
        """把文本编码成 ``[0, 256)`` 的字节 id 序列。"""
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)

    def decode(self, ids: TokenSequence) -> str:
        """把字节 id 序列还原成文本（非法字节按替换符处理）。"""
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size and (arr.min() < 0 or arr.max() > 255):
            raise ConfigError("字节 token id 必须落在 [0, 256)")
        return bytes(arr.astype(np.uint8).tolist()).decode("utf-8", errors="replace")
