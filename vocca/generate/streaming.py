"""流式生成：边解码 token 边出声。

:class:`StreamingGenerator` 把 :func:`~vocca.generate.loop.iter_tokens` 产出的
token 按 ``chunk_tokens`` 分组，每攒够一组就用编解码器的
``decode_chunk`` 解成一小段波形并 ``yield`` 出去。由于编解码器状态（相位等）
跨块携带，**把这些块拼起来与整段离线解码逐样本相等**——这条性质由
``tests/test_streaming.py`` 校验。流式的意义在于**首块延迟**低：不必等整句
生成完就能开始播放。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..codec import Codec
from ..config import GenerationConfig
from ..control import ControlSpec
from ..model import AcousticLM
from ..types import AudioChunk, TokenSequence
from .loop import iter_tokens

__all__ = ["StreamingGenerator"]


class StreamingGenerator:
    """把声学模型 + 编解码器组装成流式出声的解码器。"""

    def __init__(self, lm: AcousticLM, codec: Codec, chunk_tokens: int = 16) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens 必须为正")
        self.lm = lm
        self.codec = codec
        self.chunk_tokens = int(chunk_tokens)

    def stream(
        self,
        text_ids: TokenSequence,
        control: ControlSpec | None = None,
        cfg: GenerationConfig | None = None,
    ) -> Iterator[AudioChunk]:
        """逐块产出 :class:`~vocca.types.AudioChunk`。"""
        state = self.codec.initial_state()
        buffer: list[int] = []
        index = 0
        gen = iter_tokens(self.lm, text_ids, control, cfg)

        def flush(is_final: bool) -> AudioChunk:
            nonlocal state, index
            toks = np.asarray(buffer, dtype=np.int64)
            wav, state = self.codec.decode_chunk(toks, state)
            chunk = AudioChunk(waveform=wav, index=index, is_final=is_final, n_tokens=len(toks))
            index += 1
            buffer.clear()
            return chunk

        for token in gen:
            buffer.append(token)
            if len(buffer) >= self.chunk_tokens:
                yield flush(is_final=False)

        # 收尾：把不足一整块的尾巴也吐出来，并标记 is_final。
        if buffer or index == 0:
            yield flush(is_final=True)
        else:
            # 已经吐过至少一块，补一个空的 final 块方便下游收尾。
            yield AudioChunk(
                waveform=np.zeros(0, dtype=np.float32),
                index=index,
                is_final=True,
                n_tokens=0,
            )
