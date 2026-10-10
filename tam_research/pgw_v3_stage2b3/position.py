"""PGW-v3 Stage-2B3: fixed global positions for the frozen causal reference.

No trainable parameters, buffers, optimization, scientific seeds or GPU.
The Stage-2B2 model is imported unchanged, and the new control merely adds a
deterministic non-trainable absolute sinusoid to its token embeddings.
"""
from __future__ import annotations

import math

import torch

from tam_research.pgw_v3_stage2b2.reference import (
    CausalReference,
    CausalReferenceConfig,
)


def absolute_sinusoidal_positions(
    length: int, *, width: int = 32
) -> torch.Tensor:
    """PE[pos,2i]=sin(pos/10000^(2i/width)); PE[pos,2i+1]=cos(...).

    Float32 CPU, no tensor parameters/buffers or global RNG. Not a tokenizer
    feature, target leak, learned positional embedding or model training.
    """
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer")
    if type(width) is not int or width < 2 or width % 2:
        raise ValueError("width must be an even integer >= 2")
    positions = torch.arange(length, dtype=torch.float32, device="cpu")[:, None]
    indices = torch.arange(0, width, 2, dtype=torch.float32, device="cpu")
    rate = torch.exp((-math.log(10000.0) / width) * indices)
    radians = positions * rate[None, :]
    pe = torch.empty(length, width, dtype=torch.float32, device="cpu")
    pe[:, 0::2] = torch.sin(radians)
    pe[:, 1::2] = torch.cos(radians)
    return pe


class GlobalPositionCausalReference(CausalReference):
    """20,347-param full-prefix Transformer + parameter-free global positions.

    Inherits all frozen Stage-2B2 causal masks, 33-class answer semantics,
    auxiliary prediction objective, parameter tensors and strict input guards.
    Chunk-only is preserved only as the unchanged prior negative reference.
    """

    def __init__(
        self,
        cfg: CausalReferenceConfig = CausalReferenceConfig(attention_scope="full"),
    ) -> None:
        if cfg.attention_scope != "full":
            raise ValueError("global-position Stage-2B3 control requires full attention")
        super().__init__(cfg)

    def input_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        # Inherited forward/forward_with_aux calls frozen _validate first.
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.device.type != "cpu":
            raise ValueError("expected CPU LongTensor[B,T]")
        local = super().input_embeddings(tokens)
        global_position = absolute_sinusoidal_positions(tokens.shape[1], width=self.cfg.d_model)
        return local + global_position.to(dtype=local.dtype)[None, :, :]
