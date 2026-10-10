"""CPU-first bounded-context, fixed-chunk streaming RLT prototype.

This is an explicit new architectural experiment, NOT the existing published
RLT model running in streaming mode. Its encoder/self/cross-attention only see
the current fixed-size chunk; the sole cross-chunk learned state is the gated
affine scan's [batch,d_model] carry. Positional embeddings restart each chunk.

The no-carry control is *exactly parameter matched* and performs the same
operations, but resets the scan state at every boundary. Both models have
no attention or KV cache spanning chunks. The user of this API supplies whole
fixed-size chunks, except for the final partial chunk; splitting the same data
into different-size chunks changes the attention windows and is NOT equivalent.
Full backprop retains activations from all chunks, so training memory is NOT
constant in total sequence length.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, parameter_count
from experiments.rlt.model_gated_scan import (
    GatedScanLightStateRLT,
    associative_affine_scan,
)


@dataclass(frozen=True)
class BoundedStreamingConfig:
    rlt: RLTConfig
    chunk_size: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if self.chunk_size > self.rlt.max_seq_len:
            raise ValueError("chunk_size exceeds per-chunk learned positional embedding")
        if self.chunk_size > self.rlt.swa_window:
            raise ValueError("chunk_size must not exceed local decoder attention window")


@dataclass
class BoundedStreamState:
    recurrent: torch.Tensor
    total_tokens: int = 0


class BoundedChunkGatedRLT(GatedScanLightStateRLT):
    """Fixed-window encoder/cross-attention with optional inter-chunk scan carry.

    carry_between_chunks is a *non-trainable protocol flag*, not a new weight.
    The model/weights/control compute match exactly except for the previous
    recurrent state passed to the same affine scan.
    """

    def __init__(
        self,
        cfg: BoundedStreamingConfig,
        *,
        carry_between_chunks: bool = True,
    ) -> None:
        super().__init__(cfg.rlt)
        self.chunk_size = cfg.chunk_size
        self.carry_between_chunks = bool(carry_between_chunks)

    def init_stream(self, batch_size: int) -> BoundedStreamState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        state = self.start_state.unsqueeze(0).expand(batch_size, -1)
        return BoundedStreamState(recurrent=state, total_tokens=0)

    def forward_chunk(
        self,
        tokens: torch.Tensor,
        state: BoundedStreamState,
    ) -> tuple[torch.Tensor, BoundedStreamState]:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be int64 [batch, chunk_time]")
        batch, t = tokens.shape
        if batch == 0 or t < 1 or t > self.chunk_size:
            raise ValueError("invalid chunk time or batch")
        if state.recurrent.shape != (batch, self.cfg.d_model):
            raise ValueError("invalid stream-state dimensions")
        if state.total_tokens < 0 or state.total_tokens % self.chunk_size != 0:
            raise ValueError("chunk boundary misaligned")
        if state.recurrent.device != tokens.device:
            raise ValueError("state and tokens on different devices")

        # Each encode call sees only the current chunk and local positions.
        memory = self.encode(tokens)
        width = memory.size(-1)
        values = F.linear(memory, self.merge.weight[:, :width])
        gates = F.linear(memory, self.merge.weight[:, width:])
        retention = torch.sigmoid(gates.float() + self.GATE_LOGIT_BIAS)
        writes = (1.0 - retention) * torch.tanh(values.float())
        prev = state.recurrent if self.carry_between_chunks else (
            self.start_state.unsqueeze(0).expand(batch, -1)
        )
        scan_states = associative_affine_scan(
            retention, writes, prev.float()
        )
        hidden = memory + self.merge_norm(scan_states.to(memory.dtype))
        kv = [stage.cross_attn.precompute(memory) for stage in self.stages]
        for i, stage in enumerate(self.stages):
            hidden = hidden + self._parallel_local_self_attention(
                i, stage.self_norm(hidden)
            )
            hidden = hidden + self._parallel_causal_cross_attention(
                i, stage.cross_norm(hidden), kv[i]
            )
            hidden = hidden + stage.ff(stage.ff_norm(hidden))
        logits = self.lm_head(self.output_norm(hidden))
        next_state = BoundedStreamState(
            recurrent=scan_states[:, -1, :],
            total_tokens=state.total_tokens + t,
        )
        # The final partial chunk terminates a stream; no continuation after
        # its total_tokens becomes misaligned.
        return logits, next_state

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be int64 [batch,time]")
        batch, time = tokens.shape
        if batch == 0 or time < 1:
            raise ValueError("empty stream")
        stream = self.init_stream(batch)
        outputs: list[torch.Tensor] = []
        for lo in range(0, time, self.chunk_size):
            logits, stream = self.forward_chunk(
                tokens[:, lo:lo + self.chunk_size], stream
            )
            outputs.append(logits)
        return torch.cat(outputs, dim=1)

    def readout_parameter_count(self) -> int:
        return parameter_count(self)


def tiny_streaming_config() -> BoundedStreamingConfig:
    return BoundedStreamingConfig(
        RLTConfig(
            vocab_size=16, d_model=32, n_heads=4, n_stages=2,
            max_seq_len=8, ff_mult=2, swa_window=8,
        ),
        chunk_size=8,
    )
