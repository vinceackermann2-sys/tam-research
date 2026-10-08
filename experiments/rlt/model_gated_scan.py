from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.rlt.model_light_state import LightStateRecurrentTransformer


def associative_affine_scan(
    gates: torch.Tensor,
    writes: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Inclusive parallel prefix scan of s[t] = gates[t] * s[t-1] + writes[t].

    The affine composition (a2,b2) o (a1,b1) =
    (a2*a1, b2+a2*b1) is associative. Doubling takes ceil(log2(T))
    tensor passes, not T dependent Python/compiled graph steps. This uses
    out-of-place operations and therefore supports full autograd/BPTT.
    """
    if gates.ndim != 3 or writes.shape != gates.shape:
        raise ValueError("gates/writes must share a [batch, time, width] shape")
    batch, time_steps, width = gates.shape
    if time_steps < 1:
        raise ValueError("at least one token is required")
    if initial_state.shape not in {(width,), (batch, width)}:
        raise ValueError("initial_state must be [width] or [batch, width]")
    if gates.dtype != writes.dtype or gates.device != writes.device:
        raise ValueError("gates and writes must use the same dtype and device")

    a, b = gates, writes
    stride = 1
    while stride < time_steps:
        # All terms use the values from the *previous* doubling level.
        prev_a = torch.cat(
            (torch.ones_like(a[:, :stride, :]), a[:, :-stride, :]), dim=1
        )
        prev_b = torch.cat(
            (torch.zeros_like(b[:, :stride, :]), b[:, :-stride, :]), dim=1
        )
        b = b + a * prev_b
        a = a * prev_a
        stride *= 2
    return b + a * initial_state.to(dtype=a.dtype, device=a.device).reshape(
        -1, 1, width
    )


def sequential_affine_reference(
    gates: torch.Tensor, writes: torch.Tensor, initial_state: torch.Tensor
) -> torch.Tensor:
    """Reference semantics; testing only, never called by the scan model."""
    batch, time_steps, width = gates.shape
    state = initial_state.to(gates).reshape(-1, width).expand(batch, width)
    outputs = []
    for index in range(time_steps):
        state = gates[:, index] * state + writes[:, index]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


class GatedScanLightStateRLT(LightStateRecurrentTransformer):
    """RLT light-state redesign: parameter-neutral associative gated recurrence.

    Repurposes every weight of the existing [D,2D] merge projection.
    The first D columns project encoder memory to a candidate state;
    the second D columns project the same memory to an input-dependent gate.
    No weights are introduced, removed, or left unused.

        candidate[t] = tanh(W_value @ e[t])
        gate[t] = sigmoid(W_gate @ e[t] + fixed_logit_bias)
        s[t] = gate[t] * s[t-1] + (1-gate[t]) * candidate[t]

    An inclusive associative scan computes all states in logarithmically many
    dependency stages; RMSNorm is applied *after* the scan to preserve
    associativity. This changes RLT's recurrence equation intentionally.
    The encoder and heavy parallel decoder are unchanged.

    This is a new architecture hypothesis, NOT semantically equivalent to
    the previously trained light-state model. Validate quality independently.
    """

    GATE_LOGIT_BIAS = 2.0  # constant; no extra trainable parameter

    def gated_states(self, memory: torch.Tensor) -> torch.Tensor:
        width = memory.shape[-1]
        if self.merge.weight.shape != (width, 2 * width):
            raise RuntimeError("unexpected merge layout; cannot reuse all weights")
        # Keep projections under the caller's autocast policy and run the
        # associative scan in float32 to limit cancellation/underflow.
        value_proj = F.linear(memory, self.merge.weight[:, :width])
        gate_proj = F.linear(memory, self.merge.weight[:, width:])
        gate = torch.sigmoid(gate_proj.float() + self.GATE_LOGIT_BIAS)
        write = (1.0 - gate) * torch.tanh(value_proj.float())
        states = associative_affine_scan(
            gate, write, self.start_state.float()
        )
        return self.merge_norm(states.to(dtype=memory.dtype))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, time_steps = tokens.shape
        memory = self.encode(tokens)
        hidden = self.gated_states(memory)

        memory_kv = [
            stage.cross_attn.precompute(memory) for stage in self.stages
        ]
        for stage_index, stage in enumerate(self.stages):
            hidden = hidden + self._parallel_local_self_attention(
                stage_index, stage.self_norm(hidden)
            )
            hidden = hidden + self._parallel_causal_cross_attention(
                stage_index, stage.cross_norm(hidden), memory_kv[stage_index]
            )
            hidden = hidden + stage.ff(stage.ff_norm(hidden))

        self.last_cache_lengths = tuple(
            min(time_steps, self.cfg.swa_window) for _ in self.stages
        )
        return self.lm_head(self.output_norm(hidden))
