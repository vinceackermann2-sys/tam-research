from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.rlt.model_gated_scan import (
    GatedScanLightStateRLT,
    associative_affine_scan,
)


class AdaptiveResidualGatedScanRLT(GatedScanLightStateRLT):
    """Parameter-neutral causal dual-path memory with adaptive recurrent gain.

    The same projection columns produce:
      * the candidate update for the recurrent state;
      * the retention gate controlling the associative scan; and
      * a bounded output gain controlling how much scanned memory supplements
        the direct causal encoder path.

    e[t] = causal_encoder(tokens <= t)
    v[t] = tanh(W_value e[t])
    a[t] = sigmoid(W_gate e[t] + 2.0)
    s[t] = a[t] * s[t-1] + (1-a[t]) * v[t]
    gain[t] = 2 * sigmoid(W_gate e[t])
    h[t] = e[t] + gain[t] * RMSNorm(s[t])

    This gains flexibility WITHOUT introducing parameters or another
    projection. If W_gate is identically zero, gain=1 and h matches the
    static ResidualGatedScanRLT exactly. All sequence-dependent operations
    remain causal. This is a new hypothesis, NOT a measured quality result.
    """

    def gated_states(self, memory: torch.Tensor) -> torch.Tensor:
        width = memory.shape[-1]
        if self.merge.weight.shape != (width, 2 * width):
            raise RuntimeError("unexpected merge layout; cannot reuse all weights")

        value_proj = F.linear(memory, self.merge.weight[:, :width])
        gate_proj = F.linear(memory, self.merge.weight[:, width:])

        gate = torch.sigmoid(gate_proj.float() + self.GATE_LOGIT_BIAS)
        write = (1.0 - gate) * torch.tanh(value_proj.float())
        states = associative_affine_scan(gate, write, self.start_state.float())

        recurrent = self.merge_norm(states.to(dtype=memory.dtype))
        recurrent_gain = (2.0 * torch.sigmoid(gate_proj.float())).to(
            dtype=memory.dtype
        )
        if recurrent.shape != memory.shape:
            raise RuntimeError("encoder and scan state shape mismatch")
        return memory + recurrent_gain * recurrent
