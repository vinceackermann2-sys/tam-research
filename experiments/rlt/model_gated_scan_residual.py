from __future__ import annotations

import torch

from experiments.rlt.model_gated_scan import GatedScanLightStateRLT


class ResidualGatedScanRLT(GatedScanLightStateRLT):
    """Parameter-neutral, causal scan + direct-encoder residual hypothesis.

    Earlier GatedScanLightStateRLT initializes the parallel heavy decoder using
    *only* the state produced by an associative affine scan of encoder memory.
    Here the decoder instead starts with

        h[t] = e[t] + RMSNorm(scan(e[0:t]))

    The encoder sequence e is already causal, so this preserves future-token
    non-leakage; the scan still updates recurrent memory at every token. It
    adds no trainable parameters and uses exactly the same encoder, attention,
    cross-attention and FFN weights as the gated-scan baseline.

    This is a NEW architecture variant, not a semantics-preserving optimization.
    CPU correctness is necessary, but language quality and speed are unknown.
    """

    def gated_states(self, memory: torch.Tensor) -> torch.Tensor:
        recurrent = super().gated_states(memory)
        if recurrent.shape != memory.shape:
            raise RuntimeError("encoder and recurrent feature shape mismatch")
        return memory + recurrent
