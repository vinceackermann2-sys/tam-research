from __future__ import annotations

"""Untrained CPU-prototype successor to the failed attention-8 100M/2B screen.

NO GPU, training, scientific seed, replication, or scale-up authorization.
"""

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from architectures.cortex_s.reduced_attention_200m_panel_v2 import (
    ATTENTION_8,
    D_MODEL,
    EXPECTED_TRANSFORMER_PARAMETERS,
    N_LAYERS,
    PanelVariant,
    ScheduledAttentionBlock,
    ScheduledAttentionDenseLM,
    expected_parameter_count,
)


KERNEL_SIZE: Final = 7
SCHEDULE: Final = ATTENTION_8.attention_layers_one_based
FF_HIDDEN: Final = 2_729
SPARSE_BLOCK_COUNT: Final = N_LAYERS - len(SCHEDULE)
BASE_EXPECTED_PARAMETERS: Final = 101_746_176
CONV7_PARAMETERS: Final = SPARSE_BLOCK_COUNT * (D_MODEL * KERNEL_SIZE + 1)
EXPECTED_PARAMETERS: Final = 101_803_536

VARIANT: Final = PanelVariant(
    name="attention8_causal_depthwise_conv7_cpu_v1",
    attention_layers_one_based=SCHEDULE,
    ff_hidden=FF_HIDDEN,
    expected_parameters=BASE_EXPECTED_PARAMETERS,
)


class CausalConv7ScheduledBlock(ScheduledAttentionBlock):
    """FFN-only block gains a length-7 causal, channelwise token mixer."""

    def __init__(self, *, index: int, variant: PanelVariant = VARIANT) -> None:
        super().__init__(index=index, variant=variant)
        if self.has_attention:
            raise ValueError("causal-conv block must replace only FFN-only layers")
        self.depthwise_conv = nn.Conv1d(
            D_MODEL, D_MODEL, kernel_size=KERNEL_SIZE, groups=D_MODEL, bias=False
        )
        # Zero-gated residual initially preserves the corresponding FFN-only block.
        self.conv_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_ff(x)
        # Left padding only prevents look-ahead and retains [batch, time, width].
        mixed = self.depthwise_conv(
            F.pad(normalized.transpose(1, 2), (KERNEL_SIZE - 1, 0))
        ).transpose(1, 2)
        x = x + self.conv_gain * mixed
        return x + self.feedforward(self.norm_ff(x))


class Attention8CausalConv7LM(ScheduledAttentionDenseLM):
    """Preserve eight full-attention blocks, mix locally in the other sixteen."""

    def __init__(self) -> None:
        super().__init__(VARIANT)
        for index, block in enumerate(self.blocks):
            if block.has_attention:
                continue
            upgraded = CausalConv7ScheduledBlock(index=index, variant=VARIANT)
            # Reuse the already initialized FFN/LN; only the mixer is newly added.
            upgraded.norm_ff = block.norm_ff
            upgraded.feedforward = block.feedforward
            self.blocks[index] = upgraded


def prototype_contract() -> dict[str, object]:
    regular = expected_parameter_count(VARIANT)
    total = regular + CONV7_PARAMETERS
    if regular != BASE_EXPECTED_PARAMETERS or total != EXPECTED_PARAMETERS:
        raise RuntimeError("parameter-matching formula drift")
    if total - EXPECTED_TRANSFORMER_PARAMETERS != 16:
        raise RuntimeError("prototype is not almost exactly parameter matched")
    if SCHEDULE != (3, 6, 9, 12, 15, 18, 21, 24):
        raise RuntimeError("frozen eight-layer attention schedule drift")
    return {
        "prototype": VARIANT.name,
        "status": "CPU_PROTOTYPE_ONLY_UNTRAINED",
        "attention_layers_one_based": list(SCHEDULE),
        "conv_layers_one_based": [
            i for i in range(1, N_LAYERS + 1) if i not in SCHEDULE
        ],
        "causal_conv_kernel_size": KERNEL_SIZE,
        "causal_conv_is_depthwise": True,
        "conv_gate_initial_value": 0.0,
        "ff_hidden": FF_HIDDEN,
        "expected_parameters": total,
        "transformer_expected_parameters": EXPECTED_TRANSFORMER_PARAMETERS,
        "parameter_delta": total - EXPECTED_TRANSFORMER_PARAMETERS,
        "science_seed_assigned": False,
        "training_authorized": False,
        "gpu_authorized": False,
        "replication_authorized": False,
        "250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
    }
