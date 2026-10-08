"""CPW #1313: placement-only experiment using the *unchanged* CPW-v5 AFM.

No new memory math, extra trainable weights, task-specific parser, or GPU runner.
"""
from __future__ import annotations

import torch

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v5_afm.model import AFMCPWResearchLM
from tam_research.cpw_v5_afm.protocol import AFM_LAYER, EXPECTED_AFM_PARAMETERS
from tam_research.models import parameter_count

EARLY_AFM_LAYER = 0
LATE_AFM_LAYER = AFM_LAYER


class EarlyAFMResearchLM(AFMCPWResearchLM):
    """Move the actual, frozen late AFM block to the first layer.

    The super constructor creates exactly the frozen #1290 model, including
    all trainable tensors and their original initialization order. Changing
    position is implemented by *swapping modules* rather than reconstructing
    the AFM mixer, which makes same-seed parameter comparisons exact.
    """

    def __init__(self, cfg: CPWV1Config = CPWV1Config()):
        super().__init__(cfg)
        if cfg.n_layers != LATE_AFM_LAYER + 1:
            raise ValueError("placement test requires the frozen 15-block model")
        self.blocks[EARLY_AFM_LAYER], self.blocks[LATE_AFM_LAYER] = (
            self.blocks[LATE_AFM_LAYER],
            self.blocks[EARLY_AFM_LAYER],
        )
        self.afm_layers = (EARLY_AFM_LAYER,)

    @torch.no_grad()
    def memory_stats(self) -> dict[str, float | int]:
        """Same diagnostic fields as CPW-v5, now addressed to layer zero."""
        afm = self.blocks[EARLY_AFM_LAYER].afm
        if afm is None:
            raise RuntimeError("early AFM block is missing")
        return {
            "afm_layers": 1,
            "afm_layer_index": EARLY_AFM_LAYER,
            "memory_rank": afm.rank,
            "memory_state_scalars": afm.state_size,
            "memory_norm": (
                float(afm.last_memory_norm.cpu())
                if afm.last_memory_norm is not None else 0.0
            ),
            "mean_write_gate": (
                float(afm.last_mean_gate.cpu())
                if afm.last_mean_gate is not None else 0.0
            ),
        }


def placement_parameter_counts() -> dict[str, int]:
    """Same number of weights before and after moving the AFM block."""
    early = EarlyAFMResearchLM()
    late = AFMCPWResearchLM()
    counts = {
        "early_afm": parameter_count(early),
        "late_afm": parameter_count(late),
    }
    if counts["early_afm"] != counts["late_afm"]:
        raise AssertionError("placement unexpectedly changed parameter count")
    if counts["early_afm"] != EXPECTED_AFM_PARAMETERS:
        raise AssertionError("the exact frozen AFM parameter budget drifted")
    return counts
