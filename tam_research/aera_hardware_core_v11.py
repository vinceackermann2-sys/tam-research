from __future__ import annotations

import torch
import torch.nn as nn

from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v10 import HardwareAwareAERATextLMV10


class HardwareAwareAERATextLMV11(HardwareAwareAERATextLMV10):
    """AERA-v11: training-stable initialization over the frozen v10 runtime core.

    The first real-language v10 gate exposed a major unfairness: ResearchLM applies
    GPT-style N(0, 0.02) initialization to every Linear/Embedding, whereas the AERA
    inheritance chain left PyTorch defaults in place. In particular nn.Embedding's
    default scale made the tied token-embedding/LM-head logits enormous and produced
    an initial next-token NLL around 131 versus ~11 for the matched Transformer.

    v11 keeps v10 routing/state/memory/expert/reasoning semantics unchanged and
    applies the same Linear/Embedding initialization family as ResearchLM. Raw
    stacked expert matrices already use std=0.02 and are intentionally left alone.
    GRUCell recurrent matrices retain PyTorch's recurrent initialization rather
    than pretending they are ordinary Transformer Linear layers.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.apply(self._init_linear_embedding)
        # lm_head and token_emb share the same Parameter. `apply` may visit both
        # modules, but the final shared tensor is still one N(0,0.02) parameter.
        if self.lm_head.weight is not self.token_emb.weight:
            raise RuntimeError("AERA-v11 requires tied token embedding / LM head")

    @staticmethod
    def _init_linear_embedding(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)


def initial_logit_stats(model: HardwareAwareAERATextLMV11, tokens: torch.Tensor) -> dict[str, float]:
    """CPU-safe initialization diagnostic used before any further GPU training."""
    model.eval()
    with torch.no_grad():
        out = model(tokens, hard=False, route_mode="soft", update_memory=False)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    return {
        "mean": float(logits.float().mean()),
        "std": float(logits.float().std()),
        "abs_max": float(logits.float().abs().max()),
    }
