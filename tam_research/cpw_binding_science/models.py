"""Five *frozen-source* arms for the corrected CPW binding benchmark.

No new architecture is defined by this module. Parameter matches are *pairwise*.
GPU use, scientific seeds and paid execution are NOT authorized here.
"""
from __future__ import annotations

import torch
from torch import nn

from tam_research.cpw_binding_placement.model import EarlyAFMResearchLM
from tam_research.cpw_r1.model import CPWR1ResearchLM
from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.cpw_v5_afm.model import AFMCPWResearchLM
from tam_research.cpw_v5_afm.train import query_logits as frozen_query_logits
from tam_research.models import ModelConfig, ResearchLM, parameter_count

ARMS = ("transformer", "sequence_only", "afm_last1", "afm_first1", "r1_final")
PARAMETERS = {
    "transformer": 24_940_288,
    "sequence_only": 21_745_408,
    "afm_last1": 21_721_344,
    "afm_first1": 21_721_344,
    "r1_final": 21_745_408,
}


def build_model(arm: str) -> nn.Module:
    """Construct exact existing architectures; do not edit frozen predecessors."""
    if arm == "transformer":
        model = ResearchLM(ModelConfig(architecture="transformer"))
    elif arm == "sequence_only":
        model = SparseWorldCPWResearchLM("sequence_only", CPWV1Config())
    elif arm == "afm_last1":
        model = AFMCPWResearchLM(CPWV1Config())
    elif arm == "afm_first1":
        model = EarlyAFMResearchLM(CPWV1Config())
    elif arm == "r1_final":
        model = CPWR1ResearchLM(CPWV1Config())
    else:
        raise ValueError(f"unknown corrected-binding arm: {arm}")
    actual = parameter_count(model)
    if actual != PARAMETERS[arm]:
        raise RuntimeError(f"parameter drift for {arm}: {actual} != {PARAMETERS[arm]}")
    return model


def query_logits(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Score final queried key only. No batch metadata reaches any model."""
    if isinstance(model, CPWR1ResearchLM):
        return model.query_logits(tokens)
    if isinstance(model, (ResearchLM, SparseWorldCPWResearchLM, AFMCPWResearchLM)):
        return frozen_query_logits(model, tokens)
    raise TypeError(f"unknown corrected-binding model: {type(model).__name__}")
