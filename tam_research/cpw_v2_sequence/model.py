from __future__ import annotations

from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v2_single.model import SinglePredictorCPWResearchLM
from tam_research.models import parameter_count


class CPWV2SequenceLM(SinglePredictorCPWResearchLM):
    """Frozen physical sequence-only CPW-v2 candidate from issue #1253."""

    def __init__(self, cfg: CPWV1Config = CPWV1Config()):
        super().__init__("sequence_only", cfg)


def sequence_parameter_count() -> int:
    return parameter_count(CPWV2SequenceLM())
