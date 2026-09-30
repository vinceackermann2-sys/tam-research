from __future__ import annotations

from typing import Final

from architectures.cortex_s.reduced_attention_200m_panel_v2 import (
    ATTENTION_8,
    EXPECTED_TRANSFORMER_PARAMETERS,
    ScheduledAttentionDenseLM,
    expected_parameter_count,
)


CANDIDATE_NAME: Final = "reduced_attention_dense_v2_attention8"
ATTENTION_LAYERS_ONE_BASED: Final = ATTENTION_8.attention_layers_one_based
FF_HIDDEN: Final = ATTENTION_8.ff_hidden
EXPECTED_PARAMETERS: Final = ATTENTION_8.expected_parameters


class ReducedAttentionDenseV2Attention8LM(ScheduledAttentionDenseLM):
    """Frozen 8-attention 100M candidate selected by the 200M engineering panel."""

    def __init__(self) -> None:
        super().__init__(ATTENTION_8)


def candidate_contract() -> dict[str, object]:
    count = expected_parameter_count(ATTENTION_8)
    if count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"candidate parameter drift: {count:,} != {EXPECTED_PARAMETERS:,}"
        )
    gap = abs(count - EXPECTED_TRANSFORMER_PARAMETERS)
    if gap / EXPECTED_TRANSFORMER_PARAMETERS > 0.0001:
        raise RuntimeError("candidate parameter gap exceeds frozen tolerance")
    return {
        "candidate_name": CANDIDATE_NAME,
        "attention_layers_one_based": list(ATTENTION_LAYERS_ONE_BASED),
        "ff_hidden": FF_HIDDEN,
        "expected_parameters": EXPECTED_PARAMETERS,
        "transformer_expected_parameters": EXPECTED_TRANSFORMER_PARAMETERS,
        "parameter_gap": count - EXPECTED_TRANSFORMER_PARAMETERS,
        "world_state": False,
        "moe": False,
        "router": False,
        "scientific_training_authorized": False,
        "scientific_seed_consumed": False,
    }
