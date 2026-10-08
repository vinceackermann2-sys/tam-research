from __future__ import annotations

"""#1350 deterministic CPU-only Stage-A toy metric report (never scientific)."""

import json

from tam_research.chm_v3_daec_a_address_target_toy_1350 import (
    HISTORICAL_SCIENTIFIC_SEED_CONSUMED,
    PAID_GPU_AUTHORIZED,
    score_split,
    train_cpu_pointer_toy,
)


def main() -> None:
    if PAID_GPU_AUTHORIZED or HISTORICAL_SCIENTIFIC_SEED_CONSUMED != 2_013_161:
        raise RuntimeError("#1350 must remain CPU-only and old science STOP frozen")
    baseline, supervised = train_cpu_pointer_toy(steps=96,learning_rate=.025,memory_length=128)
    scores = {}
    for split in ("development","test"):
        scores[split] = {
            "same_initialization_untrained_pointer_control": score_split(
                baseline,split,cases=64,memory_length=128,
            ),
            "pointer_supervised_two_hop": score_split(
                supervised,split,cases=64,memory_length=128,
            ),
        }
    result = {
        "classification": "CHM_V3_DAEC_A_STAGE_A_CPU_ONLY_TOY_DIAGNOSTIC",
        "issue": 1350,
        "prior_scientific_classification": "CHM_V3_100M_DAEC_STAGE_C_STOP",
        "historical_scientific_seed_2013161_consumed": True,
        "gpu_allocated": False,
        "modal_used": False,
        "100m_transformer_comparison": False,
        "trained_full_language_model": False,
        "oracle_role_masks_supplied": True,
        "teacher_forced_first_hop_training_only": True,
        "steps": 96,
        "cpu_train_episode_count": 96,
        "memory_length": 128,
        "development_and_test": scores,
        "scientific_pass_claim": False,
        "further_gpu_authority": False,
    }
    print("CHM_V3_DAEC_A_STAGE_A_CPU_TOY_REPORT="+json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
