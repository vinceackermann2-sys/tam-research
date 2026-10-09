from __future__ import annotations

"""#1357 read-only CPU research panel. No scientific training or GPU authority."""

import json

from tam_research.chm_v3_daec_a_role_ablation_1357 import (
    HISTORICAL_SCIENTIFIC_SEED_CONSUMED,
    MEMORY_LENGTHS, MODES, PAID_GPU_AUTHORIZED,
    SPLIT_SEEDS, TRAIN_STEPS,
    evaluate_panel, train_arms,
)


def main() -> None:
    if PAID_GPU_AUTHORIZED or HISTORICAL_SCIENTIFIC_SEED_CONSUMED != 2_013_161:
        raise RuntimeError("#1357 CPU-only preregistration violated")
    arms, training = train_arms()
    panels = []
    for split in ("development", "test"):
        for length in MEMORY_LENGTHS:
            for mode in MODES:
                panels.append(evaluate_panel(arms[mode],split,length))
    report = {
        "classification":"CHM_V3_1357_SYNTHETIC_CPU_ROLE_SOFT_HARD_ABLATION",
        "issue":1357,
        "historical_scientific_classification":"CHM_V3_100M_DAEC_STAGE_C_STOP",
        "historical_scientific_seed_consumed":True,
        "gpu_allocated":False,"modal_used":False,"scientific_quality_claim":False,
        "human_language_model_evaluated":False,"matched_transformer_training":False,
        "precommitted_splits":SPLIT_SEEDS,"training_steps_per_arm":TRAIN_STEPS,
        "oracle_role_mask_only_in_arm_A":True,
        "learned_role_arm_receives_explicit_synthetic_role_cues":True,
        "learned_role_arm_receives_oracle_role_mask_at_inference":False,
        "soft_arm_uses_teacher_forced_first_hop":False,
        "pointer_ce_arms_teacher_forced_first_hop_training_only":True,
        "training":training,
        "held_out_panels":panels,
        "paid_science_authorized":False,
        "replication_authorized":False,"stage_d_authorized":False,
    }
    print("CHM_V3_1357_CPU_ABLATION_REPORT="+json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
