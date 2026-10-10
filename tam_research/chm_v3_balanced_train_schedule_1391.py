from __future__ import annotations

"""#1391 Stage-A: CPU-only counterfactually balanced TRAIN scheduler.

This is a NEW experiment, not a rerun/amendment of #1377's first CPU report.
The previously merged tiny Transformer/pointer architecture, tokenizer and
multiset-preserving benchmark stay byte-for-byte unchanged. Old source
training_episode(step) had exact positive family->answer leakage: DO NOT use.
"""

from collections import Counter
from functools import lru_cache
import random

import torch

from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS, SPLITS
from .chm_v3_counterfactual_multiset_suite_1373 import (
    FAMILIES, POSITIVES, generate_episode,
)
from .chm_v3_matched_tiny_models_1377 import (
    TinyDecoderOnly, TinyTwoHopPointer, matched_initial_models,
    learning_objective, trainable_parameter_count, encode_sealed_view,
)
from .chm_v3_counterfactual_model_view_1365 import (
    PairedEpisode, seal_model_view,
)

BALANCED_SCHEDULE_SEED = 13_772_001
TRAIN_STEPS = SPLITS["train"] * len(FAMILIES) * len(ANSWER_IDS)
assert TRAIN_STEPS == 256
TRAIN_MEMORY_LENGTH = 128
FRESH_DEVELOPMENT_ENTITY_INDEX = 1
FRESH_TEST_ENTITY_INDEX = 2
ORIGINAL_SCIENTIFIC_SEED_CONSUMED = 2_013_161
ORIGINAL_SCIENTIFIC_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
GPU_AUTHORIZED = False
SCIENTIFIC_RUN_AUTHORIZED = False
NEVER_REUSE_OLD_REPORT_TEST_ENTITY_INDEX = 0


@lru_cache(maxsize=1)
def balanced_train_schedule() -> tuple[tuple[str, int, int], ...]:
    """Exactly 8 rotations for every (family, train entity), then shuffle.

    The permutation is frozen using an entirely new CPU-only RNG seed; neither
    split nor labels are ever passed to a model's forward() method.
    """
    records = [
        (family, entity, variant)
        for family in FAMILIES
        for entity in range(SPLITS["train"])
        for variant in range(len(ANSWER_IDS))
    ]
    if len(records) != 256:
        raise RuntimeError("balanced training size not 256")
    random.Random(BALANCED_SCHEDULE_SEED).shuffle(records)
    if len(set(records)) != 256:
        raise RuntimeError("training schedule contains repeats")
    return tuple(records)


def balanced_training_episode(step: int) -> PairedEpisode:
    if type(step) is not int or not 0 <= step < TRAIN_STEPS:
        raise ValueError("balanced training index outside frozen 256-step range")
    family, entity_index, variant = balanced_train_schedule()[step]
    return generate_episode(
        "train", family, entity_index, variant,
        memory_length=TRAIN_MEMORY_LENGTH,
    )


def audit_training_balance() -> dict[str, object]:
    """Data-only proof; does not inspect development/test or run any model."""
    observations: dict[tuple[str, int], list[int | None]] = {}
    counts: Counter[str] = Counter()
    for step in range(TRAIN_STEPS):
        ep = balanced_training_episode(step)
        if ep.split != "train":
            raise RuntimeError("training data split leakage")
        counts[ep.family] += 1
        observations.setdefault((ep.family, int(ep.entity[-3:])), []).append(
            ep.gold_answer
        )
    if counts != {family: 64 for family in FAMILIES}:
        raise RuntimeError("family training balance failed")
    for (family, _), answers in observations.items():
        if len(answers) != 8:
            raise RuntimeError("not exactly eight variants per entity/family")
        if family in POSITIVES:
            if set(answers) != set(ANSWER_IDS):
                raise RuntimeError("candidate answer imbalance in positive group")
        elif set(answers) != {None}:
            raise RuntimeError("negative labels must always require abstention")
    return {
        "classification": "CHM_V3_1391_COUNTERFACTUALLY_BALANCED_TRAIN_CPU_ONLY",
        "steps_per_arm": TRAIN_STEPS,
        "family_examples": dict(counts),
        "train_entities": SPLITS["train"],
        "positive_groups": SPLITS["train"] * len(POSITIVES),
        "all_positive_groups_cover_all_eight_answers": True,
        "development_entity_reserved": FRESH_DEVELOPMENT_ENTITY_INDEX,
        "test_entity_reserved": FRESH_TEST_ENTITY_INDEX,
        "old_entity_index_zero_not_reused_in_scored_followup": True,
        "gpu_allocated": False,
        "new_scientific_attempt": False,
    }


def train_balanced_cpu_models(
    *, steps: int = 1,
) -> tuple[
    TinyDecoderOnly,
    TinyTwoHopPointer,
    TinyTwoHopPointer,
    dict[str, object],
]:
    """Up to 256 identical CPU training episodes and optimizer steps per arm.

    This source has no heldout scoring function or GPU. Run 256 steps only in a
    future once-merged, separately preregistered one-shot scored workflow.
    """
    if type(steps) is not int or not 1 <= steps <= TRAIN_STEPS:
        raise ValueError("balanced CPU training steps outside 1..256 cap")
    torch.set_num_threads(1)
    models = matched_initial_models()
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=0.005, weight_decay=0.0)
        for m in models
    ]
    tokens = 0
    for step in range(steps):
        episode = balanced_training_episode(step)
        tokens += len(encode_sealed_view(seal_model_view(episode)).token_ids)
        for arm, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = learning_objective(
                model, episode, pointer_supervision=(arm == 1),
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("nonfinite balanced CPU training loss")
            loss.backward()
            optimizer.step()
    for model in models:
        model.eval()
    return (*models, {
        "classification": "CHM_V3_1391_BALANCED_TINY_CPU_TRAINING_ONLY",
        "steps_per_arm": steps, "examples_per_arm": steps,
        "tokens_per_arm": tokens,
        "trainable_parameters": {
            name: trainable_parameter_count(model)
            for name, model in zip(
                ("transformer", "pointer_ce", "pointer_no_ce"), models,
            )
        },
        "optimizer": "AdamW(lr=0.005,weight_decay=0.0)",
        "historical_scientific_status": ORIGINAL_SCIENTIFIC_STOP,
        "historical_scientific_seed_consumed": True,
        "cpu_only": True, "new_scientific_attempt": False,
    })


__all__ = (
    "balanced_train_schedule", "balanced_training_episode",
    "audit_training_balance", "train_balanced_cpu_models",
    "TRAIN_STEPS", "BALANCED_SCHEDULE_SEED",
    "FRESH_DEVELOPMENT_ENTITY_INDEX", "FRESH_TEST_ENTITY_INDEX",
)
