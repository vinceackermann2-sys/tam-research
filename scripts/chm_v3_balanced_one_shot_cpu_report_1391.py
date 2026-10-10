from __future__ import annotations

"""#1391 new one-shot CPU-only balanced TRAIN / fresh heldout score.

Never replay the frozen #1377 original run and never evaluate its old heldout
entity index 0. Exactly one source-locked path-gated main workflow owns this.
"""

import json
import time

import torch

from tam_research.chm_v3_balanced_train_schedule_1391 import (
    TRAIN_STEPS, FRESH_DEVELOPMENT_ENTITY_INDEX, FRESH_TEST_ENTITY_INDEX,
    balanced_training_episode, audit_training_balance, train_balanced_cpu_models,
)
from tam_research.chm_v3_matched_tiny_models_1377 import (
    learning_objective, trainable_parameter_count,
)
from tam_research.chm_v3_counterfactual_multiset_suite_1373 import (
    FAMILIES, POSITIVES, generate_episode, control_report,
)
from scripts.chm_v3_matched_tiny_one_shot_cpu_report_1377 import (
    _score_one, _aggregate,
)

REPORT_MARKER = "CHM_V3_1391_BALANCED_TINY_CPU_REPORT="
ORIGINAL_CONFUNDED_RUN = 38064856253
SOURCE_SCIENCE_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
ARMS = ("transformer", "pointer_ce", "pointer_no_ce")
EVAL_ENTITY_INDICES = {
    "development": FRESH_DEVELOPMENT_ENTITY_INDEX,
    "test": FRESH_TEST_ENTITY_INDEX,
}
GPU_AUTHORIZED = False
NEW_SCIENTIFIC_RUN = False


def fresh_eval_cases(split: str):
    if split not in EVAL_ENTITY_INDICES:
        raise ValueError("only preregistered fresh dev/test splits allowed")
    index = EVAL_ENTITY_INDICES[split]
    if index == 0:
        raise ValueError("original scored heldout entity 0 is permanently off limits")
    return tuple(
        generate_episode(split, family, index, variant)
        for family in FAMILIES
        for variant in range(8)
    )


def _one_nonupdating_cpu_fwb(model: torch.nn.Module, arm: str) -> float:
    """One TRAIN-split-only positive two-hop FWB profile, no optimizer step."""
    sample = next(
        balanced_training_episode(i)
        for i in range(TRAIN_STEPS)
        if balanced_training_episode(i).family == "two_hop"
    )
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    loss, _ = learning_objective(model, sample,
                                 pointer_supervision=(arm == "pointer_ce"))
    loss.backward()
    elapsed = time.perf_counter() - started
    model.zero_grad(set_to_none=True)
    return elapsed


def main() -> None:
    torch.set_num_threads(1)
    training_balance = audit_training_balance()
    if training_balance["steps_per_arm"] != 256:
        raise RuntimeError("training schedule not frozen 256-case balance")
    started_train = time.perf_counter()
    a, b, c, train_meta = train_balanced_cpu_models(steps=TRAIN_STEPS)
    total_train_secs = time.perf_counter() - started_train
    models = dict(zip(ARMS, (a,b,c)))
    params = {arm:trainable_parameter_count(model)
              for arm,model in models.items()}
    if params["pointer_ce"] != params["pointer_no_ce"]:
        raise RuntimeError("pointer controls not parameter matched")
    if abs(params["transformer"] - params["pointer_ce"]) / max(params.values()) > 0.01:
        raise RuntimeError("parameter 1% gate drift")

    nonupdating_fwb = {
        arm:_one_nonupdating_cpu_fwb(model, arm)
        for arm,model in models.items()
    }
    scores: dict[str, dict[str, object]] = {}
    predictions: list[dict[str, object]] = []
    cpu_eval_s: dict[str, float] = {}
    for split in ("development","test"):
        episodes = fresh_eval_cases(split)
        if len(episodes) != 32:
            raise RuntimeError("exact fresh 32-case panel required")
        scores[split] = {}
        for arm, model in models.items():
            model.eval()
            started = time.perf_counter()
            rows = [_score_one(model, ep, arm) for ep in episodes]
            cpu_eval_s[f"{split}/{arm}"] = time.perf_counter() - started
            scores[split][arm] = _aggregate(rows)
            predictions.extend(rows)

    controls = {split:control_report(split) for split in ("development","test")}
    for split in controls:
        for family in POSITIVES:
            for field in ("query_only_accuracy","layout_only_accuracy",
                          "bag_of_codes_only_accuracy"):
                if controls[split]["families"][family][field] != 0.125:
                    raise RuntimeError("matched-value-bag shortcut controls failed")

    report = {
        "classification": "CHM_V3_1391_BALANCED_TINY_CPU_EXPLORATORY_REPORT",
        "issue": 1391,
        "source_original_confounded_cpu_run_not_repeated": ORIGINAL_CONFUNDED_RUN,
        "original_historical_scientific_classification": SOURCE_SCIENCE_STOP,
        "original_scientific_seed_2013161_consumed": True,
        "cpu_only": True, "gpu_allocated": False,
        "modal_used": False, "paid_gpu_used": False,
        "new_scientific_attempt": False, "breakthrough_claim": False,
        "train_schedule": training_balance,
        "steps_per_arm": TRAIN_STEPS,
        "train_tokens_per_arm": train_meta["tokens_per_arm"],
        "parameters": params,
        "total_cpu_train_seconds_all_arms": total_train_secs,
        "cpu_one_nonupdating_fwb_seconds_by_arm": nonupdating_fwb,
        "cpu_eval_seconds": cpu_eval_s,
        "heldout_entity_indices": EVAL_ENTITY_INDICES,
        "scores": scores, "negative_controls": controls,
        "per_case_predictions": predictions,
        "interpretation_ceiling": (
            "Only ONE independent entity group per family and split; no "
            "confidence interval or statistical architectural superiority. "
            "Soft-conditioned pointer hops; hard argmax locations diagnostic "
            "ONLY, not sequential hard-read inference. Identical tiny "
            "architecture with changed TRAIN schedule, but extra source "
            "pointer labels and unmatched computational work for pointer-CE. "
            "Synthetic hashed whole-identifier tokenizer yields unseen "
            "dev/test entity token embeddings, an uncontrolled OOD confound. "
            "Not general-language/100M performance, not a breakthrough."
        ),
    }
    if len(predictions) != 192:
        raise RuntimeError("not all 192 fresh heldout predictions logged")
    if set(EVAL_ENTITY_INDICES.values()) != {1,2}:
        raise RuntimeError("fresh heldout index contract drift")
    print(REPORT_MARKER + json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
