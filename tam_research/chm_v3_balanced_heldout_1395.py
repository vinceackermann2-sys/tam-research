from __future__ import annotations

"""#1395 new source-locked, CPU-only balanced-schedule heldout scorer.

Do not run this module's main on PR CI, unmerged heads, or old heldout
entity index 0. One original merge-only Actions execution is intended.
The 100M DAEC Stage-C STOP and consumed seed are immutable.
"""

from collections import Counter
import hashlib
import json
import os
import time

import torch

from .chm_v3_balanced_train_schedule_1391 import (
    TRAIN_STEPS, FRESH_DEVELOPMENT_ENTITY_INDEX, FRESH_TEST_ENTITY_INDEX,
    train_balanced_cpu_models, audit_training_balance,
)
from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from .chm_v3_counterfactual_model_view_1365 import (
    PairedEpisode, STALE_CODE, oracle_read, seal_model_view,
)
from .chm_v3_counterfactual_multiset_suite_1373 import (
    FAMILIES, POSITIVES, paired_group, control_report,
)
from .chm_v3_matched_tiny_models_1377 import (
    encode_sealed_view, predict_from_sealed_view, trainable_parameter_count,
)

REPORT_MARKER = "CHM_V3_1395_BALANCED_CPU_HELDOUT_RESULT="
CLASSIFICATION = "CHM_V3_1395_BALANCED_CPU_SYNTHETIC_EXPLORATORY"
ARMS = ("transformer", "pointer_ce", "pointer_no_ce")
EVAL_ENTITIES = {
    "development": FRESH_DEVELOPMENT_ENTITY_INDEX,
    "test": FRESH_TEST_ENTITY_INDEX,
}
HISTORICAL_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
HISTORICAL_SEED_CONSUMED = 2013161
SOURCE_BLOBS = {
    "model": "ffe14b0701e18493a2bf9b45a1238b5acd5e3eab",
    "balanced_schedule": "6d40dafc7a487cfc8d5a39d5067ff00190e60181",
    "benchmark": "1b71eede754f7396ad2e7284693d6d5c1b8273b4",
    "sealed_view": "5bce89954d7a4fd788d1d1bdd972abf744399715",
}
# Source-lock strings are attestations for the workflow to verify via git,
# not substitutes for those independent checks.


def heldout_cases(split: str) -> tuple[PairedEpisode, ...]:
    if split not in EVAL_ENTITIES:
        raise ValueError("only independently frozen development/test splits")
    entity = EVAL_ENTITIES[split]
    if entity == 0 or split == "train":
        raise ValueError("do not reuse originally scored entity")
    episodes = tuple(
        ep for family in FAMILIES
        for ep in paired_group(split, family, entity, memory_length=128)
    )
    if len(episodes) != 32 or Counter(ep.family for ep in episodes) != {
        name: 8 for name in FAMILIES
    }:
        raise RuntimeError("heldout case geometry drift")
    return episodes


def _gold_indices(ep: PairedEpisode) -> tuple[int | None, int | None, bool]:
    """Evaluator only: never pass gold/pointers to the forward method."""
    code, relation_line, answer_line = oracle_read(ep)
    if code != ep.gold_answer:
        raise RuntimeError("oracle disagreement")
    if answer_line is None:
        return None, None, False
    encoded = encode_sealed_view(seal_model_view(ep))
    anchors = dict(encoded.line_anchors)
    first_line = relation_line if relation_line is not None else answer_line
    start = anchors[answer_line]
    end = min(
        (p for _, p in encoded.line_anchors if p > start),
        default=encoded.memory_length,
    )
    matches = [
        p for p, observed_code in encoded.code_tokens
        if start <= p < end and observed_code == ep.gold_answer
    ]
    if len(matches) != 1:
        raise RuntimeError("authoritative answer not uniquely copyable")
    return anchors[first_line], matches[0], True


def score_one(model: torch.nn.Module, ep: PairedEpisode, arm: str) -> dict[str, object]:
    if arm not in ARMS or ep.split not in EVAL_ENTITIES and ep.split != "train":
        raise ValueError("unknown arm or split")
    # CRITICAL: Inference is performed before the evaluator inspects gold.
    prediction = predict_from_sealed_view(model, seal_model_view(ep))
    gold_first, gold_second, copyable = _gold_indices(ep)
    chosen_first = prediction["first_selected_visible_token_index"]
    chosen_second = prediction["second_selected_visible_token_index"]
    stale_positions = set()
    if ep.family == "overwrite":
        encoded = encode_sealed_view(seal_model_view(ep))
        stale_positions = {p for p, code in encoded.code_tokens if code == STALE_CODE}
    return {
        "split": ep.split, "arm": arm, "family": ep.family,
        "entity": ep.entity, "variant": ep.variant,
        "gold_answer_id_evaluator_only": ep.gold_answer,
        "predicted_answer_id": prediction["predicted_answer_id"],
        "answer_correct": prediction["predicted_answer_id"] == ep.gold_answer,
        "confidence": prediction["confidence"],
        "abstained": prediction["abstained"],
        "first_selected_visible_token_index": chosen_first,
        "second_selected_visible_token_index": chosen_second,
        "gold_first_visible_token_index_evaluator_only": gold_first,
        "gold_second_visible_token_index_evaluator_only": gold_second,
        "answer_copyable_from_memory": copyable,
        "first_index_match": None if gold_first is None or arm == "transformer"
            else chosen_first == gold_first,
        "second_index_match": None if gold_second is None or arm == "transformer"
            else chosen_second == gold_second,
        "second_index_points_to_stale_token": None if arm == "transformer"
            else chosen_second in stale_positions,
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 32:
        raise ValueError("need 32 predictions per split and arm")
    result = {}
    for family in FAMILIES:
        part = [r for r in rows if r["family"] == family]
        if len(part) != 8 or {r["variant"] for r in part} != set(range(8)):
            raise ValueError("incomplete or duplicate eight-way family")
        predicted = [r["predicted_answer_id"] for r in part]
        correct = sum(r["answer_correct"] is True for r in part)
        family_result = {
            "cases": 8, "correct": correct, "accuracy": correct / 8,
            "distinct_predicted_answers_or_abstentions": len(set(predicted)),
            "independent_entity_groups": 1, "confidence_interval_valid": False,
            "abstentions": sum(r["abstained"] is True for r in part),
            "first_pointer_exact": sum(r["first_index_match"] is True for r in part),
            "second_pointer_exact": sum(r["second_index_match"] is True for r in part),
            "second_pointer_stale": sum(r["second_index_points_to_stale_token"] is True for r in part),
        }
        if family in POSITIVES:
            if {r["gold_answer_id_evaluator_only"] for r in part} != set(ANSWER_IDS):
                raise RuntimeError("unbalanced heldout answer rotation")
            family_result["counterfactual_answer_following"] = correct / 8
        else:
            if any(r["gold_answer_id_evaluator_only"] is not None for r in part):
                raise RuntimeError("negative example has a gold answer")
            family_result["negative_false_positives"] = sum(p is not None for p in predicted)
        result[family] = family_result
    return {
        "cases": 32, "accuracy": sum(r["answer_correct"] is True for r in rows) / 32,
        "families": result,
    }


def evaluate_models(
    models: tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module],
    metadata: dict[str, object],
    train_seconds: float,
) -> dict[str, object]:
    if len(models) != 3 or metadata.get("steps_per_arm") != TRAIN_STEPS:
        raise RuntimeError("source-locked run requires full 256-step training")
    if not all(next(m.parameters()).device.type == "cpu" for m in models):
        raise RuntimeError("GPU use not permitted")
    params = {arm: trainable_parameter_count(m) for arm, m in zip(ARMS, models)}
    if params["pointer_ce"] != params["pointer_no_ce"]:
        raise RuntimeError("pointer ablation parameter mismatch")
    if abs(params["pointer_ce"] - params["transformer"]) / max(params.values()) > 0.01:
        raise RuntimeError("parameter matching >1%")
    panels, times, rows = {}, {}, []
    for split in EVAL_ENTITIES:
        examples = heldout_cases(split)
        panels[split] = {}
        for arm, model in zip(ARMS, models):
            model.eval()
            started = time.perf_counter()
            with torch.no_grad():
                scored = [score_one(model, ep, arm) for ep in examples]
            times[f"{split}/{arm}"] = time.perf_counter() - started
            rows.extend(scored)
            panels[split][arm] = aggregate(scored)
    controls = {split: control_report(split) for split in EVAL_ENTITIES}
    for split, values in controls.items():
        for family in POSITIVES:
            for key in (
                "query_only_accuracy", "layout_only_accuracy", "bag_of_codes_only_accuracy",
            ):
                if values["families"][family][key] != 0.125:
                    raise RuntimeError("negative-control shortcut guard failed")
    if len(rows) != 192 or len({
        (r["split"], r["arm"], r["family"], r["variant"]) for r in rows
    }) != 192:
        raise RuntimeError("duplicate or missing heldout predictions")
    report = {
        "classification": CLASSIFICATION,
        "issue": 1395,
        "source_commit": os.environ.get("GITHUB_SHA", "local-not-scored"),
        "frozen_input_blob_ids": SOURCE_BLOBS,
        "historical_scientific_classification": HISTORICAL_STOP,
        "historical_scientific_seed_consumed": HISTORICAL_SEED_CONSUMED,
        "historical_scientific_parity_checkpoint_proof": "missing",
        "gpu_used": False, "modal_used": False, "new_scientific_attempt": False,
        "breakthrough_claim": False,
        "training": metadata,
        "training_steps_per_arm": TRAIN_STEPS,
        "training_tokens_per_arm": metadata["tokens_per_arm"],
        "parameters": params,
        "total_cpu_train_seconds_all_arms": train_seconds,
        "cpu_eval_seconds_by_split_and_arm": times,
        "evaluation_entity_index": EVAL_ENTITIES,
        "scores": panels, "negative_controls": controls,
        "per_case_predictions": rows,
        "interpretation_ceiling": (
            "Synthetic 1 entity per family/split; no valid confidence intervals. "
            "Hashed whole-identifier train/test OOD; unequal pointer supervision/FLOPs. "
            "Soft-conditioned second hop; hard indices are diagnostics only, "
            "not sequential hard-read evaluation. No 100M/LM breakthrough claim."
        ),
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["sha256_before_digest_field"] = hashlib.sha256(raw.encode()).hexdigest()
    return report


def main() -> None:
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("CPU thread cap failed")
    audit_training_balance()
    started = time.perf_counter()
    a, b, c, metadata = train_balanced_cpu_models(steps=TRAIN_STEPS)
    seconds = time.perf_counter() - started
    report = evaluate_models((a, b, c), metadata, seconds)
    print(REPORT_MARKER + json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
