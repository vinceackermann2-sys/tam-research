from __future__ import annotations

"""#1377 exactly-once CPU-only tiny trainable-arms report (not scientific).

Frozen panels: 48 sequential training episodes per arm; development and test
each cover one entire 8-way answer rotation of entity index 0, separately
for direct/overwrite/two_hop/no_match. So this has only one independent
entity per family/split; NO inferential confidence interval is justified.

DO NOT run on source/PR CI. One push-to-main path-gated dedicated GitHub
Actions workflow runs this once and archives the complete per-case report.
"""

import hashlib
import json
import time
from collections import defaultdict

import torch

from tam_research.chm_v3_matched_tiny_models_1377 import (
    EncodedView, TinyDecoderOnly, TinyTwoHopPointer, encode_sealed_view,
    learning_objective, matched_initial_models, predict_from_sealed_view,
    train_cpu_models, training_episode, trainable_parameter_count,
)
from tam_research.chm_v3_counterfactual_model_view_1365 import (
    PairedEpisode, oracle_read, seal_model_view,
)
from tam_research.chm_v3_counterfactual_multiset_suite_1373 import (
    FAMILIES, POSITIVES, generate_episode, control_report,
)

REPORT_MARKER = "CHM_V3_1377_TINY_CPU_REPORT="
EXPECTED_STATUS = "CHM_V3_100M_DAEC_STAGE_C_STOP"
ARMS = ("transformer", "pointer_ce", "pointer_no_ce")
SPLITS = ("development", "test")
TRAIN_STEPS = 48
EVAL_ENTITY_INDEX = 0
EVAL_VARIANTS = tuple(range(8))
GPU_USED = False
NEW_SCIENTIFIC_RUN = False


def eval_cases(split: str) -> tuple[PairedEpisode, ...]:
    if split not in SPLITS:
        raise ValueError("only development/test allowed in locked panel")
    return tuple(
        generate_episode(split, family, EVAL_ENTITY_INDEX, variant)
        for family in FAMILIES for variant in EVAL_VARIANTS
    )


def evaluator_held_gold_indices(ep: PairedEpisode) -> tuple[int | None, int | None]:
    """Find source labels AFTER prediction; they never enter model.forward()."""
    _, relation_pos, answer_pos = oracle_read(ep)
    if answer_pos is None:
        return None, None
    encoded = encode_sealed_view(seal_model_view(ep))
    anchors = dict(encoded.line_anchors)
    first_phys = relation_pos if relation_pos is not None else answer_pos
    first_tok = anchors[first_phys]
    answer_start = anchors[answer_pos]
    answer_end = min(
        (start for _, start in encoded.line_anchors if start > answer_start),
        default=encoded.memory_length,
    )
    matches = [
        tok for tok, code in encoded.code_tokens
        if answer_start <= tok < answer_end and code == ep.gold_answer
    ]
    if len(matches) != 1:
        raise ValueError("heldout gold code not uniquely visible on gold record")
    return first_tok, matches[0]


def _score_one(model: torch.nn.Module, ep: PairedEpisode, arm: str) -> dict[str, object]:
    # Keep model-facing inputs sealed; read any ground truth only after forward.
    view = seal_model_view(ep)
    pred = predict_from_sealed_view(model, view)
    gold_first, gold_second = evaluator_held_gold_indices(ep)
    correct = pred["predicted_answer_id"] == ep.gold_answer
    return {
        "arm": arm, "split": ep.split, "family": ep.family,
        "entity": ep.entity, "variant": ep.variant,
        "gold_answer_id_evaluator_only": ep.gold_answer,
        "predicted_answer_id": pred["predicted_answer_id"],
        "answer_correct": bool(correct),
        "abstained": pred["abstained"],
        "confidence": pred["confidence"],
        "first_selected_visible_token_index": pred["first_selected_visible_token_index"],
        "second_selected_visible_token_index": pred["second_selected_visible_token_index"],
        "gold_first_visible_token_index_evaluator_only": gold_first,
        "gold_second_visible_token_index_evaluator_only": gold_second,
        "first_index_match": (
            None if gold_first is None or arm == "transformer"
            else pred["first_selected_visible_token_index"] == gold_first
        ),
        "second_index_match": (
            None if gold_second is None or arm == "transformer"
            else pred["second_selected_visible_token_index"] == gold_second
        ),
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    by_family: dict[str, object] = {}
    for family in FAMILIES:
        sub = [row for row in rows if row["family"] == family]
        if len(sub) != 8:
            raise ValueError("missing one complete 8-way counterfactual family")
        correct = sum(row["answer_correct"] is True for row in sub)
        predicted = [row["predicted_answer_id"] for row in sub]
        result = {
            "cases": len(sub),
            "correct": correct,
            "answer_accuracy": correct / len(sub),
            "unique_predicted_answers_or_abstentions": len(set(predicted)),
            "abstentions": sum(row["abstained"] is True for row in sub),
            "independent_entity_groups": 1,
            "confidence_interval_estimable": False,
        }
        if family in POSITIVES:
            result["per_rotation_answer_following"] = correct / len(sub)
            result["gold_answer_id_count"] = len({
                row["gold_answer_id_evaluator_only"] for row in sub
            })
        else:
            result["negative_false_positive_count"] = sum(
                row["predicted_answer_id"] is not None for row in sub
            )
        pointer_first = [r["first_index_match"] for r in sub if r["first_index_match"] is not None]
        pointer_second = [r["second_index_match"] for r in sub if r["second_index_match"] is not None]
        result["first_token_index_correct"] = sum(pointer_first) if pointer_first else None
        result["second_token_index_correct"] = sum(pointer_second) if pointer_second else None
        by_family[family] = result
    return {
        "cases": len(rows), "accuracy": sum(r["answer_correct"] for r in rows) / len(rows),
        "per_family": by_family,
    }


def _profile_one_fwb(model: torch.nn.Module, arm: str) -> float:
    """Non-updating CPU forward/backward timing on the same TRAIN episode."""
    ep = training_episode(6)  # positive two-hop; never uses dev/test
    model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    loss, _ = learning_objective(model, ep, pointer_supervision=(arm == "pointer_ce"))
    loss.backward()
    duration = time.perf_counter() - start
    model.zero_grad(set_to_none=True)
    return duration


def main() -> None:
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("CPU thread cap not respected")
    cpu_train_started = time.perf_counter()
    a, b, c, metadata = train_cpu_models(steps=TRAIN_STEPS)
    train_seconds_total = time.perf_counter() - cpu_train_started
    models = dict(zip(ARMS, (a, b, c)))
    params = {arm: trainable_parameter_count(model) for arm, model in models.items()}
    if abs(params["pointer_ce"] - params["transformer"]) / max(params.values()) > 0.01:
        raise RuntimeError("source parameter matching gate failed")

    forward_backward_times = {
        arm: _profile_one_fwb(model, arm) for arm, model in models.items()
    }
    reports: dict[str, object] = {}
    all_predictions: list[dict[str, object]] = []
    evaluate_times: dict[str, float] = {}
    for split in SPLITS:
        episodes = eval_cases(split)
        split_report: dict[str, object] = {}
        for arm, model in models.items():
            started = time.perf_counter()
            model.eval()
            rows = [_score_one(model, ep, arm) for ep in episodes]
            evaluate_times[f"{split}/{arm}"] = time.perf_counter() - started
            all_predictions.extend(rows)
            split_report[arm] = _aggregate(rows)
        reports[split] = split_report

    controls = {split: control_report(split) for split in SPLITS}
    for split in SPLITS:
        for family in POSITIVES:
            for field in ("query_only_accuracy", "layout_only_accuracy",
                          "bag_of_codes_only_accuracy"):
                if controls[split]["families"][family][field] != 0.125:
                    raise RuntimeError("frozen benchmark negative-control gate changed")

    report = {
        "classification": "CHM_V3_1377_MATCHED_TINY_CPU_EXPLORATORY_REPORT",
        "issue": 1377,
        "historical_scientific_classification": EXPECTED_STATUS,
        "historical_scientific_seed_2013161_consumed": True,
        "scientific_breakthrough_claim": False,
        "gpu_allocated": GPU_USED,
        "modal_used": False,
        "paid_compute_used": False,
        "new_scientific_attempt": NEW_SCIENTIFIC_RUN,
        "training_steps_per_arm": TRAIN_STEPS,
        "training_examples_per_arm": TRAIN_STEPS,
        "training_token_count_per_arm": metadata["training_token_count_per_arm"],
        "parameters": params,
        "total_cpu_train_seconds_all_three_arms": train_seconds_total,
        "one_additional_nonupdating_cpu_fwb_seconds_by_arm": forward_backward_times,
        "cpu_eval_seconds_by_split_and_arm": evaluate_times,
        "evaluation_panels": {
            "development": {"entity_index": EVAL_ENTITY_INDEX, "cases_per_arm": 32},
            "test": {"entity_index": EVAL_ENTITY_INDEX, "cases_per_arm": 32},
        },
        "scores": reports,
        "negative_controls": controls,
        "per_case_predictions": all_predictions,
        "interpretation_ceiling": (
            "One entity group per task family and split; no valid uncertainty CI. "
            "Soft first hop and soft-conditioned second hop; argmax indices are "
            "DIAGNOSTICS, not sequential hard-read inference. Parameter parity "
            "is NOT FLOP or training-label parity. Synthetic-only comparison, "
            "not Transformer/LLM superiority and not new 100M scientific evidence."
        ),
    }
    if len(all_predictions) != 3 * 2 * 32:
        raise RuntimeError("missing predictions for fixed heldout panels")
    print(REPORT_MARKER + json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
