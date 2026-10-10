from __future__ import annotations

"""#1415 one-shot CPU-only alias-vs-whole-hash four-arm synthetic score.

Original 100M CHM-v3 STOP and seed 2013161 remain immutable.
Only unique formerly UNSCORED dev/test entity index 3 can be evaluated.
NEVER rerun this entrypoint after original main push; no PR CI scoring.
"""

from collections import Counter
import json
import math
import time

import torch

from .chm_v3_alias_tiny_adapters_1410 import (
    _encode_alias_view, matched_original_and_alias_models,
)
from .chm_v3_balanced_train_schedule_1391 import (
    TRAIN_STEPS, audit_training_balance, balanced_training_episode,
)
from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from .chm_v3_counterfactual_multiset_suite_1373 import (
    FAMILIES, POSITIVES, generate_episode, model_code_bag,
)
from .chm_v3_counterfactual_model_view_1365 import (
    PairedEpisode, redact_memory_values, seal_model_view,
)
from .chm_v3_matched_tiny_models_1377 import (
    encode_sealed_view, learning_objective,
    predict_from_sealed_view, trainable_parameter_count,
)
from scripts.chm_v3_matched_tiny_one_shot_cpu_report_1377 import (
    evaluator_held_gold_indices,
)

REPORT_MARKER = "CHM_V3_1415_ALIAS_VS_HASH_ONE_SHOT="
CLASSIFICATION = "CHM_V3_1415_ALIAS_VS_HASH_CPU_EXPLORATORY"
ARMS = ("whole_decoder", "alias_decoder", "whole_pointer_ce", "alias_pointer_ce")
SPLITS = ("development", "test")
HELDOUT_ENTITY_INDEX = 3
CONSUMED_OLD_DEVELOPMENT = (0, 1)
CONSUMED_OLD_TEST = (0, 2)
PRIOR_CONFUNDED_RUN = 38064856253
PRIOR_BALANCED_RUN = 38066888678
STAGE_C_STOP = "CHM_V3_100M_DAEC_STAGE_C_STOP"
GPU_AUTHORIZED = False
NEW_SCIENTIFIC_ATTEMPT = False


def fresh_unscored_panel(split: str) -> tuple[PairedEpisode, ...]:
    """Only the prospectively frozen unused split-specific entity index 3."""
    if split not in SPLITS:
        raise ValueError("only frozen development/test split is allowed")
    if HELDOUT_ENTITY_INDEX in (CONSUMED_OLD_DEVELOPMENT
                                if split == "development" else CONSUMED_OLD_TEST):
        raise RuntimeError("proposed heldout entity was previously scored")
    return tuple(generate_episode(split, family, HELDOUT_ENTITY_INDEX, variant)
                 for family in FAMILIES for variant in range(8))


def analytic_blind_controls(episodes: tuple[PairedEpisode, ...]) -> dict[str, object]:
    """Only evaluates input invariance; does NOT score old consumed panels."""
    by_family: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        group = [x for x in episodes if x.family == family]
        if len(group) != 8 or {x.variant for x in group} != set(range(8)):
            raise RuntimeError("incomplete eight-way matched binding group")
        views = [seal_model_view(x) for x in group]
        if len({v.query for v in views}) != 1:
            raise RuntimeError("query-text shortcut")
        if len({redact_memory_values(v) for v in views}) != 1:
            raise RuntimeError("redacted layout depends on answer rotation")
        if len({model_code_bag(v) for v in views}) != 1:
            raise RuntimeError("global code bag depends on answer rotation")
        if family in POSITIVES:
            if {x.gold_answer for x in group} != set(ANSWER_IDS):
                raise RuntimeError("positive correct answers not 8-way balanced")
            expected_accuracy = 0.125
        else:
            if any(x.gold_answer is not None for x in group):
                raise RuntimeError("no-match group unexpectedly answerable")
            expected_accuracy = None
        by_family[family] = {
            "eight_way_complete": True,
            "query_only_upper_bound_accuracy": expected_accuracy,
            "redacted_layout_only_upper_bound_accuracy": expected_accuracy,
            "bag_of_codes_only_upper_bound_accuracy": expected_accuracy,
            "always_abstain_correct": len(group) if family == "no_match" else 0,
        }
    return by_family


def _score_one(model: torch.nn.Module, episode: PairedEpisode, arm: str
              ) -> dict[str, object]:
    """Gold is inaccessible until AFTER model returns its sealed-text prediction."""
    view = seal_model_view(episode)
    prediction = predict_from_sealed_view(model, view)
    gold_first, gold_second = evaluator_held_gold_indices(episode)
    is_pointer = arm.endswith("pointer_ce")
    return {
        "arm": arm, "split": episode.split, "family": episode.family,
        "entity": episode.entity, "counterfactual_variant": episode.variant,
        "predicted_answer_id": prediction["predicted_answer_id"],
        "abstained": prediction["abstained"],
        "confidence": prediction["confidence"],
        "first_model_token_index": prediction["first_selected_visible_token_index"],
        "second_model_token_index": prediction["second_selected_visible_token_index"],
        "gold_answer_id_evaluator_only": episode.gold_answer,
        "gold_first_token_index_evaluator_only": gold_first,
        "gold_second_token_index_evaluator_only": gold_second,
        "correct_answer": prediction["predicted_answer_id"] == episode.gold_answer,
        "first_index_exact_match": (
            (prediction["first_selected_visible_token_index"] == gold_first)
            if is_pointer and gold_first is not None else None
        ),
        "second_index_exact_match": (
            (prediction["second_selected_visible_token_index"] == gold_second)
            if is_pointer and gold_second is not None else None
        ),
    }


def summarize_arm(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 32:
        raise ValueError("exactly 32 heldout case rows per arm required")
    report: dict[str, object] = {}
    for family in FAMILIES:
        subset = [row for row in rows if row["family"] == family]
        if len(subset) != 8 or {
            row["counterfactual_variant"] for row in subset
        } != set(range(8)):
            raise ValueError("eight-case family evaluation incomplete")
        n_correct = sum(row["correct_answer"] is True for row in subset)
        gold_ids = {row["gold_answer_id_evaluator_only"] for row in subset}
        result = {
            "cases": 8, "correct": n_correct,
            "accuracy": n_correct / 8,
            "unique_predicted_codes_or_abstention": len({
                row["predicted_answer_id"] for row in subset
            }),
            "abstentions": sum(row["abstained"] is True for row in subset),
            "independent_entity_groups": 1,
            "confidence_interval_justified": False,
        }
        if family in POSITIVES:
            if gold_ids != set(ANSWER_IDS):
                raise ValueError("heldout answer rotation broken")
            result["answer_flip_following_fraction"] = n_correct / 8
        else:
            result["no_match_false_positive_count"] = sum(
                row["predicted_answer_id"] is not None for row in subset
            )
        first = [row["first_index_exact_match"] for row in subset
                 if row["first_index_exact_match"] is not None]
        second = [row["second_index_exact_match"] for row in subset
                  if row["second_index_exact_match"] is not None]
        result["first_diagnostic_index_hits"] = (
            sum(x is True for x in first) if first else None
        )
        result["second_diagnostic_index_hits"] = (
            sum(x is True for x in second) if second else None
        )
        report[family] = result
    return {
        "cases": 32,
        "overall_accuracy": sum(row["correct_answer"] is True for row in rows) / 32,
        "positive_memory_accuracy": sum(row["correct_answer"] is True for row in rows
                                        if row["family"] in POSITIVES) / 24,
        "per_family": report,
    }


def _record_cpu_encoder_overhead_on_train() -> dict[str, float | int]:
    """Timing on TRAIN views only, separate from model/optimizer work."""
    views = [seal_model_view(balanced_training_episode(step))
             for step in range(TRAIN_STEPS)]
    baseline_tokens = alias_tokens = 0
    t0 = time.perf_counter()
    for view in views:
        baseline_tokens += len(encode_sealed_view(view).token_ids)
    t1 = time.perf_counter()
    for view in views:
        alias_tokens += len(_encode_alias_view(view).token_ids)
    t2 = time.perf_counter()
    if baseline_tokens != alias_tokens:
        raise RuntimeError("different input-token totals between encoders")
    return {
        "train_views": TRAIN_STEPS,
        "shared_token_count": baseline_tokens,
        "whole_encoder_cpu_seconds": t1 - t0,
        "alias_encoder_cpu_seconds_including_legacy_parity_validation": t2 - t1,
    }


def main() -> None:
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("single-thread CPU budget violated")
    train_audit = audit_training_balance()
    if (TRAIN_STEPS != 256 or train_audit["steps_per_arm"] != 256 or
        not train_audit["all_positive_groups_cover_all_eight_answers"]):
        raise RuntimeError("original frozen balanced training schedule violated")
    models = matched_original_and_alias_models()
    if tuple(models) != ARMS:
        raise RuntimeError("arm initialization order changed")
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.)
        for name, model in models.items()
    }
    counts = {name: trainable_parameter_count(model)
              for name, model in models.items()}
    if counts["whole_decoder"] != counts["alias_decoder"]:
        raise RuntimeError("decoder parameter parity broken")
    if counts["whole_pointer_ce"] != counts["alias_pointer_ce"]:
        raise RuntimeError("pointer parameter parity broken")
    if abs(counts["whole_decoder"] - counts["whole_pointer_ce"]) / max(
        counts.values()
    ) > .01:
        raise RuntimeError("tiny 1% parameter gate changed")
    encoder_timing = _record_cpu_encoder_overhead_on_train()
    train_tokens = 0
    train_seconds = {arm: 0.0 for arm in ARMS}
    for step in range(TRAIN_STEPS):
        episode = balanced_training_episode(step)
        if episode.split != "train":
            raise RuntimeError("train-only split guard")
        train_tokens += len(encode_sealed_view(seal_model_view(episode)).token_ids)
        for arm, model in models.items():
            started = time.perf_counter()
            optimizers[arm].zero_grad(set_to_none=True)
            loss, _ = learning_objective(
                model, episode, pointer_supervision=arm.endswith("pointer_ce"),
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite balanced CPU loss")
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all().item())
                       for p in model.parameters()):
                raise FloatingPointError("non-finite CPU gradient")
            optimizers[arm].step()
            train_seconds[arm] += time.perf_counter() - started
    for model in models.values():
        model.eval()

    scores: dict[str, object] = {}
    blind_controls: dict[str, object] = {}
    predictions: list[dict[str, object]] = []
    eval_seconds: dict[str, float] = {}
    for split in SPLITS:
        episodes = fresh_unscored_panel(split)
        if len(episodes) != 32:
            raise RuntimeError("fresh heldout panel not 32 cases")
        blind_controls[split] = analytic_blind_controls(episodes)
        scores[split] = {}
        for arm, model in models.items():
            started = time.perf_counter()
            rows = [_score_one(model, episode, arm) for episode in episodes]
            eval_seconds[f"{split}/{arm}"] = time.perf_counter() - started
            scores[split][arm] = summarize_arm(rows)
            predictions.extend(rows)
    if len(predictions) != 4 * 2 * 32 or len({
        (x["arm"], x["split"], x["family"], x["counterfactual_variant"])
        for x in predictions
    }) != 256:
        raise RuntimeError("missing/duplicate source-locked heldout predictions")
    result = {
        "classification": CLASSIFICATION,
        "issue": 1415,
        "historical_100m_scientific_status": STAGE_C_STOP,
        "historical_scientific_seed_2013161_consumed": True,
        "prior_confounded_48_step_run_not_replayed": PRIOR_CONFUNDED_RUN,
        "prior_balanced_256_step_run_not_replayed": PRIOR_BALANCED_RUN,
        "new_scientific_attempt": NEW_SCIENTIFIC_ATTEMPT,
        "gpu_allocated": False, "modal_used": False,
        "breakthrough_claim": False,
        "cpu_threads": torch.get_num_threads(),
        "train_steps_per_arm": TRAIN_STEPS,
        "train_example_count_per_arm": TRAIN_STEPS,
        "training_tokens_per_arm": train_tokens,
        "train_audit": train_audit,
        "parameters": counts,
        "train_seconds_by_arm": train_seconds,
        "encoder_cpu_timing": encoder_timing,
        "eval_seconds_by_split_arm": eval_seconds,
        "heldout_entity_by_split": {
            "development": HELDOUT_ENTITY_INDEX,
            "test": HELDOUT_ENTITY_INDEX,
        },
        "scores": scores, "analytic_memory_blind_controls": blind_controls,
        "predictions": predictions,
        "interpretation_limit": (
            "Only one independent entity per family and split: no CI or "
            "statistical superiority. Alias exposes visible query equality as "
            "explicit input inductive bias; this is NOT an architectural "
            "Transformer-versus-CPW comparison. Pointer has extra TRAIN-only "
            "gold source labels and different FLOPs vs decoder, while the "
            "second pointer hop is SOFT-conditioned and hard index argmax is "
            "diagnostic only. Original full-name hash has unseen heldout "
            "identifier token embeddings. Synthetic toy, not general language, "
            "100M improvement or scientific breakthrough."
        ),
    }
    print(REPORT_MARKER + json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
