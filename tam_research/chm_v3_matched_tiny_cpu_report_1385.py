from __future__ import annotations

"""#1385 one-shot, CPU-only synthetic #1377 matched-tiny quality report.

This is an engineering feasibility report and never a 100M scientific run.
No old checkpoint, CUDA, Modal, paid GPU, scientific seed or external datasets.
The input to every prediction is ONLY the frozen SealedModelView.
"""

from collections import Counter
import hashlib
import json
import re
import time
from typing import Any

import torch

from .chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS
from .chm_v3_counterfactual_model_view_1365 import (
    STALE_CODE, SealedModelView, oracle_read, seal_model_view,
)
from .chm_v3_counterfactual_multiset_suite_1373 import (
    FAMILIES, POSITIVES, generate_episode, model_code_bag, paired_group,
)
from .chm_v3_matched_tiny_models_1377 import (
    ORIGINAL_SCIENTIFIC_STATUS, encode_sealed_view, matched_initial_models,
    predict_from_sealed_view, train_cpu_models, train_only_targets,
)

ISSUE = 1385
PARENT_ISSUE = 1377
ARCHIVED_SCIENTIFIC_SEED_CONSUMED = 2_013_161
TRAIN_STEPS_PER_ARM = 48
HELDOUT_SPLITS = ("development", "test")
HELDOUT_ENTITY_INDEX = 0
VARIANTS = tuple(range(8))
MEMORY_LENGTH = 128
MAX_TOTAL_WALL_SECONDS = 2400
ARMS = ("transformer", "pointer_ce", "pointer_no_ce")
EXPECTED_EPISODES_PER_SPLIT = 32
CPU_ONLY = True
PAID_TRAINING_AUTHORIZED = False
SUCCESS_LABEL = "CHM_V3_1385_TINY_CPU_SYNTHETIC_UNMATCHED_COMPUTE_REPORT"


def _canonical_hash(data: object) -> str:
    content = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def frozen_heldout_groups() -> tuple[tuple[str, str, tuple], ...]:
    """Exactly four independent eight-rotation groups per held-out split."""
    items = []
    for split in HELDOUT_SPLITS:
        for family in FAMILIES:
            group = paired_group(split, family, HELDOUT_ENTITY_INDEX,
                                 memory_length=MEMORY_LENGTH)
            assert tuple(ep.variant for ep in group) == VARIANTS
            assert all(ep.split == split and ep.family == family for ep in group)
            assert all(ep.memory_length == MEMORY_LENGTH for ep in group)
            views = [seal_model_view(ep) for ep in group]
            assert len({ep.query for ep in group}) == 1
            assert len({model_code_bag(view) for view in views}) == 1
            # Positive bindings rotate eight times without changing query/bag.
            if family in POSITIVES:
                assert {ep.gold_answer for ep in group} == set(ANSWER_IDS)
            else:
                assert all(ep.gold_answer is None for ep in group)
                assert len(set(views)) == 1
            items.append((split, family, group))
    if len(items) != 8:
        raise RuntimeError("#1385 requires eight split-family groups")
    return tuple(items)


def _source_line_for_token(token_index: int | None, encoded: object) -> int | None:
    if token_index is None:
        return None
    visible = [(int(line), int(start)) for line, start in encoded.line_anchors
               if start <= int(token_index)]
    if not visible:
        return None
    return max(visible, key=lambda item: item[1])[0]


def redacted_code_control_view(view: SealedModelView) -> SealedModelView:
    """Preserve parseable memory records but replace all CODE values with stale.

    This intentionally is NOT a primary model input or actual no-memory test:
    it is a separate counterfactual layout-only sentinel control.
    """
    return SealedModelView(
        query=view.query,
        memory_text=re.sub(r"CODE-[0-9]+", f"CODE-{STALE_CODE}", view.memory_text),
        answer_options=view.answer_options,
    )


@torch.no_grad()
def _evaluate_case(model: torch.nn.Module, ep: object, arm: str) -> dict[str, Any]:
    view = seal_model_view(ep)
    pred = predict_from_sealed_view(model, view)
    # Gold position, answer and family remain outside model.forward.
    correct_answer, relation_pos, value_pos = oracle_read(ep)
    first_line = relation_pos if relation_pos is not None else value_pos
    gold_class, first_target, second_candidate = train_only_targets(ep)
    assert correct_answer == ep.gold_answer
    assert (gold_class == len(ANSWER_IDS)) == (correct_answer is None)
    encoded = encode_sealed_view(view)
    second_target = (
        encoded.code_tokens[second_candidate][0]
        if second_candidate is not None else None
    )
    first = pred["first_selected_visible_token_index"]
    second = pred["second_selected_visible_token_index"]
    if arm == "transformer":
        assert first is None and second is None
    else:
        assert first is not None and second is not None
        assert 0 <= int(first) < encoded.memory_length
        assert second in {position for position, _ in encoded.code_tokens}
    second_codes = {position: code for position, code in encoded.code_tokens}
    copied_code = second_codes.get(second)
    return {
        "split": ep.split, "family": ep.family,
        "entity_index": HELDOUT_ENTITY_INDEX, "variant": ep.variant,
        "arm": arm, "gold_answer_id": correct_answer,
        "predicted_answer_id": pred["predicted_answer_id"],
        "confidence": float(pred["confidence"]),
        "abstained": bool(pred["abstained"]),
        "correct": pred["predicted_answer_id"] == correct_answer,
        "gold_first_source_line": first_line,
        "gold_second_source_line": value_pos,
        "gold_first_token_index": first_target,
        "gold_second_code_token_index": second_target,
        "predicted_first_token_index": first,
        "predicted_second_code_token_index": second,
        "predicted_first_source_line": _source_line_for_token(first, encoded),
        "predicted_second_source_line": _source_line_for_token(second, encoded),
        "first_exact": first == first_target if first is not None and first_target is not None else None,
        "first_source_line_correct": (
            _source_line_for_token(first, encoded) == first_line
            if first is not None and first_line is not None else None
        ),
        "second_exact": second == second_target if second is not None and second_target is not None else None,
        "second_source_line_correct": (
            _source_line_for_token(second, encoded) == value_pos
            if second is not None and value_pos is not None else None
        ),
        "hard_copied_visible_code_id": copied_code,
        "hard_copy_selects_stale_code": copied_code == STALE_CODE if copied_code is not None else None,
        "no_match_false_positive": (
            bool(not pred["abstained"]) if ep.family == "no_match" else None
        ),
        "encoded_tokens": len(encoded.token_ids),
    }


def run_cpu_one_shot_report() -> dict[str, Any]:
    if PAID_TRAINING_AUTHORIZED or ORIGINAL_SCIENTIFIC_STATUS != "CHM_V3_100M_DAEC_STAGE_C_STOP":
        raise RuntimeError("frozen #1385 no-GPU/STOP invariant failed")
    if ARCHIVED_SCIENTIFIC_SEED_CONSUMED != 2_013_161:
        raise RuntimeError("historical seed status changed")
    torch.set_num_threads(1)
    start = time.monotonic()
    groups = frozen_heldout_groups()
    group_identities = [{
        "split": split, "family": family, "entity": episodes[0].entity,
        "memory_length": episodes[0].memory_length, "cases": len(episodes),
        "query": episodes[0].query,
        "memory_code_bag": model_code_bag(seal_model_view(episodes[0])),
    } for split, family, episodes in groups]
    heldout_hash = _canonical_hash(group_identities)

    train_start = time.monotonic()
    transformer, pointer_ce, pointer_no_ce, training = train_cpu_models(
        steps=TRAIN_STEPS_PER_ARM
    )
    training_sec = time.monotonic() - train_start
    models = dict(zip(ARMS, (transformer, pointer_ce, pointer_no_ce)))
    for m in models.values():
        m.eval()
    if training["steps_per_arm"] != 48 or training["historical_scientific_status"] != ORIGINAL_SCIENTIFIC_STATUS:
        raise RuntimeError("trainer contract drift")
    params = training["parameters"]
    if abs(params["pointer_ce"] - params["transformer"]) / max(params.values()) > 0.01:
        raise RuntimeError("tiny parameter mismatch >1%")

    results: list[dict[str, Any]] = []
    eval_seconds: dict[str, float] = {}
    redacted_controls: list[dict[str, Any]] = []
    for arm, model in models.items():
        arm_start = time.monotonic()
        for split, family, episodes in groups:
            # One prediction per visible case, no test tuning or model changes.
            for ep in episodes:
                results.append(_evaluate_case(model, ep, arm))
            if family in POSITIVES:
                # Across the eight rotations, the redacted view is identical:
                # evaluate it ONCE per family/arm/split, not eight fake samples.
                redacted_views = [redacted_code_control_view(seal_model_view(ep))
                                  for ep in episodes]
                if len(set(redacted_views)) != 1:
                    raise RuntimeError("redacted counterfactual still leaks answer variant")
                view = redacted_views[0]
                assert len(encode_sealed_view(view).code_tokens) >= 8
                redacted_pred = predict_from_sealed_view(model, view)
                redacted_correct = sum(
                    redacted_pred["predicted_answer_id"] == ep.gold_answer
                    for ep in episodes
                )
                assert redacted_correct in (0, 1)
                redacted_controls.append({
                    "split": split, "family": family, "arm": arm,
                    "group_count": 1, "rotations": 8,
                    "predicted_answer_id": redacted_pred["predicted_answer_id"],
                    "correct_among_8_rotations": redacted_correct,
                    "control_kind": "all_code_values_replaced_with_CODE-19001",
                })
            if time.monotonic() - start > MAX_TOTAL_WALL_SECONDS:
                raise TimeoutError("precommitted CPU-only 2400s total wall cap exceeded")
        eval_seconds[arm] = time.monotonic() - arm_start

    panels = []
    for split in HELDOUT_SPLITS:
        for family in FAMILIES:
            for arm in ARMS:
                rows = [r for r in results if r["split"] == split and
                        r["family"] == family and r["arm"] == arm]
                assert len(rows) == 8
                panels.append({
                    "split": split, "family": family, "arm": arm,
                    "scored": 8, "correct": sum(r["correct"] for r in rows),
                    "abstained": sum(r["abstained"] for r in rows),
                    "first_exact": sum(r["first_exact"] is True for r in rows),
                    "first_source_line_correct": sum(
                        r["first_source_line_correct"] is True for r in rows),
                    "second_exact": sum(r["second_exact"] is True for r in rows),
                    "second_source_line_correct": sum(
                        r["second_source_line_correct"] is True for r in rows),
                    "hard_copy_selects_stale": sum(
                        r["hard_copy_selects_stale_code"] is True for r in rows),
                    "negative_false_positive": sum(
                        r["no_match_false_positive"] is True for r in rows),
                    "independent_entity_groups": 1,
                })

    if len(results) != 3 * 2 * EXPECTED_EPISODES_PER_SPLIT or len(panels) != 24:
        raise RuntimeError("one-shot scored panel count drift")
    if len(redacted_controls) != 3 * 2 * len(POSITIVES):
        raise RuntimeError("redacted controls count drift")
    if any(r["gold_answer_id"] not in ANSWER_IDS for r in results
           if r["family"] in POSITIVES):
        raise RuntimeError("unbalanced label envelope")
    for split in HELDOUT_SPLITS:
        for family in POSITIVES:
            ids=[r["gold_answer_id"] for r in results
                 if r["split"]==split and r["family"]==family and r["arm"]=="transformer"]
            if set(ids)!=set(ANSWER_IDS):
                raise RuntimeError("counterfactual rotations did not cover all answers")

    return {
        "classification": SUCCESS_LABEL,
        "control_issue": ISSUE,
        "prereg_issue": PARENT_ISSUE,
        "historical_scientific_classification": ORIGINAL_SCIENTIFIC_STATUS,
        "historical_scientific_seed_consumed": True,
        "gpu_used": False, "modal_used": False, "new_scientific_attempt": False,
        "paid_training_authorized": False,
        "synthetic_cpu_only": True,
        "training_episodes_per_arm": 48,
        "optimizer_updates_per_arm": 48,
        "same_training_episode_order_for_all_arms": True,
        "training": training,
        "training_wall_seconds_total": training_sec,
        "evaluation_wall_seconds_by_arm": eval_seconds,
        "total_wall_seconds": time.monotonic() - start,
        "train_dev_test_entity_splits_disjoint": True,
        "heldout_entity_index": HELDOUT_ENTITY_INDEX,
        "heldout_memory_length": MEMORY_LENGTH,
        "heldout_split_family_group_manifest_sha256": heldout_hash,
        "evaluations_per_split_per_arm": EXPECTED_EPISODES_PER_SPLIT,
        "independent_positive_groups_per_split": len(POSITIVES),
        "cases_per_positive_group": 8,
        "parameter_parity_within_1_percent": True,
        "compute_parity_verified": False,
        "interpretation": (
            "Tiny synthetic 48-update CPU study only; pointer heads have extra "
            "forward/backward compute and training objectives differ. Four independent "
            "entity groups per split, 8 correlated rotations each. No natural-language "
            "LLM or historic 100M scientific conclusion."
        ),
        "answer_loss_plus_pointer_supervision_only_in_arm_B": True,
        "pointer_first_hop_teacher_forced_in_forward": False,
        "case_rows": results,
        "per_family": panels,
        "redacted_controls": redacted_controls,
        "query_only_or_bag_only_positive_group_ceiling": "1/8",
        "scientific_quality_pass_authorized": False,
        "stage_d_authorized": False,
        "replication_authorized": False,
    }


__all__ = (
    "ISSUE", "TRAIN_STEPS_PER_ARM", "frozen_heldout_groups",
    "redacted_code_control_view", "run_cpu_one_shot_report",
)
