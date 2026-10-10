from __future__ import annotations

from collections import Counter
from pathlib import Path
import inspect

import pytest

from scripts import chm_v3_matched_tiny_one_shot_cpu_report_1377 as report
from tam_research.chm_v3_counterfactual_model_view_1365 import (
    seal_model_view, oracle_read,
)
from tam_research.chm_v3_matched_tiny_models_1377 import encode_sealed_view


@pytest.mark.parametrize("split", report.SPLITS)
def test_locked_panel_only_has_eight_counterfactuals_per_family(split):
    cases = report.eval_cases(split)
    assert len(cases) == 32
    assert Counter(ep.family for ep in cases) == {
        "direct": 8, "overwrite": 8, "two_hop": 8, "no_match": 8
    }
    assert {ep.entity for ep in cases} == {f"V3-{split}-E000"}
    for family in report.FAMILIES:
        group = [ep for ep in cases if ep.family == family]
        assert {ep.variant for ep in group} == set(range(8))
        assert len({ep.query for ep in group}) == 1
        assert len({tuple((f.position, f.subject, f.kind, f.document)
                          for f in ep.facts) for ep in group}) == 1


@pytest.mark.parametrize("family", report.FAMILIES)
def test_evaluator_labels_come_from_gold_memory_only_after_model_prediction(family):
    ep = [e for e in report.eval_cases("test") if e.family == family][3]
    first, second = report.evaluator_held_gold_indices(ep)
    _, rel, actual = oracle_read(ep)
    if family == "no_match":
        assert (first, second) == (None, None)
        return
    assert first is not None and second is not None
    encoded = encode_sealed_view(seal_model_view(ep))
    assert first in dict(encoded.line_anchors).values()
    assert second in [p for p, code in encoded.code_tokens if code == ep.gold_answer]
    if family == "two_hop":
        assert first == dict(encoded.line_anchors)[rel]
    else:
        assert first == dict(encoded.line_anchors)[actual]
    score_source = inspect.getsource(report._score_one)
    assert score_source.index("predict_from_sealed_view(model, view)") < (
        score_source.index("evaluator_held_gold_indices(ep)")
    )


def test_mocked_aggregates_keep_no_match_and_pointer_diagnostics_separate():
    rows = [
        {"family": family,
         "gold_answer_id_evaluator_only": None if family == "no_match" else i,
         "predicted_answer_id": None if family == "no_match" else i,
         "answer_correct": True, "abstained": family == "no_match",
         "first_index_match": None, "second_index_match": None}
        for family in report.FAMILIES for i in range(8)
    ]
    metrics = report._aggregate(rows)
    assert metrics["cases"] == 32 and metrics["accuracy"] == 1.0
    assert metrics["per_family"]["direct"]["gold_answer_id_count"] == 8
    assert metrics["per_family"]["two_hop"]["per_rotation_answer_following"] == 1.0
    assert metrics["per_family"]["no_match"]["negative_false_positive_count"] == 0
    assert all(v["confidence_interval_estimable"] is False
               for v in metrics["per_family"].values())


def test_static_single_shot_workflow_and_result_contract():
    root = Path(__file__).resolve().parent.parent
    workflow = (root / ".github/workflows/chm-v3-1377-tiny-one-shot-cpu-report-v1.yml"
               ).read_text(encoding="utf-8")
    source = inspect.getsource(report)
    assert "on:\n  push:\n    branches: [main]" in workflow
    assert "paths:\n      - .github/workflows/chm-v3-1377-tiny-one-shot-cpu-report-v1.yml" in workflow
    assert "workflow_dispatch" not in workflow and "schedule:" not in workflow
    assert 'CUDA_VISIBLE_DEVICES: ""' in workflow
    assert "timeout-minutes: 35" in workflow
    for name in ("chm_v3_matched_tiny_models_1377.py",
                 "chm_v3_counterfactual_multiset_suite_1373.py",
                 "chm_v3_counterfactual_model_view_1365.py"):
        assert name in workflow
    assert "CHM_V3_1377_TINY_CPU_REPORT=" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "source-locked" in workflow
    assert report.TRAIN_STEPS == 48
    assert report.EVAL_ENTITY_INDEX == 0
    assert report.GPU_USED is False and report.NEW_SCIENTIFIC_RUN is False
    for forbidden in ("modal.App(", "modal.run(", "torch.cuda", ".cuda(", "torch.load("):
        assert forbidden not in source
    assert "eval_cases(split)" in source
    assert "one_additional_nonupdating_cpu_fwb_seconds_by_arm" in source


def test_no_held_out_cases_enter_training_schedule_or_gradients():
    train_src = inspect.getsource(report.main)
    assert "train_cpu_models(steps=TRAIN_STEPS)" in train_src
    assert "for split in SPLITS:" in train_src
    assert train_src.index("train_cpu_models(steps=TRAIN_STEPS)") < (
        train_src.index("for split in SPLITS:")
    )
    assert "torch.set_num_threads(1)" in train_src
    assert "len(all_predictions) != 3 * 2 * 32" in train_src
    assert "per_case_predictions" in train_src
    assert "interpretation_ceiling" in train_src
