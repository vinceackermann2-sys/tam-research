from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import tam_research.chm_v3_alias_vs_hash_one_shot_1415 as score
from tam_research.chm_v3_counterfactual_multiset_suite_1373 import (
    FAMILIES, POSITIVES, paired_group,
)


def test_pre_frozen_unique_unused_heldout_contract_without_instantiating_it():
    assert score.HELDOUT_ENTITY_INDEX == 3
    assert 3 not in score.CONSUMED_OLD_DEVELOPMENT
    assert 3 not in score.CONSUMED_OLD_TEST
    assert score.SPLITS == ("development","test")
    assert score.ARMS == (
        "whole_decoder", "alias_decoder", "whole_pointer_ce", "alias_pointer_ce"
    )
    assert score.TRAIN_STEPS == 256
    assert score.PRIOR_CONFUNDED_RUN == 38064856253
    assert score.PRIOR_BALANCED_RUN == 38066888678
    assert score.GPU_AUTHORIZED is False
    assert score.NEW_SCIENTIFIC_ATTEMPT is False
    for bad in ("train", "random", ""):
        with pytest.raises(ValueError,match="development/test"):
            score.fresh_unscored_panel(bad)


def test_train_only_multiset_blind_baselines_source_free_and_balanced():
    # No previously scored development or test episode is even constructed.
    cases = tuple(ep for family in FAMILIES
                  for ep in paired_group("train", family, 4))
    controls = score.analytic_blind_controls(cases)
    assert set(controls) == set(FAMILIES)
    for family in POSITIVES:
        cell = controls[family]
        assert cell["eight_way_complete"]
        assert cell["query_only_upper_bound_accuracy"] == .125
        assert cell["redacted_layout_only_upper_bound_accuracy"] == .125
        assert cell["bag_of_codes_only_upper_bound_accuracy"] == .125
        assert cell["always_abstain_correct"] == 0
    assert controls["no_match"]["always_abstain_correct"] == 8
    assert controls["no_match"]["query_only_upper_bound_accuracy"] is None


def test_aggregate_scoring_geometry_using_only_invented_rows_not_heldout():
    examples = [
        {
            "family": family,
            "counterfactual_variant": i,
            "correct_answer": family == "no_match",
            "predicted_answer_id": None,
            "abstained": True,
            "gold_answer_id_evaluator_only": (
                None if family == "no_match" else 18001 + i
            ),
            "first_index_exact_match": None,
            "second_index_exact_match": None,
        }
        for family in FAMILIES for i in range(8)
    ]
    result = score.summarize_arm(examples)
    assert result["cases"] == 32
    assert result["overall_accuracy"] == .25
    assert result["positive_memory_accuracy"] == 0.0
    assert result["per_family"]["no_match"]["no_match_false_positive_count"] == 0
    assert all(x["confidence_interval_justified"] is False
               for x in result["per_family"].values())


def test_one_shot_source_pins_same_train_order_and_scoring_after_optimizer():
    source = inspect.getsource(score.main)
    assert "audit_training_balance()" in source
    assert "matched_original_and_alias_models()" in source
    assert "balanced_training_episode(step)" in source
    assert "range(TRAIN_STEPS)" in source
    assert "learning_objective(" in source
    assert 'arm.endswith("pointer_ce")' in source
    assert "fresh_unscored_panel(split)" in source
    assert "len(predictions) != 4 * 2 * 32" in source
    assert source.index("for step in range(TRAIN_STEPS):") < source.index(
        "for split in SPLITS:"
    )
    assert source.index("model.eval()") < source.index("fresh_unscored_panel(split)")
    assert "torch.set_num_threads(1)" in source
    assert "evaluator_held_gold_indices(episode)" in inspect.getsource(score._score_one)
    assert inspect.getsource(score._score_one).index(
        "predict_from_sealed_view(model, view)"
    ) < inspect.getsource(score._score_one).index(
        "evaluator_held_gold_indices(episode)"
    )


def test_first_push_only_cpu_scored_workflow_is_static_and_fail_closed():
    wf = (Path(__file__).resolve().parents[1] /
          ".github/workflows/chm-v3-1415-alias-vs-hash-cpu-one-shot.yml"
         ).read_text(encoding="utf-8")
    assert "on:\n  push:\n    branches: [main]" in wf
    assert (
        "paths:\n      - .github/workflows/chm-v3-1415-alias-vs-hash-cpu-one-shot.yml"
    ) in wf
    assert "github.run_attempt == 1" in wf
    assert "workflow_dispatch" not in wf
    assert "schedule:" not in wf
    assert 'CUDA_VISIBLE_DEVICES: ""' in wf
    assert "timeout-minutes: 35" in wf
    assert "source-locked" in wf
    assert "actions/upload-artifact@v4" in wf
    assert "CHM_V3_1415_ALIAS_VS_HASH_ONE_SHOT=" in wf
    assert "256" in wf
    for name in (
        "chm_v3_alias_tiny_adapters_1410.py",
        "chm_v3_query_anchored_alias_1407.py",
        "chm_v3_balanced_train_schedule_1391.py",
        "chm_v3_matched_tiny_models_1377.py",
        "chm_v3_counterfactual_multiset_suite_1373.py",
        "chm_v3_alias_vs_hash_one_shot_1415.py",
    ):
        assert name in wf
    for banned in (
        "modal.App(", "modal.run(", "torch.cuda", ".cuda(",
        "torch.load(", "checkpoint.load(",
    ):
        assert banned not in inspect.getsource(score)


def test_training_only_encoder_timing_cannot_use_heldout_examples():
    src = inspect.getsource(score._record_cpu_encoder_overhead_on_train)
    assert "balanced_training_episode(step)" in src
    assert "encode_sealed_view(view)" in src
    assert "_encode_alias_view(view)" in src
    assert "fresh_unscored_panel(" not in src
    assert 'generate_episode("test"' not in src
    assert 'generate_episode("development"' not in src
    assert score.STAGE_C_STOP == "CHM_V3_100M_DAEC_STAGE_C_STOP"
