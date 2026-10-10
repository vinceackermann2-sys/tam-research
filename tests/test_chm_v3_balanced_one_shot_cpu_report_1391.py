from __future__ import annotations

from collections import Counter
from pathlib import Path
import inspect

import pytest

from scripts import chm_v3_balanced_one_shot_cpu_report_1391 as report
from tam_research.chm_v3_counterfactual_model_view_1365 import seal_model_view
from tam_research.chm_v3_counterfactual_multiset_suite_1373 import FAMILIES
from tam_research.chm_v3_balanced_train_schedule_1391 import TRAIN_STEPS


def test_fresh_panels_are_disjoint_from_old_scored_entity_zero():
    dev = report.fresh_eval_cases("development")
    test = report.fresh_eval_cases("test")
    assert len(dev) == len(test) == 32
    assert report.EVAL_ENTITY_INDICES == {"development":1,"test":2}
    assert set(x.entity for x in dev) == {"V3-development-E001"}
    assert set(x.entity for x in test) == {"V3-test-E002"}
    assert set(x.entity for x in dev).isdisjoint(set(x.entity for x in test))
    for split, cases in (("development",dev),("test",test)):
        assert Counter(ep.family for ep in cases) == dict.fromkeys(FAMILIES,8)
        for family in FAMILIES:
            group = [ep for ep in cases if ep.family == family]
            assert {ep.variant for ep in group} == set(range(8))
            assert len({ep.query for ep in group}) == 1
            assert len({seal_model_view(ep).answer_options for ep in group}) == 1
    for invalid in ("train","bad",""):
        with pytest.raises(ValueError, match="fresh"):
            report.fresh_eval_cases(invalid)


def test_pr_is_source_only_and_workflow_is_one_shot_main_path_gated():
    root = Path(__file__).resolve().parent.parent
    wf = (root / ".github/workflows/chm-v3-1391-balanced-one-shot-cpu-report-v1.yml"
         ).read_text(encoding="utf-8")
    source = inspect.getsource(report)
    assert "on:\n  push:\n    branches: [main]" in wf
    assert "paths:\n      - .github/workflows/chm-v3-1391-balanced-one-shot-cpu-report-v1.yml" in wf
    assert "workflow_dispatch" not in wf and "schedule:" not in wf
    assert 'CUDA_VISIBLE_DEVICES: ""' in wf
    assert "timeout-minutes: 35" in wf
    assert "source-locked" in wf
    assert "actions/upload-artifact@v4" in wf
    assert "CHM_V3_1391_BALANCED_TINY_CPU_REPORT=" in wf
    for path in ("chm_v3_balanced_train_schedule_1391.py",
                 "chm_v3_matched_tiny_models_1377.py",
                 "chm_v3_counterfactual_multiset_suite_1373.py",
                 "chm_v3_counterfactual_model_view_1365.py"):
        assert path in wf
    assert report.ORIGINAL_CONFUNDED_RUN == 38064856253
    assert report.GPU_AUTHORIZED is False if hasattr(report,"GPU_AUTHORIZED") else True
    assert report.TRAIN_STEPS == TRAIN_STEPS == 256
    assert report.EVAL_ENTITY_INDICES == {"development":1,"test":2}
    assert "train_balanced_cpu_models(steps=TRAIN_STEPS)" in source
    assert source.index("train_balanced_cpu_models(steps=TRAIN_STEPS)") < (
        source.index('for split in ("development","test"):')
    )
    assert "len(predictions) != 192" in source
    assert "192" in wf
    for forbidden in ("modal.App(", "modal.run(", "torch.cuda", ".cuda(", "torch.load("):
        assert forbidden not in source


def test_fixed_protocol_refuses_reuse_of_old_scored_index_zero():
    assert 0 not in report.EVAL_ENTITY_INDICES.values()
    assert report.SOURCE_SCIENCE_STOP == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert report.NEW_SCIENTIFIC_RUN is False
    assert report.FAMILIES == ("direct","overwrite","two_hop","no_match")
    assert set(report.ARMS) == {"transformer","pointer_ce","pointer_no_ce"}
