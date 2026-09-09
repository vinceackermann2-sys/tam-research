from __future__ import annotations

import json
from pathlib import Path

from architectures.cortex_s.experiments.scale100m_2b.systems_profile_v1 import (
    CLASSIFICATION,
    CONSUMED_ENGINEERING_SEEDS,
    ENGINEERING_SEED,
    FORBIDDEN_SCIENTIFIC_SEEDS,
    PARENT_V3_ISSUE,
    PARENT_V3_JOB_ID,
    PARENT_V3_RUN_ID,
    PARENT_V3_SOURCE_SHA,
    PROFILE_STEPS,
    RESULT_ROOT,
    TRIGGER_TITLE,
    WEIGHT_MATERIALIZATION_REPEATS,
    _event_device_time_us,
    validate_profile_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_profile_v1_seed_namespace_and_authority_are_frozen() -> None:
    protocol = validate_profile_protocol()
    assert CLASSIFICATION == "ENGINEERING_SYSTEMS_PROFILE_ONLY"
    assert ENGINEERING_SEED == 2026090906
    assert ENGINEERING_SEED not in CONSUMED_ENGINEERING_SEEDS
    assert ENGINEERING_SEED not in FORBIDDEN_SCIENTIFIC_SEEDS
    assert 2026090905 in CONSUMED_ENGINEERING_SEEDS
    assert 8100 in FORBIDDEN_SCIENTIFIC_SEEDS
    assert {48131, 48132, 48133}.issubset(set(FORBIDDEN_SCIENTIFIC_SEEDS))
    assert TRIGGER_TITLE == "[modal-cortex-s-100m-systems-profile-v1]"
    assert RESULT_ROOT == "/vol/cortex-s-v0/100m-systems-profile-v1"
    assert protocol["full_training_authorized"] is False
    assert protocol["next_stage_authorized"] is False
    assert protocol["scientific_evidence"] is False
    json.dumps(protocol)


def test_profile_v1_is_bound_to_consumed_v3_evidence() -> None:
    assert PARENT_V3_SOURCE_SHA == "3a9538228d100a40818030a64b5b9eaf7311f94a"
    assert PARENT_V3_ISSUE == 802
    assert PARENT_V3_RUN_ID == 34345899535
    assert PARENT_V3_JOB_ID == 102447296828
    assert PROFILE_STEPS == 2
    assert WEIGHT_MATERIALIZATION_REPEATS == 20


def test_profiler_device_time_reader_accepts_current_and_legacy_names() -> None:
    class Current:
        self_device_time_total = 12.5

    class Legacy:
        self_cuda_time_total = 7.25

    class Empty:
        pass

    assert _event_device_time_us(Current()) == 12.5
    assert _event_device_time_us(Legacy()) == 7.25
    assert _event_device_time_us(Empty()) == 0.0


def test_profile_launcher_is_single_use_and_cannot_launch_training() -> None:
    source = (REPO_ROOT / "modal_cortex_s_100m_systems_profile_v1.py").read_text(
        encoding="utf-8"
    )
    assert 'ENGINEERING_SEED = 2_026_090_906' in source
    assert 'APP_NAME = "cortex-s-v0-100m-systems-profile-v1"' in source
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-profile-v1"' in source
    assert 'H100_DISPATCH_CONSUMED.json' in source
    assert 'RESULT.json' in source
    assert 'phase: str = "systems-profile-v1"' in source
    assert "def full_2b(" not in source
    assert "train_full_2b" not in source
    assert "paired-seed8100" not in source
    assert '"full_training_authorized": False' in source
    assert '"next_stage_authorized": False' in source


def test_profile_workflow_is_issue_only_and_source_bound() -> None:
    workflow = (
        REPO_ROOT / ".github/workflows/modal-cortex-s-100m-systems-profile-v1.yml"
    ).read_text(encoding="utf-8")
    assert "[modal-cortex-s-100m-systems-profile-v1]" in workflow
    assert "workflow_dispatch" not in workflow
    assert "systems-profile-v1" in workflow
    assert "git rev-parse origin/main" in workflow
    assert "modal_cortex_s_100m_systems_profile_v1.py" in workflow


def test_profile_document_explicitly_requires_separate_candidate_stage() -> None:
    doc = (REPO_ROOT / "architectures/cortex_s/SYSTEMS_PROFILE_100M_V1.md").read_text(
        encoding="utf-8"
    )
    assert "ENGINEERING_SYSTEMS_PROFILE_ONLY" in doc
    assert "2026090906" in doc
    assert "cannot authorize 2B training" in doc
    assert "requires a separate preregistered candidate stage" in doc
