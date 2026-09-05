from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_8_issue569_end_to_end_systems_memory_safe as memory_safe
from tam_research import aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py"
LAUNCHER = ROOT / "modal_aera_v26_9_issue646_issue643_e2e_guard_continuation_l4_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-9-issue646-issue643-e2e-guard-continuation-l4.yml"
CPU_TEST = ROOT / "tests/test_aera_v26_9_issue646_issue643_e2e_guard_continuation_cpu.py"

SOURCE_MAIN = "25fd672e923ea66bab5a529de0c3e8a8680bf41e"
SOURCE_TREE = "118f66652b7767a979fb7126aa71603e41c29723"
ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
LAUNCHER_BLOB = "f39ebfca13fd11bb4893710754643e2df1c428a8"
WORKFLOW_BLOB = "f981bca0d1a6f095ee1fef8e14ef28cc297a87ca"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue646_preserves_frozen_issue643_scientific_adapter() -> None:
    assert ADAPTER.exists()
    assert _git_blob(ADAPTER) == ADAPTER_BLOB
    assert systems.RESEARCH_ISSUE == 643
    assert base.SYSTEM_BATCH_SIZES == (8, 64)
    assert base.SYSTEM_WARMUP_CALLS == 3
    assert base.SYSTEM_TIMED_CALLS_PER_ROUND == 20
    assert base.SYSTEM_ROUNDS == 5
    assert base.BATCH8_MIN_FULL_SPEED_RATIO == 0.25
    assert base.BATCH64_MIN_FULL_SPEED_RATIO == 1.25
    assert (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) == (1e-2, 1e-2)
    assert base.EXPECTED_STATE_BYTES == 77_760
    assert (
        base.EXPECTED_SELECTED_WRITES,
        base.EXPECTED_CANDIDATES,
        base.EXPECTED_VECTOR_UPDATES,
    ) == (16, 255, 1)

    inherited = memory_safe.issue569_systems_protocol()
    assert inherited["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert inherited["hard"] is True
    assert inherited["route_mode"] == "hard_sparse"
    assert inherited["dense_masked_sparse_credit"] is False
    assert inherited["chunk_batch_rows"] == 1
    assert inherited["all_elements_covered"] is True
    assert inherited["sampling_or_approximation"] is False
    assert inherited["result_dependent_chunk_sizing"] is False


def test_issue646_cpu_contract_is_nonexecuting_and_keeps_higher_flags_false() -> None:
    contract = systems.cpu_contract_preflight_issue643()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_construction_performed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["systems_measurement_performed"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["s2_authorized"] is False
    assert contract["fresh_scientific_seed_authorized"] is False
    assert contract["independent_replication_credit"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False


def test_issue646_launcher_is_fresh_result_namespace_and_exactly_one_l4_systems_call() -> None:
    source = LAUNCHER.read_text()
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert f'BOUND_SOURCE_MAIN = "{SOURCE_MAIN}"' in source
    assert f'BOUND_SOURCE_TREE = "{SOURCE_TREE}"' in source
    assert 'CONTINUATION_ISSUE = 646' in source
    assert 'RESEARCH_ISSUE = 643' in source
    assert 'APP_NAME = "aera-v26-9-issue646-issue643-e2e-guard-continuation-l4"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert (
        'RESULT_PATH = "/vol/aera-v26/issue646-issue643-e2e-guard-continuation/result.json"'
        in source
    )
    assert "/vol/aera-v26/issue643-bounded-memory-end-to-end-systems/result.json" not in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 600" in source
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("result = run_end_to_end_systems_v26_9_bounded_memory()") == 1
    assert source.count("volume.commit()") == 1
    assert source.index("volume.commit()") < source.index("print(RESULT_MARKER +")
    assert source.count("result_path.exists()") == 2
    assert "issue645_consumed_pre_modal" in source
    assert "scientific_seed_consumed" in source
    assert "breakthrough_proven" in source


def test_issue646_workflow_fixes_multiline_comment_guard_without_scientific_relaxation() -> None:
    source = WORKFLOW.read_text()
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    assert "types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "[aera-v26-9-issue646-issue643-e2e-guard-continuation-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source

    assert "mapfile -t freeze_comments" not in source
    assert "freeze_comment_count=" in source
    assert '| .id' in source
    assert "freeze_comment=" in source
    assert '| .body' in source
    assert 'test "${freeze_comment_count}" = "1"' in source

    assert "33991449361" in source
    assert "101374299440" in source
    assert 'select(.name=="Authenticate Modal")' in source
    assert 'select(.name=="Run canonical issue643 bounded-memory end-to-end systems L4 gate")' in source
    assert '= "skipped"' in source
    assert "old_result_report_count" in source
    assert 'test "${old_result_report_count}" = "0"' in source

    assert "BOUND_MAIN_TO_BE_FILLED_AFTER_MERGE" not in source
    assert "## #646 sole L4 #643 E2E guard-continuation authorization" in source
    assert "modal deploy" not in source.lower()
    assert "rerun" in source.lower()
    assert "timeout increase" in source.lower()


def test_issue646_paths_are_exactly_three_additive_continuation_files() -> None:
    assert LAUNCHER.exists()
    assert WORKFLOW.exists()
    assert CPU_TEST.exists()
    issue646_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*issue646*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    assert issue646_paths == sorted(
        [
            ".github/workflows/aera-v26-9-issue646-issue643-e2e-guard-continuation-l4.yml",
            "modal_aera_v26_9_issue646_issue643_e2e_guard_continuation_l4_app.py",
            "tests/test_aera_v26_9_issue646_issue643_e2e_guard_continuation_cpu.py",
        ]
    )
