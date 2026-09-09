from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/aera-issue790-event-memory-seed37641-runner.yml"
ORIGINAL_TEST = ROOT / "tests/test_aera_issue790_event_memory_seed37641_runner_cpu.py"
ADAPTER = ROOT / "tam_research/aera_issue790_event_memory_seed37641_scientific_adapter.py"
PROTOCOL = ROOT / "docs/aera_issue790_event_memory_seed37641_protocol.json"
RUNNER = ROOT / "modal_aera_issue790_event_memory_seed37641_runner.py"


def _blob(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()


def test_issue811_exact_corrected_workflow_and_frozen_runtime_files() -> None:
    assert _blob(WORKFLOW) == "f4357904b4316b6e2bc2dfcc6ef390a5317d2da5"
    assert _blob(ORIGINAL_TEST) == "94670f0f77f5ee86ce58ac9e5b3af29be22f2b7d"
    assert _blob(ADAPTER) == "931ba460cae55ceb940b61d92288668f6e02ff53"
    assert _blob(PROTOCOL) == "9f865dff148aa46ffd9957259aeb4c0591595cad"
    assert _blob(RUNNER) == "58bc3bbe16492fa8161456ea5103653dc320f6b6"


def test_issue811_workflow_uses_machine_readable_correction_freeze() -> None:
    source = WORKFLOW.read_text()
    assert 'test "${workflow_blob}" = "250f966c2c1ced919663ca8ee57aa03a2edf8852"' in source
    assert 'test "${test_blob}" = "94670f0f77f5ee86ce58ac9e5b3af29be22f2b7d"' in source
    assert "## #806 pre-ref provenance-guard correction freeze" in source
    assert "CORRECTED_WORKFLOW_BLOB=" in source
    assert 'corrected_workflow_blob="$(printf' in source
    assert 'test -n "${corrected_workflow_blob}"' in source
    assert (
        'git hash-object .github/workflows/aera-issue790-event-memory-seed37641-runner.yml)" = "${corrected_workflow_blob}"'
        in source
    )
    assert (
        'git hash-object tests/test_aera_issue790_event_memory_seed37641_runner_cpu.py)" = "${test_blob}"'
        in source
    )


def test_issue811_workflow_uses_merged_refresh_lineage_not_sibling_ancestry() -> None:
    source = WORKFLOW.read_text()
    assert 'git cat-file -e "${frozen_commit}^{commit}"' in source
    assert 'test "$(git rev-parse "${frozen_commit}^{tree}")" = "${frozen_tree}"' in source
    assert 'git merge-base --is-ancestor "${frozen_commit}" "${bound_main}"' not in source

    merged = "49be1185d38c0a38bcd5f38ded4430689c22b753"
    refreshed_head = "ad454169c1380ed303383e7911f5045c7c47084a"
    refreshed_tree = "a492b2cc9ece22a3ab8db94536e3923b4ebb34ff"
    assert f'git cat-file -e "{merged}^{{commit}}"' in source
    assert f'test "$(git rev-parse {merged}^{{tree}})" = "{refreshed_tree}"' in source
    assert f'git merge-base --is-ancestor {merged} "${{bound_main}}"' in source
    assert f'git cat-file -e "{refreshed_head}^{{commit}}"' in source
    assert f'test "$(git rev-parse {refreshed_head}^{{tree}})" = "{refreshed_tree}"' in source
    assert f'git merge-base --is-ancestor {refreshed_head} "${{bound_main}}"' in source


def test_issue811_dynamic_frozen_blob_guards_are_preserved() -> None:
    source = WORKFLOW.read_text()
    dynamic = {
        "adapter_blob": ("ADAPTER_BLOB", "tam_research/aera_issue790_event_memory_seed37641_scientific_adapter.py"),
        "protocol_blob": ("PROTOCOL_BLOB", "docs/aera_issue790_event_memory_seed37641_protocol.json"),
        "runner_blob": ("RUNNER_BLOB", "modal_aera_issue790_event_memory_seed37641_runner.py"),
        "test_blob": ("RUNNER_TEST_BLOB", "tests/test_aera_issue790_event_memory_seed37641_runner_cpu.py"),
    }
    for variable, (freeze_key, path) in dynamic.items():
        assert f'{variable}="$(printf' in source
        assert f"s/^{freeze_key}=" in source
        assert f'git hash-object {path})" = "${{{variable}}}"' in source

    # The #809 false-negative was demanding this SHA literally in YAML. The
    # identity is instead proven above by direct file hashing and by the
    # workflow's RUNNER_BLOB -> runner_blob runtime check.
    assert "58bc3bbe16492fa8161456ea5103653dc320f6b6" not in source


def test_issue811_literal_scientific_dependency_guards_remain_exact() -> None:
    source = WORKFLOW.read_text()
    for sha in (
        "aa90b3ae3266786323b0517de8385135869c3653",
        "afc939a69633f68ded05eb585c95a599a8f5c981",
        "1489df753f2eee0f3acc5aba049c0eeb40fce10d",
        "b695b1b7b7476be3a16433c96dff32870f2e1f49",
        "40003b68987d026265b1d53fcb637f18277f9cac",
        "981602432b684989f7a5011ff953e6965368c5e5",
        "97861a2407876f62665b13140c2135b4a11d4597",
        "ec0b5d29b3d4ac27bd60fd9c152480b9f177c3e9",
    ):
        assert sha in source


def test_issue811_seed37641_namespace_and_explicit_scientific_auth_are_unchanged() -> None:
    source = WORKFLOW.read_text()
    assert "[aera-event-memory-repair-seed37641-preauth-v1]" in source
    assert "[aera-event-memory-repair-seed37641-l4-v1]" in source
    assert "/vol/aera-capability/event-memory-repair-v2-seed37641/result.json" in source
    assert "/vol/aera-capability/event-memory-repair-v2-seed37641/checkpoints" in source
    assert "## #790 repaired seed 37641 L4 scientific authorization" in source
    assert "Authorize seed: `37641`" in source
    assert "GITHUB_RUN_ATTEMPT" in source
    assert source.count("modal_aera_issue790_event_memory_seed37641_runner.py::preauth_main") == 1
    assert source.count("modal_aera_issue790_event_memory_seed37641_runner.py::l4_main") == 1
    assert "[aera-event-memory-repair-seed27641-l4-v1]" not in source
    assert "/event-memory-repair-v1-seed27641/" not in source
