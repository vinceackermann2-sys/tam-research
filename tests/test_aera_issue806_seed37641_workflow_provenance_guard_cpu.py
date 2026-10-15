from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/aera-issue790-event-memory-seed37641-runner.yml"
ORIGINAL_TEST = ROOT / "tests/test_aera_issue790_event_memory_seed37641_runner_cpu.py"


def _blob(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()


def test_issue806_preserves_original_issue790_cpu_contract_blob() -> None:
    assert _blob(ORIGINAL_TEST) == "94670f0f77f5ee86ce58ac9e5b3af29be22f2b7d"


def test_issue806_workflow_uses_corrected_refresh_lineage_not_sibling_ancestry() -> None:
    source = WORKFLOW.read_text()

    # Historical #790 freeze identities remain evidence, not current workflow identity.
    assert 'test "${workflow_blob}" = "250f966c2c1ced919663ca8ee57aa03a2edf8852"' in source
    assert 'test "${test_blob}" = "94670f0f77f5ee86ce58ac9e5b3af29be22f2b7d"' in source
    assert 'git cat-file -e "${frozen_commit}^{commit}"' in source
    assert 'test "$(git rev-parse "${frozen_commit}^{tree}")" = "${frozen_tree}"' in source
    assert 'git merge-base --is-ancestor "${frozen_commit}" "${bound_main}"' not in source

    # The one #806 freeze supplies the current workflow blob after the blob exists.
    assert "## #806 pre-ref provenance-guard correction freeze" in source
    assert "CORRECTED_WORKFLOW_BLOB=" in source
    assert 'corrected_workflow_blob="$(printf' in source
    assert 'git hash-object .github/workflows/aera-issue790-event-memory-seed37641-runner.yml)" = "${corrected_workflow_blob}"' in source
    assert 'git hash-object tests/test_aera_issue790_event_memory_seed37641_runner_cpu.py)" = "${test_blob}"' in source

    # Exact merged base-refresh provenance must gate every future bound execution commit.
    merged = "49be1185d38c0a38bcd5f38ded4430689c22b753"
    refreshed_head = "ad454169c1380ed303383e7911f5045c7c47084a"
    refreshed_tree = "a492b2cc9ece22a3ab8db94536e3923b4ebb34ff"
    assert f'git cat-file -e "{merged}^{{commit}}"' in source
    assert f'git rev-parse {merged}^{{tree}}' in source
    assert refreshed_tree in source
    assert f'git merge-base --is-ancestor {merged} "${{bound_main}}"' in source
    assert f'git cat-file -e "{refreshed_head}^{{commit}}"' in source
    assert f'git rev-parse {refreshed_head}^{{tree}}' in source
    assert f'git merge-base --is-ancestor {refreshed_head} "${{bound_main}}"' in source


def test_issue806_preserves_seed37641_namespace_and_scientific_authority_gates() -> None:
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


def test_issue806_preserves_all_scientific_runtime_blob_guards() -> None:
    source = WORKFLOW.read_text()
    required = {
        "931ba460cae55ceb940b61d92288668f6e02ff53",
        "9f865dff148aa46ffd9957259aeb4c0591595cad",
        "58bc3bbe16492fa8161456ea5103653dc320f6b6",
        "94670f0f77f5ee86ce58ac9e5b3af29be22f2b7d",
        "aa90b3ae3266786323b0517de8385135869c3653",
        "afc939a69633f68ded05eb585c95a599a8f5c981",
        "1489df753f2eee0f3acc5aba049c0eeb40fce10d",
        "b695b1b7b7476be3a16433c96dff32870f2e1f49",
        "40003b68987d026265b1d53fcb637f18277f9cac",
        "981602432b684989f7a5011ff953e6965368c5e5",
        "97861a2407876f62665b13140c2135b4a11d4597",
        "ec0b5d29b3d4ac27bd60fd9c152480b9f177c3e9",
    }
    for sha in required:
        assert sha in source
