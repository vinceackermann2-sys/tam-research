from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_10_issue716_l4_auth_guard_repair1.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-10-issue716-l4-auth-guard-repair1.yml"

LAUNCHER_BLOB = "84cd58e1f3eed3a51efaff2d7cce2e674c29405f"
WORKFLOW_BLOB = "4dbdb80245b6767fc07931c45190663d8c9197af"
ISSUE710_LAUNCHER_BLOB = "b885250753ea169cd89dd7a978bb3647fd261fe8"
V26_10_IMPL_BLOB = "d8f691c198eed1fa96bcbb78a4e76cad82d18779"
AUTHORIZATION_HEADING = "## #716 sole L4 authorization-heading guard repair1 authorization"

FROZEN_FILES = {
    "modal_aera_v26_10_issue710_memory_safe_harness.py": ISSUE710_LAUNCHER_BLOB,
    "tam_research/aera_hardware_core_v26_10_latent_depth_sync_coalescing.py": V26_10_IMPL_BLOB,
    "modal_aera_v26_10_issue706_latent_depth_sync_coalescing.py": "e23072aa83b0bc20ccd72e7567f234e9f4a931f6",
    "modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py": "72f27391ff2f0a7bff8d4532f307ddc4869cf494",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _launcher_tree() -> ast.Module:
    return ast.parse(LAUNCHER.read_text())


def test_issue716_exact_files_and_frozen_dependencies():
    assert LAUNCHER.exists() and WORKFLOW.exists()
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    ast.parse(LAUNCHER.read_text())
    for rel, expected in FROZEN_FILES.items():
        path = ROOT / rel
        assert path.exists(), rel
        assert _git_blob(path) == expected, rel


def test_issue716_is_orchestration_only_and_uses_frozen710_local_body():
    text = LAUNCHER.read_text()
    assert "import modal_aera_v26_10_issue710_memory_safe_harness as frozen710" in text
    assert "frozen710.preflight.local()" in text
    assert "frozen710.run_microbenchmark.local()" in text
    assert "frozen710.preflight.remote()" not in text
    assert "frozen710.run_microbenchmark.remote()" not in text
    assert "def _chunked_logit_equivalence" not in text
    assert "torch.cuda.Event" not in text
    assert "torch.profiler" not in text
    assert "install_latent_depth_sync_coalescing_v26_10" not in text
    assert "_capture_decisions" not in text
    assert "_event_timed_call" not in text
    assert "_trace_sync_count" not in text
    for forbidden in (".backward(", "torch.optim.", ".step(", ".zero_grad("):
        assert forbidden not in text


def test_issue716_identity_patch_is_narrow_and_restored_in_finally():
    tree = _launcher_tree()
    patch_assign = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_PATCH_FIELDS" for t in node.targets)
    )
    assert isinstance(patch_assign.value, ast.Dict)
    keys = {
        key.value for key in patch_assign.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert keys == {
        "RESULT_PATH",
        "SOURCE_MAIN",
        "SOURCE_TREE",
        "RESEARCH_ISSUE",
        "CONSUMED_TRIGGER",
        "CONSUMED_FAILURE_COMMENT",
        "CONSUMED_RUN",
        "CONSUMED_JOB",
        "CONSUMED_ATTEMPT",
        "PREAUTH_MARKER",
        "L4_START_MARKER",
        "RESULT_MARKER",
        "SUMMARY_MARKER",
    }
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_successor_identity")
    tries = [node for node in ast.walk(fn) if isinstance(node, ast.Try)]
    assert len(tries) == 1
    assert tries[0].finalbody, "successor identity globals must restore in finally"
    fn_text = ast.get_source_segment(LAUNCHER.read_text(), fn)
    assert "setattr(frozen710, name, value)" in fn_text
    assert "previous = {name: getattr(frozen710, name) for name in _PATCH_FIELDS}" in fn_text


def test_issue716_freezes_all_inherited_scientific_and_performance_fixtures():
    text = LAUNCHER.read_text()
    for required in (
        'RESULT_PATH = "/vol/aera-v26/issue716-v26-10-l4-auth-guard-repair1/result.json"',
        'SOURCE_MAIN = "8371d9ab3988df33c7b92b529016618ebf60f9ee"',
        'SOURCE_TREE = "fd171423b5878c9082e689d7d83da9f8c7c81569"',
        "RESEARCH_ISSUE = 716",
        "CONSUMED_TRIGGER = 715",
        "CONSUMED_FAILURE_COMMENT = 5573729021",
        "CONSUMED_RUN = 34145700849",
        "CONSUMED_JOB = 101817171862",
        "CONSUMED_ATTEMPT = 1",
        f'ISSUE710_LAUNCHER_BLOB = "{ISSUE710_LAUNCHER_BLOB}"',
        f'V26_10_IMPL_BLOB = "{V26_10_IMPL_BLOB}"',
        "LOGIT_COMPARE_BATCH_CHUNK = frozen710.LOGIT_COMPARE_BATCH_CHUNK",
        "BATCHES = tuple(frozen710.BATCHES)",
        "WARMUP_CALLS = frozen710.WARMUP_CALLS",
        "TIMED_CALLS_PER_CONDITION = frozen710.TIMED_CALLS_PER_CONDITION",
        "PROFILE_CALLS_PER_CONDITION = frozen710.PROFILE_CALLS_PER_CONDITION",
        "MIN_LATENCY_IMPROVEMENT = frozen710.MIN_LATENCY_IMPROVEMENT",
        "MIN_STREAM_SYNCHRONIZE_REDUCTION = frozen710.MIN_STREAM_SYNCHRONIZE_REDUCTION",
        "INTEGRATED_ATOL = frozen710.INTEGRATED_ATOL",
        "INTEGRATED_RTOL = frozen710.INTEGRATED_RTOL",
    ):
        assert required in text
    assert "frozen710.LOGIT_COMPARE_BATCH_CHUNK != 1" in text
    assert "tuple(frozen710.BATCHES) != (8, 64)" in text
    assert "frozen710.MIN_LATENCY_IMPROVEMENT != 0.05" in text
    assert "frozen710.MIN_STREAM_SYNCHRONIZE_REDUCTION != 20" in text


def test_issue716_remote_import_graph_packages_frozen710_root_launcher():
    text = LAUNCHER.read_text()
    assert 'ISSUE710_LAUNCHER = "modal_aera_v26_10_issue710_memory_safe_harness.py"' in text
    assert '.add_local_file(\n    ISSUE710_LAUNCHER, f"/root/{ISSUE710_LAUNCHER}"\n)' in text
    assert 'got["issue710_launcher"] = _blob(Path(f"/root/{ISSUE710_LAUNCHER}"))' in text


def test_issue716_workflow_uses_one_exact_authorization_heading_everywhere():
    text = WORKFLOW.read_text()
    assert AUTHORIZATION_HEADING in LAUNCHER.read_text()
    # One freeze-field literal plus the two executable auth-count checks.
    assert text.count(AUTHORIZATION_HEADING) == 3
    assert "## #710 sole L4 memory-safe harness authorization" in text  # predecessor failure evidence only
    assert "## #710 sole L4 memory-safe optimization microbenchmark authorization" in text  # predecessor failure evidence only
    assert "AUTHORIZATION_HEADING=## #716 sole L4 authorization-heading guard repair1 authorization" in text
    assert "issues/716/comments?per_page=100" in text


def test_issue716_workflow_is_owner_only_fresh_namespace_attempt1():
    text = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert "workflow_dispatch:" not in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert "[aera-v26-10-issue716-l4-auth-guard-repair1-preauth]" in text
    assert "[aera-v26-10-issue716-l4-auth-guard-repair1-l4]" in text
    assert "[aera-v26-10-issue710-memory-safe-harness-l4]" not in text
    assert "34145700849" in text and "101817171862" in text and "5573729021" in text


def test_issue716_preauth_and_l4_are_separate_and_no_direct_systems_pass():
    text = WORKFLOW.read_text()
    assert "modal_aera_v26_10_issue716_l4_auth_guard_repair1.py::preauth_main" in text
    assert "modal_aera_v26_10_issue716_l4_auth_guard_repair1.py::l4_main" in text
    assert "steps.guard.outputs.mode == 'preauth'" in text
    assert "steps.guard.outputs.mode == 'l4'" in text
    assert "result_absent=true" in text
    assert "systems PASS earned: `false`" in text
    assert "separately preregistered fresh full Transformer-relative E2E systems gate" in text


def test_issue716_no_old_result_namespace_or_trigger_reuse():
    launcher = LAUNCHER.read_text()
    workflow = WORKFLOW.read_text()
    assert "/vol/aera-v26/issue710-v26-10-memory-safe-harness/result.json" not in launcher
    assert "/vol/aera-v26/issue716-v26-10-l4-auth-guard-repair1/result.json" in launcher
    assert "CONSUMED_TRIGGER = 715" in launcher
    assert "CONSUMED_TRIGGER=715" in workflow
