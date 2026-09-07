from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_10_issue724_source_bound_provenance_repair1.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-10-issue724-source-bound-provenance-repair1.yml"

LAUNCHER_BLOB = "c58b9988d28c1a3a4c9bb9d1ec8f704dfd377285"
WORKFLOW_BLOB = "c0955686c86611c465b680079c0bf4e582020590"
INHERITED = {
    "modal_aera_v26_10_issue710_memory_safe_harness.py": "b885250753ea169cd89dd7a978bb3647fd261fe8",
    "tam_research/aera_hardware_core_v26_10_latent_depth_sync_coalescing.py": "d8f691c198eed1fa96bcbb78a4e76cad82d18779",
    "modal_aera_v26_10_issue716_l4_auth_guard_repair1.py": "84cd58e1f3eed3a51efaff2d7cce2e674c29405f",
    ".github/workflows/aera-v26-10-issue716-l4-auth-guard-repair1.yml": "4dbdb80245b6767fc07931c45190663d8c9197af",
    "tests/test_aera_v26_10_issue716_l4_auth_guard_repair1_cpu.py": "13f72d319bf685e33978e15b6d88344bc5e7d565",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue724_files_parse_and_inherited_blobs_are_exact():
    assert LAUNCHER.exists() and WORKFLOW.exists()
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    ast.parse(LAUNCHER.read_text())
    for rel, expected in INHERITED.items():
        path = ROOT / rel
        assert path.exists(), rel
        assert _git_blob(path) == expected, rel


def test_issue724_wrapper_separates_frozen_source_from_execution_binding():
    text = LAUNCHER.read_text()
    for required in (
        'SOURCE_MAIN = "8371d9ab3988df33c7b92b529016618ebf60f9ee"',
        'SOURCE_TREE = "fd171423b5878c9082e689d7d83da9f8c7c81569"',
        'ORCHESTRATION_BASE_MAIN = "67344b9535dd305f1478ea6ea692fe45a64c86d8"',
        'ORCHESTRATION_BASE_TREE = "a208565bdad203e2cfb2354f53c07d022b977d85"',
        'RESULT_PATH = "/vol/aera-v26/issue724-v26-10-source-bound-provenance-repair1/result.json"',
        'RESEARCH_ISSUE = 724',
        'CONSUMED_TRIGGER = 723',
        'CONSUMED_FAILURE_COMMENT = 5574302025',
        'CONSUMED_RUN = 34150467379',
        'CONSUMED_JOB = 101831501185',
        'CONSUMED_ATTEMPT = 1',
        'AUTHORIZATION_HEADING = "## #724 sole L4 source-bound provenance repair1 authorization"',
        'BOUND_MAIN_ENV = "AERA_ISSUE724_BOUND_MAIN"',
        'out["source_main"] = SOURCE_MAIN',
        'out["bound_main"] = bound_main',
        'def preflight(bound_main: str)',
        'def run_microbenchmark(bound_main: str)',
        'evidence = preflight.remote(bound_main)',
        'summary = run_microbenchmark.remote(bound_main)',
    ):
        assert required in text
    assert 'out["source_main"] = bound_main' not in text
    assert 'SOURCE_MAIN = ORCHESTRATION_BASE_MAIN' not in text


def test_issue724_wrapper_preserves_scientific_and_performance_boundary():
    text = LAUNCHER.read_text()
    for required in (
        'ISSUE710_LAUNCHER_BLOB = "b885250753ea169cd89dd7a978bb3647fd261fe8"',
        'V26_10_IMPL_BLOB = "d8f691c198eed1fa96bcbb78a4e76cad82d18779"',
        'LOGIT_COMPARE_BATCH_CHUNK = frozen710.LOGIT_COMPARE_BATCH_CHUNK',
        'BATCHES = tuple(frozen710.BATCHES)',
        'WARMUP_CALLS = frozen710.WARMUP_CALLS',
        'TIMED_CALLS_PER_CONDITION = frozen710.TIMED_CALLS_PER_CONDITION',
        'PROFILE_CALLS_PER_CONDITION = frozen710.PROFILE_CALLS_PER_CONDITION',
        'MIN_LATENCY_IMPROVEMENT = frozen710.MIN_LATENCY_IMPROVEMENT',
        'MIN_STREAM_SYNCHRONIZE_REDUCTION = frozen710.MIN_STREAM_SYNCHRONIZE_REDUCTION',
        'INTEGRATED_ATOL = frozen710.INTEGRATED_ATOL',
        'INTEGRATED_RTOL = frozen710.INTEGRATED_RTOL',
        'with _successor_identity():\n        inherited = frozen710.preflight.local()',
        'with _successor_identity():\n        summary = frozen710.run_microbenchmark.local()',
    ):
        assert required in text
    forbidden = (
        '.backward(', 'torch.optim.', '.step(', '.zero_grad(',
        'MIN_LATENCY_IMPROVEMENT = 0.', 'MIN_STREAM_SYNCHRONIZE_REDUCTION = 0',
    )
    for token in forbidden:
        assert token not in text


def test_issue724_patches_identity_and_result_only_then_restores():
    tree = ast.parse(LAUNCHER.read_text())
    assign = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == '_PATCH_FIELDS' for t in node.targets)
    )
    assert isinstance(assign.value, ast.Dict)
    keys = {k.value for k in assign.value.keys if isinstance(k, ast.Constant)}
    assert keys == {
        'RESULT_PATH','SOURCE_MAIN','SOURCE_TREE','RESEARCH_ISSUE',
        'CONSUMED_TRIGGER','CONSUMED_FAILURE_COMMENT','CONSUMED_RUN','CONSUMED_JOB',
        'CONSUMED_ATTEMPT','PREAUTH_MARKER','L4_START_MARKER','RESULT_MARKER','SUMMARY_MARKER',
    }
    text = LAUNCHER.read_text()
    assert 'finally:' in text
    assert 'setattr(frozen710, name, value)' in text


def test_issue724_workflow_is_owner_only_fresh_namespace_and_one_attempt():
    text = WORKFLOW.read_text()
    assert 'issues:\n    types: [opened]' in text
    assert 'github.event.issue.user.login == github.repository_owner' in text
    assert 'workflow_dispatch:' not in text
    assert 'pull_request:' not in text
    assert 'push:' not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert '[aera-v26-10-issue724-source-bound-provenance-repair1-preauth]' in text
    assert '[aera-v26-10-issue724-source-bound-provenance-repair1-l4]' in text
    assert '## #724 sole L4 source-bound provenance repair1 authorization' in text
    assert 'AERA_ISSUE724_BOUND_MAIN: ${{ steps.guard.outputs.bound_main }}' in text


def test_issue724_workflow_explicitly_rejects_issue723_provenance_bug():
    text = WORKFLOW.read_text()
    assert "'source_main':'8371d9ab3988df33c7b92b529016618ebf60f9ee'" in text
    assert "'bound_main':os.environ['BOUND_MAIN']" in text
    assert "'source_main':os.environ['BOUND_MAIN']" not in text
    assert "#716 preauth drift source_main: '8371d9ab3988df33c7b92b529016618ebf60f9ee'" in text
    assert 'CONSUMED_FAILURE_COMMENT=5574302025' in text
    assert 'CONSUMED_RUN=34150467379' in text
    assert 'CONSUMED_JOB=101831501185' in text


def test_issue724_workflow_freezes_all_inherited_thresholds():
    text = WORKFLOW.read_text()
    for required in (
        'ISSUE710_LAUNCHER_BLOB=b885250753ea169cd89dd7a978bb3647fd261fe8',
        'V26_10_IMPL_BLOB=d8f691c198eed1fa96bcbb78a4e76cad82d18779',
        'LOGIT_COMPARE_BATCH_CHUNK=1', 'BATCHES=8,64', 'WARMUP_CALLS=3',
        'TIMED_CALLS_PER_CONDITION=20', 'PROFILE_CALLS_PER_CONDITION=1',
        'MIN_LATENCY_IMPROVEMENT=0.05', 'MIN_STREAM_SYNCHRONIZE_REDUCTION=20',
        'INTEGRATED_ATOL=0.01', 'INTEGRATED_RTOL=0.01',
    ):
        assert required in text or required in LAUNCHER.read_text()


def test_issue724_evidence_never_auto_earns_systems_pass():
    launcher = LAUNCHER.read_text()
    workflow = WORKFLOW.read_text()
    for flag in (
        'systems_pass_earned','architecture_freeze_authorized','s2_authorized',
        'fresh_scientific_seed_authorized','independent_replication_credit',
        '100m_authorized','breakthrough_proven',
    ):
        assert flag in launcher
    assert 'A PASS authorizes only a separately preregistered fresh full frozen Transformer-relative E2E systems gate.' in workflow
    assert 'It does not itself earn systems PASS or higher-stage authority.' in workflow
