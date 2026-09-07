from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_10_issue724_source_bound_provenance_repair1.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-10-issue724-source-bound-provenance-repair1.yml"
SELF = ROOT / "tests/test_aera_v26_10_issue724_source_bound_provenance_repair1_cpu.py"

LAUNCHER_BLOB = "c58b9988d28c1a3a4c9bb9d1ec8f704dfd377285"
WORKFLOW_BLOB = "58ff1e03dc1f0e21d21452288fbe739e51ccd0a4"
PREVIOUS_WORKFLOW_BLOB = "c0955686c86611c465b680079c0bf4e582020590"
PREVIOUS_CPU_TEST_BLOB = "9227b776736ae63344c38d51e1197b31ab1b1447"
INHERITED = {
    "modal_aera_v26_10_issue710_memory_safe_harness.py": "b885250753ea169cd89dd7a978bb3647fd261fe8",
    "tam_research/aera_hardware_core_v26_10_latent_depth_sync_coalescing.py": "d8f691c198eed1fa96bcbb78a4e76cad82d18779",
    "modal_aera_v26_10_issue716_l4_auth_guard_repair1.py": "84cd58e1f3eed3a51efaff2d7cce2e674c29405f",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue727_changes_only_workflow_contract_while_wrapper_and_science_stay_exact():
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    ast.parse(LAUNCHER.read_text())
    for rel, expected in INHERITED.items():
        assert _git_blob(ROOT / rel) == expected, rel


def test_issue727_workflow_uses_three_distinct_freeze_layers():
    text = WORKFLOW.read_text()
    assert '## #724 pre-implementation source-bound provenance repair1 freeze' in text
    assert '## #724 correction1 CPU static-contract freeze after consumed #725' in text
    assert '## #727 pre-implementation stale immutable-guard pointer repair1 freeze' in text
    assert 'ORIGINAL_FREEZE_COMMENT=5574367353' in text
    assert 'CORRECTION1_FREEZE_COMMENT=5574467360' in text
    assert 'CORRECTION1_CPU_TEST_BLOB=9227b776736ae63344c38d51e1197b31ab1b1447' in text
    assert 'CORRECTION1_TREE=8876c7f62082626b3ba260b341709dd3bdabca67' in text
    assert 'CORRECTION1_COMMIT=907089eba01315dd7908fd01c949eb62c31179bc' in text
    assert 'PREVIOUS_WORKFLOW_BLOB=c0955686c86611c465b680079c0bf4e582020590' in text
    assert 'PREVIOUS_CPU_TEST_BLOB=9227b776736ae63344c38d51e1197b31ab1b1447' in text
    assert 'workflow_blob="$(printf' in text
    assert 'cpu_test_blob="$(printf' in text
    assert 'repair_commit="$(printf' in text
    assert 'git merge-base --is-ancestor "${repair_commit}" "${bound_main}"' in text


def test_issue727_does_not_revert_to_consumed_original_current_lineage():
    text = WORKFLOW.read_text()
    # Original #724 freeze remains evidence for science/provenance constants only.
    # Current-file/tree/commit checks come from correction1 + #727 repair freeze.
    assert 'CORRECTION1_CPU_TEST_BLOB=9227b776736ae63344c38d51e1197b31ab1b1447' in text
    assert 'FROZEN_TREE=8876c7f62082626b3ba260b341709dd3bdabca67' in text
    assert 'FROZEN_COMMIT=907089eba01315dd7908fd01c949eb62c31179bc' in text
    assert 'd32131108418097ee1e8dd652d10f1e60da14888' not in text
    assert '821ac66b355f4702bdb9be710fb37f90968cf93d' not in text
    assert '008d207dea7b21194b5e622a6c6004012f03ba73' not in text


def test_issue727_keeps_unused_issue724_trigger_and_result_namespaces_exact():
    text = WORKFLOW.read_text()
    assert '[aera-v26-10-issue724-source-bound-provenance-repair1-preauth]' in text
    assert '[aera-v26-10-issue724-source-bound-provenance-repair1-l4]' in text
    assert '/vol/aera-v26/issue724-v26-10-source-bound-provenance-repair1/result.json' in text
    assert '## #724 sole L4 source-bound provenance repair1 authorization' in text
    assert '🔎 **AERA-v26.10 #724 source-bound preauthorization evidence**' in text
    assert '⚙️ **AERA-v26.10 #724 optimization microbenchmark evidence**' in text


def test_issue727_preserves_source_vs_bound_provenance_semantics():
    text = WORKFLOW.read_text()
    assert "'source_main':'8371d9ab3988df33c7b92b529016618ebf60f9ee'" in text
    assert "'source_tree':'fd171423b5878c9082e689d7d83da9f8c7c81569'" in text
    assert "'bound_main':os.environ['BOUND_MAIN']" in text
    assert "'source_main':os.environ['BOUND_MAIN']" not in text
    assert 'AERA_ISSUE724_BOUND_MAIN: ${{ steps.guard.outputs.bound_main }}' in text
    launcher = LAUNCHER.read_text()
    assert 'out["source_main"] = SOURCE_MAIN' in launcher
    assert 'out["bound_main"] = bound_main' in launcher
    assert 'out["source_main"] = bound_main' not in launcher


def test_issue727_preserves_all_scientific_and_performance_thresholds():
    text = WORKFLOW.read_text()
    for required in (
        'ISSUE710_LAUNCHER_BLOB=b885250753ea169cd89dd7a978bb3647fd261fe8',
        'V26_10_IMPL_BLOB=d8f691c198eed1fa96bcbb78a4e76cad82d18779',
        'LOGIT_COMPARE_BATCH_CHUNK=1',
        'BATCHES=8,64',
        'WARMUP_CALLS=3',
        'TIMED_CALLS_PER_CONDITION=20',
        'PROFILE_CALLS_PER_CONDITION=1',
        'MIN_LATENCY_IMPROVEMENT=0.05',
        'MIN_STREAM_SYNCHRONIZE_REDUCTION=20',
        'INTEGRATED_ATOL=0.01',
        'INTEGRATED_RTOL=0.01',
    ):
        assert required in text
    launcher = LAUNCHER.read_text()
    assert 'LOGIT_COMPARE_BATCH_CHUNK = frozen710.LOGIT_COMPARE_BATCH_CHUNK' in launcher
    assert 'MIN_LATENCY_IMPROVEMENT = frozen710.MIN_LATENCY_IMPROVEMENT' in launcher
    assert 'MIN_STREAM_SYNCHRONIZE_REDUCTION = frozen710.MIN_STREAM_SYNCHRONIZE_REDUCTION' in launcher


def test_issue727_workflow_is_owner_only_attempt1_and_has_no_manual_retry_surface():
    text = WORKFLOW.read_text()
    assert 'issues:\n    types: [opened]' in text
    assert 'github.event.issue.user.login == github.repository_owner' in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert 'workflow_dispatch:' not in text
    assert 'pull_request:' not in text
    assert 'push:' not in text
    assert 'modal deploy' not in text


def test_issue727_is_static_orchestration_only_and_wrapper_unchanged():
    text = WORKFLOW.read_text()
    launcher = LAUNCHER.read_text()
    assert 'python -m py_compile modal_aera_v26_10_issue724_source_bound_provenance_repair1.py' in text
    for forbidden in ('.backward(', 'torch.optim.', '.zero_grad('):
        assert forbidden not in text
    assert 'frozen710.run_microbenchmark.local()' in launcher
    assert '_chunked_logit_equivalence' not in launcher
    assert SELF.exists()
