from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from tam_research import aera_v26_5_end_to_end_systems as base

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_10_issue710_memory_safe_harness.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-10-issue710-memory-safe-harness.yml"
FROZEN_706_LAUNCHER = ROOT / "modal_aera_v26_10_issue706_latent_depth_sync_coalescing.py"
FROZEN_IMPL = ROOT / "tam_research/aera_hardware_core_v26_10_latent_depth_sync_coalescing.py"

LAUNCHER_BLOB = "b885250753ea169cd89dd7a978bb3647fd261fe8"
WORKFLOW_BLOB = "53e1e22c0b421c51ef6299e777132d7a21dc771d"
ISSUE706_LAUNCHER_BLOB = "e23072aa83b0bc20ccd72e7567f234e9f4a931f6"
V26_10_IMPL_BLOB = "d8f691c198eed1fa96bcbb78a4e76cad82d18779"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _load_chunked_comparator():
    tree = ast.parse(LAUNCHER.read_text())
    fn = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_chunked_logit_equivalence"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            fn,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    ns: dict[str, Any] = {
        "Any": Any,
        "INTEGRATED_ATOL": 1e-2,
        "INTEGRATED_RTOL": 1e-2,
        "LOGIT_COMPARE_BATCH_CHUNK": 1,
    }
    exec(compile(module, str(LAUNCHER), "exec"), ns)
    return ns["_chunked_logit_equivalence"]


def _assert_same_result(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    chunked = _load_chunked_comparator()(reference, candidate)
    frozen = base._logit_equivalence(reference, candidate)
    assert chunked.keys() == frozen.keys()
    for key in ("pass", "allclose", "dtype_device_shape_exact", "atol", "rtol"):
        assert chunked[key] == frozen[key]
    if math.isnan(frozen["max_abs"]):
        assert math.isnan(chunked["max_abs"])
    else:
        assert chunked["max_abs"] == pytest.approx(frozen["max_abs"], abs=0.0, rel=0.0)


def test_issue710_exact_files_and_frozen_candidate_bytes():
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    assert _git_blob(FROZEN_706_LAUNCHER) == ISSUE706_LAUNCHER_BLOB
    assert _git_blob(FROZEN_IMPL) == V26_10_IMPL_BLOB
    ast.parse(LAUNCHER.read_text())


def test_issue710_chunked_comparator_exact_pass():
    x = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 10
    _assert_same_result(x, x.clone())


def test_issue710_chunked_comparator_tolerance_pass_and_fail():
    ref = torch.ones((3, 4, 5), dtype=torch.float32)
    close = ref.clone()
    close[2, 3, 4] += 0.019
    _assert_same_result(ref, close)
    assert _load_chunked_comparator()(ref, close)["pass"] is True

    far = ref.clone()
    far[2, 3, 4] += 0.05
    _assert_same_result(ref, far)
    assert _load_chunked_comparator()(ref, far)["pass"] is False


def test_issue710_chunked_comparator_metadata_mismatch_matches_frozen():
    ref = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    cand = ref.to(torch.float64)
    _assert_same_result(ref, cand)
    result = _load_chunked_comparator()(ref, cand)
    assert result["dtype_device_shape_exact"] is False
    assert result["pass"] is False


def test_issue710_chunked_comparator_nan_semantics_match_frozen():
    ref = torch.zeros((3, 2, 2), dtype=torch.float32)
    cand = ref.clone()
    ref[2, 1, 1] = float("nan")
    cand[2, 1, 1] = float("nan")
    _assert_same_result(ref, cand)
    result = _load_chunked_comparator()(ref, cand)
    assert result["allclose"] is False
    assert math.isnan(result["max_abs"])


def test_issue710_global_max_can_live_in_later_chunk():
    ref = torch.zeros((4, 2, 3), dtype=torch.float32)
    cand = ref.clone()
    cand[0, 0, 0] = 0.02
    cand[3, 1, 2] = 0.25
    _assert_same_result(ref, cand)
    result = _load_chunked_comparator()(ref, cand)
    assert result["max_abs"] == pytest.approx(0.25)


def test_issue710_launcher_is_harness_only_and_preserves_frozen_gates():
    text = LAUNCHER.read_text()
    for required in (
        'SOURCE_MAIN = "c2e9c17d7bb34d7652fc6759c80193f2d951f2ce"',
        'SOURCE_TREE = "ff3913de8e605aba10f87839676c9b582304233c"',
        'RESEARCH_ISSUE = 710',
        'CONSUMED_TRIGGER = 709',
        'CONSUMED_FAILURE_COMMENT = 5573348027',
        'CONSUMED_PRECISION_COMMENT = 5573367269',
        'CONSUMED_RUN = 34141205064',
        'CONSUMED_JOB = 101803372290',
        'CONSUMED_ATTEMPT = 1',
        'LOGIT_COMPARE_BATCH_CHUNK = 1',
        'RESULT_PATH = "/vol/aera-v26/issue710-v26-10-memory-safe-harness/result.json"',
        'MIN_LATENCY_IMPROVEMENT = frozen706.MIN_LATENCY_IMPROVEMENT',
        'MIN_STREAM_SYNCHRONIZE_REDUCTION = frozen706.MIN_STREAM_SYNCHRONIZE_REDUCTION',
        'WARMUP_CALLS = frozen706.WARMUP_CALLS',
        'TIMED_CALLS_PER_CONDITION = frozen706.TIMED_CALLS_PER_CONDITION',
        'PROFILE_CALLS_PER_CONDITION = frozen706.PROFILE_CALLS_PER_CONDITION',
        'logit_eq = _chunked_logit_equivalence(baseline_logits, candidate_logits)',
        'del baseline_output, candidate_output, baseline_logits, candidate_logits',
        'samples[name].append(frozen706._event_timed_call(call))',
        'baseline_profile = frozen706._trace_sync_count(',
        'candidate_profile = frozen706._trace_sync_count(',
        '"systems_pass_earned": False',
        '"architecture_freeze_authorized": False',
        '"100m_authorized": False',
        '"breakthrough_proven": False',
    ):
        assert required in text
    assert 'base._logit_equivalence(' not in text
    assert '.backward(' not in text
    assert 'torch.optim.' not in text
    assert '.step(' not in text


def test_issue710_workflow_is_owner_only_fresh_namespace_and_attempt_one():
    text = WORKFLOW.read_text()
    assert 'issues:\n    types: [opened]' in text
    assert 'github.event.issue.user.login == github.repository_owner' in text
    assert 'workflow_dispatch:' not in text
    assert 'pull_request:' not in text
    assert 'push:' not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert '[aera-v26-10-issue710-memory-safe-harness-preauth]' in text
    assert '[aera-v26-10-issue710-memory-safe-harness-l4]' in text
    assert '5573348027' in text and '5573367269' in text
    assert '34141205064' in text and '101803372290' in text
    assert '## #710 sole L4 memory-safe harness authorization' in text
    assert 'AERA_V26_10_ISSUE710_PREAUTH_JSON=' in text
    assert 'AERA_V26_10_ISSUE710_SUMMARY_JSON=' in text


def test_issue710_workflow_requires_exact_freeze_fields():
    text = WORKFLOW.read_text()
    for required in (
        'SOURCE_MAIN=c2e9c17d7bb34d7652fc6759c80193f2d951f2ce',
        'SOURCE_TREE=ff3913de8e605aba10f87839676c9b582304233c',
        'BRANCH=infra/aera-v26-10-issue710-memory-safe-harness-repair1',
        'RESULT_PATH=/vol/aera-v26/issue710-v26-10-memory-safe-harness/result.json',
        'ISSUE706_LAUNCHER_BLOB=e23072aa83b0bc20ccd72e7567f234e9f4a931f6',
        'V26_10_IMPL_BLOB=d8f691c198eed1fa96bcbb78a4e76cad82d18779',
        'LOGIT_COMPARE_BATCH_CHUNK=1',
        'MIN_LATENCY_IMPROVEMENT=0.05',
        'MIN_STREAM_SYNCHRONIZE_REDUCTION=20',
        'INTEGRATED_ATOL=0.01',
        'INTEGRATED_RTOL=0.01',
        'LAUNCHER_BLOB=',
        'WORKFLOW_BLOB=',
        'CPU_TEST_BLOB=',
        'FROZEN_TREE=',
        'FROZEN_COMMIT=',
    ):
        assert required in text
