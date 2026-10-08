from __future__ import annotations

from collections import Counter
import ast
import json
import re
from pathlib import Path

import pytest

from tam_research.chm_v3_100m_daec_position_oracle import (
    ARCHIVED_RUN_ID, ARCHIVED_JOB_ID, analyze_position_oracle, sha256_json,
)
from tam_research.chm_v3_100m_daec_copy_candidate_overlap import FROZEN_COPY_TOKEN_COUNTS

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "tests/fixtures/chm_v3_1340_archived_hard_copy_traces.json"
SCRIPT = ROOT / "scripts/chm_v3_1340_cpu_position_oracle.py"
WORKFLOW = ROOT / ".github/workflows/chm-v3-daec-position-oracle-1340-cpu-v1.yml"


def _fixture() -> dict:
    return json.loads(ARCHIVE.read_text(encoding="utf-8"))


def test_archived_512_scored_probe_traces_are_consumed_and_immutable() -> None:
    a = _fixture()
    assert a["archived_actions_run_id"] == ARCHIVED_RUN_ID == 37760374218
    assert a["archived_actions_job_id"] == ARCHIVED_JOB_ID == 113255052797
    assert a["source_sha"] == "eb4f0e36ee6c4095e550f26cb7a52c20c5116aa9"
    assert a["scientific_seed"] == 2013161
    assert a["scientific_seed_consumed"] is True
    assert a["scientific_classification"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert a["scientific_passed"] is False
    assert a["generator_version"] == "chm-v1-100m-heldout-aligned-v4"
    assert a["probe_seed"] == 977301
    assert a["counts"] == {
        "probes": 512, "training_tokens_per_model": 33554432,
        "optimizer_steps_per_model": 2048,
    }
    assert len(a["rows"]) == 512
    family_counts = Counter(r["family"] for r in a["rows"])
    assert family_counts == Counter({
        "rare_fact": 128, "overwrite": 128,
        "two_hop": 128, "local_negative": 128,
    })
    memory = [r for r in a["rows"] if r["memory_count"]]
    assert len(memory) == 384
    assert all(r["memory_count"] in (512, 1024) for r in memory)
    assert Counter(r["copied_token_id"] for r in memory) == FROZEN_COPY_TOKEN_COUNTS
    assert all(not r["copied_matches_answer"] for r in memory)
    assert all(r["first_hop_index"] is None and r["second_hop_index"] is None
               and r["copied_token_id"] is None
               for r in a["rows"] if r["family"] == "local_negative")


def test_reconstruction_requires_exact_original_source_and_stopped_seed() -> None:
    original = _fixture()
    for key, bad in (
        ("source_sha", "0" * 40), ("scientific_seed", 2013162),
        ("scientific_seed_consumed", False),
        ("scientific_classification", "CHM_V3_100M_DAEC_POSITIVE_DEVELOPMENT_SCREEN"),
        ("scientific_passed", True),
        ("generator_version", "some-other-generator"),
        ("archived_actions_run_id", 0),
    ):
        with pytest.raises(ValueError):
            analyze_position_oracle({**original, key: bad}, lambda x: [3])
    with pytest.raises(ValueError, match="512"):
        analyze_position_oracle({**original, "rows": original["rows"][:511]}, lambda x: [3])


def test_position_oracle_helper_is_pure_and_cpu_only() -> None:
    import inspect
    import tam_research.chm_v3_100m_daec_position_oracle as helper
    src = inspect.getsource(helper)
    assert "modal.App(" not in src
    assert "optimizer.step(" not in src
    assert "torch.cuda" not in src
    assert "run_scientific(" not in src
    assert "write_text(" not in src
    assert len(sha256_json({"a": 1})) == 64
    ast.parse(SCRIPT.read_text(encoding="utf-8"))


def test_cpu_workflow_is_once_only_and_has_no_gpu_or_modal_dispatch() -> None:
    src = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in src
    assert "types: [opened]" in src
    assert "[chm-v3-daec-position-oracle-1340-cpu-v1]" in src
    assert 'test "$RUN_ATTEMPT" = "1"' in src
    assert 'github.event.issue.user.login == github.repository_owner' in src
    assert "torch==2.10.0" in src
    assert "tiktoken" in src
    assert "scripts/chm_v3_1340_cpu_position_oracle.py" in src
    assert "chm_v3_1340_archived_hard_copy_traces.json" in src
    assert "CHM_V3_1340_CPU_POSITION_ORACLE_AUDIT_PASS" in src
    assert "gpu_allocated=false" in src
    assert "new_scientific_attempt=false" in src
    assert "modal run" not in src
    assert "modal deploy" not in src
    assert "gpu: L4" not in src
    assert "MODAL_TOKEN" not in src
    expressions = [line for line in src.splitlines() if "${{" in line]
    assert len(expressions) == 5
    for line in expressions:
        assert re.search(r"\$\{\{\s*[^{}\n]+\s*\}\}", line), line


def test_actual_frozen_gpt2_reconstruction_when_tokenizer_is_installed() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    report = analyze_position_oracle(_fixture(), enc.encode, decode=enc.decode)
    assert report["classification"] == "CHM_V3_1340_CPU_ONLY_POSITION_ORACLE_FORENSICS"
    assert report["memory_probe_count"] == 384
    assert report["copy_hit_count"] == 0
    assert report["copy_candidate_overlap_count"] == 0
    assert report["new_training_or_gpu_authorized"] is False
    assert report["historical_trained_checkpoint_parity_proved"] is False
    assert report["scientific_result_reclassified"] is False
