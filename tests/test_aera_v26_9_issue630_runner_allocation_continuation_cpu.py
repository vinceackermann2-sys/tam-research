from __future__ import annotations

from pathlib import Path
import hashlib

import tam_research.aera_v26_9_issue625_schema_guard_repair1 as repair

ROOT = Path(__file__).resolve().parents[1]
ISSUE625_REPAIR_BLOB = "92d06a4954bca1b302355e81f5bf09b06fcee222"
ISSUE625_LAUNCHER_BLOB = "01fdb99f9b072d4622b9b176423a6e2453b13324"
ISSUE625_WORKFLOW_BLOB = "da67fa2699b2b413c3fb3270bf0d7ba590fa39d2"
ISSUE625_CPU_TEST_BLOB = "63c6c61a75b9e88e4391711fcbe37e4e0b802457"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue630_frozen_scientific_source_is_exact() -> None:
    expected = {
        ROOT / "tam_research/aera_v26_9_issue625_schema_guard_repair1.py": ISSUE625_REPAIR_BLOB,
        ROOT / "modal_aera_v26_9_issue625_schema_guard_repair1_l4_app.py": ISSUE625_LAUNCHER_BLOB,
        ROOT / ".github/workflows/aera-v26-9-issue625-schema-guard-repair1-l4.yml": ISSUE625_WORKFLOW_BLOB,
        ROOT / "tests/test_aera_v26_9_issue625_schema_guard_repair1_cpu.py": ISSUE625_CPU_TEST_BLOB,
        ROOT / "tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py": V26_9_BACKEND_BLOB,
    }
    assert {str(path.relative_to(ROOT)): _blob(path) for path in expected} == {
        str(path.relative_to(ROOT)): sha for path, sha in expected.items()
    }


def test_issue630_reuses_exact_issue625_cpu_scientific_contract() -> None:
    contract = repair.cpu_contract_preflight_issue625()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_constructed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["corpus_accessed"] is False
    assert contract["training_performed"] is False
    assert contract["optimizer_created"] is False
    assert contract["backward_performed"] is False
    assert contract["scientific_seed_consumed"] is False
    protocol = repair.issue625_protocol()
    assert protocol["design_seed"] == 891_475_817
    assert protocol["batch_sizes"] == (8, 64)
    assert protocol["validity_kinds"] == ("mixed", "full")
    assert protocol["projected_query_dtype"] == "float32"
    assert protocol["similarity_dtype"] == "bfloat16"
    assert protocol["atol"] == 1e-2
    assert protocol["rtol"] == 1e-2


def test_issue630_launcher_is_orchestration_only_and_one_shot() -> None:
    launcher = (
        ROOT / "modal_aera_v26_9_issue630_runner_allocation_continuation_l4_app.py"
    ).read_text()
    assert launcher.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in launcher
    assert (
        'RESULT_PATH = "/vol/aera-v26/issue630-runner-allocation-continuation/result.json"'
        in launcher
    )
    assert launcher.count("run_schema_guard_repair1_gate_v26_9_issue625(") == 1
    assert "volume.commit()" in launcher
    assert "ISSUE625_REPAIR_BLOB" in launcher
    assert "load_issue602_preserved_authority_issue625" in (
        ROOT / "tam_research/aera_v26_9_issue625_schema_guard_repair1.py"
    ).read_text()


def test_issue630_workflow_guards_consumed_prerunner_boundary() -> None:
    workflow = (
        ROOT / ".github/workflows/aera-v26-9-issue630-runner-allocation-continuation-l4.yml"
    ).read_text()
    assert "workflow_dispatch" not in workflow
    assert "cancel-in-progress" not in workflow
    assert "GITHUB_RUN_ATTEMPT" in workflow
    assert "## #630 sole L4 runner-allocation continuation authorization" in workflow
    assert "[aera-v26-9-issue630-runner-allocation-continuation-l4]" in workflow
    assert "33846259975" in workflow
    assert "100938724271" in workflow
    assert "runner_id" in workflow
    assert "runner_name" in workflow
    assert ".steps | length" in workflow
    assert workflow.count(
        "modal run modal_aera_v26_9_issue630_runner_allocation_continuation_l4_app.py"
    ) == 1
    assert "modal deploy" not in workflow.lower()


def test_issue630_higher_authorizations_remain_false() -> None:
    protocol = repair.issue625_protocol()
    false_keys = (
        "gpu_authorized_by_probe_module",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
        "scientific_seed_consumed",
    )
    assert all(protocol[key] is False for key in false_keys)
