from __future__ import annotations

from pathlib import Path
import hashlib
import inspect

import tam_research.aera_v26_9_issue622_corrected_autocast_dtype_gate as frozen622
import tam_research.aera_v26_9_issue625_schema_guard_repair1 as repair

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MAIN = "7c4dd6ac038943e882035ff92a6336a764369c61"
SOURCE_TREE = "9f2977e3f4698593b4c030352aa1561e3b4ad33d"
ISSUE622_PROBE_BLOB = "4e08ac9af18f666f09009e4d2c5822b11e91c2c1"
ISSUE622_LAUNCHER_BLOB = "e0fa3b0856b9750402209c6487f407b189672436"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE602_PROBE_BLOB = "456203f515d67d1c92b0a9c3e0e59ce4137ac10a"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue625_frozen_lineage_exact() -> None:
    assert repair.SOURCE_MAIN == SOURCE_MAIN
    assert repair.SOURCE_TREE == SOURCE_TREE
    expected = {
        ROOT / "tam_research/aera_v26_9_issue622_corrected_autocast_dtype_gate.py": ISSUE622_PROBE_BLOB,
        ROOT / "modal_aera_v26_9_issue622_corrected_autocast_dtype_gate_l4_app.py": ISSUE622_LAUNCHER_BLOB,
        ROOT / "tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py": V26_9_BACKEND_BLOB,
        ROOT / "tam_research/aera_v26_9_issue602_identity_weight_visibility_probe.py": ISSUE602_PROBE_BLOB,
    }
    assert {str(path.relative_to(ROOT)): _blob(path) for path in expected} == {
        str(path.relative_to(ROOT)): sha for path, sha in expected.items()
    }


def test_issue625_cpu_contract_is_zero_gpu_and_same_scientific_surface() -> None:
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
    assert protocol["schema_guard_repair_only"] is True
    assert protocol["issue602_identity_field"] == "protocol.research_issue"
    assert protocol["design_seed"] == 891_475_817
    assert protocol["batch_sizes"] == (8, 64)
    assert protocol["validity_kinds"] == ("mixed", "full")
    assert protocol["projected_query_dtype"] == "float32"
    assert protocol["similarity_dtype"] == "bfloat16"
    assert protocol["atol"] == 1e-2 and protocol["rtol"] == 1e-2


def test_issue625_loader_changes_only_identity_location() -> None:
    source = inspect.getsource(repair.load_issue602_preserved_authority_issue625)
    assert 'protocol = payload.get("protocol")' in source
    assert 'protocol.get("research_issue") != 602' in source
    assert 'payload.get("research_issue")' not in source
    frozen_source = inspect.getsource(frozen622.load_issue602_preserved_authority)
    for token in (
        'payload.get("decision") != "FAIL"',
        'payload.get("overall_pass") is not False',
        'gate_meta.get("v26_9_backend_blob")',
        'preserved.get("decision") != "PASS"',
        'mixed.get("timing_decision_bearing") is not False',
    ):
        assert token in source
        assert token in frozen_source


def test_issue625_wrapper_restores_frozen_loader_and_reuses_row_evaluator() -> None:
    source = inspect.getsource(repair.run_schema_guard_repair1_gate_v26_9_issue625)
    assert "original_loader = frozen622.load_issue602_preserved_authority" in source
    assert "frozen622.load_issue602_preserved_authority = load_issue602_preserved_authority_issue625" in source
    assert "frozen622.run_corrected_autocast_dtype_gate_v26_9_issue622" in source
    assert "finally:" in source
    assert "frozen622.load_issue602_preserved_authority = original_loader" in source
    assert "_run_preserved_issue558_surface" not in source
    assert "_advance_generator_through_issue558_regular_rows" not in source


def test_issue625_launcher_and_workflow_keep_one_shot_boundary() -> None:
    launcher = (ROOT / "modal_aera_v26_9_issue625_schema_guard_repair1_l4_app.py").read_text()
    workflow = (ROOT / ".github/workflows/aera-v26-9-issue625-schema-guard-repair1-l4.yml").read_text()
    assert launcher.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in launcher
    assert 'RESULT_PATH = "/vol/aera-v26/issue625-schema-guard-repair1/result.json"' in launcher
    assert "volume.commit()" in launcher
    assert workflow.count("modal run modal_aera_v26_9_issue625_schema_guard_repair1_l4_app.py") == 1
    assert "workflow_dispatch" not in workflow
    assert "cancel-in-progress" not in workflow
    assert "GITHUB_RUN_ATTEMPT" in workflow
    assert "## #625 sole L4 schema-guard repair1 authorization" in workflow
    assert "[aera-v26-9-issue625-schema-guard-repair1-l4]" in workflow
    assert "33808524208" in workflow and "100824653900" in workflow
    assert "modal deploy" not in workflow.lower()


def test_issue625_higher_authorizations_remain_false() -> None:
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