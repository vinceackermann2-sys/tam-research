from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_v26_9_issue602_identity_weight_visibility_probe as issue602
from tam_research import aera_v26_9_issue622_corrected_autocast_dtype_gate as probe
from tam_research import aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as v26_9

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MAIN = "caa7b019e9232d607d69b0e422e6d9550d675ff4"
SOURCE_TREE = "fd76f479a16036bcc81d3e48ba70956fc79c409e"
PROBE_BLOB = "4e08ac9af18f666f09009e4d2c5822b11e91c2c1"
LAUNCHER_BLOB = "e0fa3b0856b9750402209c6487f407b189672436"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE602_PROBE_BLOB = "456203f515d67d1c92b0a9c3e0e59ce4137ac10a"
ISSUE602_RESULT_SHA256 = "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue622_frozen_lineage_and_added_files_exact() -> None:
    assert probe.SOURCE_MAIN == SOURCE_MAIN
    assert probe.SOURCE_TREE == SOURCE_TREE
    assert probe.ISSUE602_RESULT_SHA256 == ISSUE602_RESULT_SHA256
    assert probe.ISSUE602_PROBE_BLOB == ISSUE602_PROBE_BLOB
    assert probe.V26_9_BACKEND_BLOB == V26_9_BACKEND_BLOB

    expected = {
        ROOT / "tam_research/aera_v26_9_issue622_corrected_autocast_dtype_gate.py": PROBE_BLOB,
        ROOT / "modal_aera_v26_9_issue622_corrected_autocast_dtype_gate_l4_app.py": LAUNCHER_BLOB,
        ROOT / "tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py": V26_9_BACKEND_BLOB,
        ROOT / "tam_research/aera_v26_9_issue602_identity_weight_visibility_probe.py": ISSUE602_PROBE_BLOB,
    }
    assert {str(path.relative_to(ROOT)): _blob(path) for path in expected} == {
        str(path.relative_to(ROOT)): sha for path, sha in expected.items()
    }


def test_issue622_cpu_contract_is_zero_gpu_and_fresh_design_only() -> None:
    contract = probe.cpu_contract_preflight_issue622()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["synthetic_only"] is True
    assert contract["model_constructed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["corpus_accessed"] is False
    assert contract["training_performed"] is False
    assert contract["optimizer_created"] is False
    assert contract["backward_performed"] is False
    assert contract["scientific_seed_consumed"] is False

    protocol = probe.issue622_protocol()
    assert protocol["design_seed"] == 891_475_817
    assert protocol["design_seed"] != issue602.DESIGN_SEED
    assert protocol["design_seed_only"] is True
    assert protocol["issue558_fixtures_rerun"] is False
    assert protocol["issue602_fixtures_rerun"] is False
    assert protocol["fresh_rows"] == 4
    assert protocol["batch_sizes"] == (8, 64)
    assert protocol["validity_kinds"] == ("mixed", "full")
    assert protocol["atol"] == 1e-2
    assert protocol["rtol"] == 1e-2
    assert protocol["timing_decision_bearing"] is False


def test_issue622_corrected_dtype_contract_is_exact() -> None:
    protocol = probe.issue622_protocol()
    assert protocol["identity_dtype"] == "float32"
    assert protocol["context_dtype"] == "float32"
    assert protocol["projected_query_dtype"] == "float32"
    assert protocol["similarity_dtype"] == "bfloat16"
    assert protocol["durable_keys_dtype"] == "float32"
    assert protocol["durable_values_dtype"] == "float32"
    assert protocol["durable_strengths_dtype"] == "float32"
    assert protocol["valid_dtype"] == "bool"
    assert protocol["normalized_keys_dtype"] == "float32"

    source = inspect.getsource(probe._corrected_row)
    assert 'row["projected_query_dtype"] == "torch.float32"' in source
    assert 'row["similarity_dtype"] == "torch.bfloat16"' in source
    assert 'row["issue602_dtype_split_exact"]' in source
    assert 'row["non_dtype_pass"]' in source


def test_issue622_reuses_frozen_row_math_but_does_not_rerun_old_fixtures() -> None:
    source = (
        ROOT / "tam_research/aera_v26_9_issue622_corrected_autocast_dtype_gate.py"
    ).read_text()
    assert "frozen602._integrated_row(" in source
    assert "frozen602._make_integrated_case(" in source
    assert "_run_preserved_issue558_surface(" not in source
    assert "_advance_generator_through_issue558_regular_rows(" not in source
    assert "frozen553.run_ficem_read_mixed_dtype_probe_v26_7(" not in source
    assert "torch.Generator().manual_seed(DESIGN_SEED)" in source
    assert "DESIGN_SEED = 891_475_817" in source
    assert "load_issue602_preserved_authority(" in source
    assert "issue602_remains_authoritative_fail" in source


def test_issue622_v26_9_backend_still_uses_identity_weight_visibility() -> None:
    assert v26_9.read_dispatch_kind(
        torch.float32, torch.bfloat16, torch.float32, torch.float32
    ) == "mixed-identity-weight-visibility-v26.9"
    assert v26_9.initial_weight_visibility_kind(
        torch.float32, torch.bfloat16, torch.float32, torch.float32
    ) == "float32"
    tail_source = inspect.getsource(v26_9.fused_ficem_read_tail_v26_9)
    backend_source = inspect.getsource(
        v26_9.IdentityWeightVisibilityTritonFICEMReadWriteBackend.read
    )
    assert "WEIGHT_VISIBLE_BF16=identity_dtype is torch.bfloat16" in tail_source
    assert "identity_dtype=identity_source.dtype" in backend_source


def test_issue622_launcher_and_workflow_preserve_one_shot_boundary() -> None:
    launcher = (
        ROOT / "modal_aera_v26_9_issue622_corrected_autocast_dtype_gate_l4_app.py"
    ).read_text()
    workflow = (
        ROOT / ".github/workflows/aera-v26-9-issue622-corrected-autocast-dtype-gate-l4.yml"
    ).read_text()

    assert launcher.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in launcher
    assert (
        'RESULT_PATH = "/vol/aera-v26/issue622-corrected-autocast-dtype-gate/result.json"'
        in launcher
    )
    assert "refusing duplicate issue622 run because durable result exists" in launcher
    assert "volume.commit()" in launcher
    assert "ISSUE602_RESULT_SHA256" in launcher

    assert "workflow_dispatch" not in workflow
    assert "cancel-in-progress" not in workflow
    assert "GITHUB_RUN_ATTEMPT" in workflow
    assert "## #622 sole L4 corrected dtype gate authorization" in workflow
    assert "Authorize main:" in workflow
    assert (
        workflow.count(
            "modal run modal_aera_v26_9_issue622_corrected_autocast_dtype_gate_l4_app.py"
        )
        == 1
    )
    assert "[aera-v26-9-issue622-corrected-autocast-dtype-gate-l4]" in workflow
    assert "canonical" in workflow.lower()
    assert "modal deploy" not in workflow.lower()


def test_issue622_all_higher_authorizations_remain_false() -> None:
    protocol = probe.issue622_protocol()
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
