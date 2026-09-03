from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as v26_9
from tam_research import aera_v26_9_issue602_identity_weight_visibility_probe as probe

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MAIN = "37d2352050730c75dff0ab4b547e990b7865a95d"
SOURCE_TREE = "c3fd42879162cdc5e01b1ed0fcc34f2f82aa454f"
PROBE_BLOB = "456203f515d67d1c92b0a9c3e0e59ce4137ac10a"
LAUNCHER_BLOB = "55bddd82c51a7740096d5460e1f6567557a3af81"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
V26_9_CPU_TEST_BLOB = "305ec5732c46ceab2de9116898c54beb859e41e8"
V26_8_BACKEND_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
ISSUE558_PROBE_BLOB = "99ab8252f2b594404aae1ca86752eaa902eb80a5"
FROZEN_ISSUE553_PROBE_BLOB = "ff9a47f510be07e8adeff018f327338147163cdb"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
FACTORIZED_V25_BLOB = "f8cce87fa4dcae69fd171ba95fcbdab50e743a2f"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue602_frozen_lineage_and_new_files_exact() -> None:
    assert probe.SOURCE_MAIN == SOURCE_MAIN
    assert probe.SOURCE_TREE == SOURCE_TREE
    expected = {
        ROOT / "tam_research/aera_v26_9_issue602_identity_weight_visibility_probe.py": PROBE_BLOB,
        ROOT / "modal_aera_v26_9_issue602_identity_weight_visibility_l4_app.py": LAUNCHER_BLOB,
        ROOT / "tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py": V26_9_BACKEND_BLOB,
        ROOT / "tests/test_aera_v26_9_issue600_identity_weight_visibility_cpu.py": V26_9_CPU_TEST_BLOB,
        ROOT / "tam_research/aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py": V26_8_BACKEND_BLOB,
        ROOT / "tam_research/aera_v26_8_issue558_ficem_read_mixed_strength_precision_probe.py": ISSUE558_PROBE_BLOB,
        ROOT / "tam_research/aera_v26_7_issue553_ficem_read_mixed_dtype_probe.py": FROZEN_ISSUE553_PROBE_BLOB,
        ROOT / "tam_research/aera_v26_3_ficem_read_probe.py": HISTORICAL_PROBE_BLOB,
        ROOT / "tam_research/aera_hardware_core_v26_3_ficem_read_triton.py": REPAIR5_BACKEND_BLOB,
        ROOT / "tam_research/aera_hardware_core_v26_6_ficem_write_materialize_cast.py": V26_6_WRITE_BLOB,
        ROOT / "tam_research/aera_hardware_core_v26.py": V26_INTERFACE_BLOB,
        ROOT / "tam_research/aera_hardware_core_v25_1_compact.py": STABLE_REFERENCE_BLOB,
        ROOT / "tam_research/aera_hardware_core_v25.py": FACTORIZED_V25_BLOB,
    }
    assert {str(path.relative_to(ROOT)): _blob(path) for path in expected} == {
        str(path.relative_to(ROOT)): sha for path, sha in expected.items()
    }


def test_issue602_cpu_preflight_is_non_gpu_and_preserves_authorities() -> None:
    contract = probe.cpu_contract_preflight_issue602()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["synthetic_only"] is True
    assert contract["model_constructed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["corpus_accessed"] is False
    assert contract["scientific_seed_consumed"] is False

    protocol = probe.issue602_protocol()
    assert protocol["issue558_surface_preserved_wholesale"] is True
    assert protocol["issue558_thresholds_relaxed"] is False
    assert protocol["issue558_trigger"] == 561
    assert protocol["issue558_decision"] == "PASS"
    assert protocol["issue594_run"] == 33772104621
    assert protocol["issue597_run"] == 33774062361
    assert protocol["integrated_rows"] == 4
    assert protocol["integrated_atol"] == 1e-2
    assert protocol["integrated_rtol"] == 1e-2
    assert protocol["integrated_timing_decision_bearing"] is False


def test_issue602_v26_9_identity_visibility_contract_is_exact() -> None:
    assert v26_9.read_dispatch_kind(
        torch.float32, torch.bfloat16, torch.float32, torch.float32
    ) == "mixed-identity-weight-visibility-v26.9"
    assert v26_9.initial_weight_visibility_kind(
        torch.float32, torch.bfloat16, torch.float32, torch.float32
    ) == "float32"
    assert v26_9.initial_weight_visibility_kind(
        torch.bfloat16, torch.bfloat16, torch.float32, torch.float32
    ) == "bfloat16"
    assert v26_9.read_dispatch_kind(
        torch.float32, torch.float32, torch.float32, torch.float32
    ) == "historical-repair5"

    tail_source = inspect.getsource(v26_9.fused_ficem_read_tail_v26_9)
    backend_source = inspect.getsource(
        v26_9.IdentityWeightVisibilityTritonFICEMReadWriteBackend.read
    )
    assert "identity_dtype: torch.dtype" in tail_source
    assert "WEIGHT_VISIBLE_BF16=identity_dtype is torch.bfloat16" in tail_source
    assert "identity_dtype=identity_source.dtype" in backend_source
    assert "memory.out(recalled)" in backend_source


def test_issue602_probe_preserves_558_and_adds_only_frozen_split() -> None:
    source = (ROOT / "tam_research/aera_v26_9_issue602_identity_weight_visibility_probe.py").read_text()
    assert "frozen553.run_ficem_read_mixed_dtype_probe_v26_7()" in source
    assert "identity_dtype=similarity.dtype" in source
    assert "thresholds_relaxed" in source
    assert 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)' in source
    assert "identity_dtype=torch.float32" in source
    assert 'return frozen.ReadCase(' in source
    assert '"bfloat16", batch_size, validity_kind, identity, context, state' in source
    assert "for batch_size in BATCH_SIZES:" in source
    assert "for validity_kind in VALIDITY_KINDS:" in source
    assert "integrated_rows" in source
    assert "torch.softmax(safe_logits.float(), dim=-1).to(identity_dtype)" in source
    assert "selection_semantically_equivalent" in source
    assert "direct_tail_topology_pass" in source
    assert "full_backend_no_reference_tail_ops" in source


def test_issue602_launcher_and_workflow_keep_one_shot_boundary() -> None:
    launcher = (ROOT / "modal_aera_v26_9_issue602_identity_weight_visibility_l4_app.py").read_text()
    workflow = (ROOT / ".github/workflows/aera-v26-9-issue602-identity-weight-visibility-l4.yml").read_text()

    assert launcher.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in launcher
    assert 'RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"' in launcher
    assert "refusing duplicate issue602 run because durable result exists" in launcher
    assert "volume.commit()" in launcher

    assert "workflow_dispatch" not in workflow
    assert "cancel-in-progress" not in workflow
    assert "GITHUB_RUN_ATTEMPT" in workflow
    assert "## #602 sole L4 READ gate authorization" in workflow
    assert "Authorize main:" in workflow
    assert workflow.count("modal run modal_aera_v26_9_issue602_identity_weight_visibility_l4_app.py") == 1
    assert "[aera-v26-9-issue602-identity-weight-visibility-l4]" in workflow
    assert "canonical" in workflow.lower()
    assert "rerun" not in workflow.lower()
    assert "redispatch" not in workflow.lower()
    assert "modal deploy" not in workflow.lower()


def test_issue602_all_higher_authorizations_remain_false() -> None:
    protocol = probe.issue602_protocol()
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
