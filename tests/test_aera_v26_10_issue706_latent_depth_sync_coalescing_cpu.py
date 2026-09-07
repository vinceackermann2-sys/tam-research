from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from tam_research.aera_hardware_core_v5 import DtypeSafeChunkLatentReasoner
from tam_research.aera_hardware_core_v25_1_nohost import (
    ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner,
)
from tam_research.aera_hardware_core_v26_10_latent_depth_sync_coalescing import (
    CoalescedDepthPrefixLatentReasonerV2610,
    install_latent_depth_sync_coalescing_v26_10,
    latent_depth_sync_coalescing_v26_10_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "tam_research/aera_hardware_core_v26_10_latent_depth_sync_coalescing.py"
LAUNCHER = ROOT / "modal_aera_v26_10_issue706_latent_depth_sync_coalescing.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-10-issue706-latent-depth-sync-coalescing.yml"
IMPL_BLOB = "d8f691c198eed1fa96bcbb78a4e76cad82d18779"
LAUNCHER_BLOB = "e23072aa83b0bc20ccd72e7567f234e9f4a931f6"
WORKFLOW_BLOB = "c842f636797d7a5c424d2dc8718f023efc2efe90"

FROZEN_FILES = {
    "tam_research/aera_hardware_core_v25_1_nohost.py": "237e5615cf32f644e8675808a6b3e9adaf04fb23",
    "tam_research/aera_hardware_core_v26.py": "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    "tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py": "512572340cc09e2e7ad6729712258c12cb377ef2",
    "tam_research/aera_v26_5_end_to_end_systems.py": "c9731cae7e386f09b2a190b045532591c4fa00be",
    "tam_research/aera_v25_post8471_triage.py": "e5ce8cdda0777dce97816e2640f4492803a6b191",
    "tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py": "b81cc209f5d95abbe1fb8bd620c78e87c067bc19",
    "modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py": "72f27391ff2f0a7bff8d4532f307ddc4869cf494",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _reasoner_pair(d_model: int = 8, max_steps: int = 4):
    torch.manual_seed(706)
    source_a = DtypeSafeChunkLatentReasoner(d_model, max_steps)
    source_b = DtypeSafeChunkLatentReasoner(d_model, max_steps)
    source_b.load_state_dict(source_a.state_dict(), strict=True)
    baseline = ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner(source_a)
    candidate_source = ExecutionEquivalentNoHostDtypeSafeChunkLatentReasoner(source_b)
    candidate = CoalescedDepthPrefixLatentReasonerV2610(candidate_source)
    return baseline.eval(), candidate.eval()


def test_issue706_exact_detached_files_and_frozen_dependencies():
    assert IMPL.exists() and LAUNCHER.exists() and WORKFLOW.exists()
    assert _git_blob(IMPL) == IMPL_BLOB
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    ast.parse(IMPL.read_text())
    ast.parse(LAUNCHER.read_text())
    for rel, expected in FROZEN_FILES.items():
        path = ROOT / rel
        assert path.exists(), rel
        assert _git_blob(path) == expected, rel


def test_issue706_protocol_is_execution_only_and_authority_bounded():
    protocol = latent_depth_sync_coalescing_v26_10_protocol()
    assert protocol["research_issue"] == 706
    assert protocol["source_main"] == "4883abe053347379812901c5c2057da70c11e3e7"
    assert protocol["source_tree"] == "27be25ba578aa530c1e6d832db1a38be4c8bc22a"
    assert protocol["target"] == "launch_or_idle_reduction"
    assert protocol["max_reason_steps"] == 4
    assert protocol["host_count_materializations_per_reasoner_invocation_target"] == 1
    assert protocol["legacy_per_depth_nonzero_compactions_target"] == 0
    for key in (
        "hard_depth_choice_changed",
        "gru_cell_equation_changed",
        "gru_updates_per_example_changed",
        "inactive_rows_execute_gru",
        "dense_masked_sparse_credit",
        "optional_stage_routing_changed",
        "expert_routing_changed",
        "ficem_changed",
        "memory_write_changed",
        "durable_state_changed",
        "learned_parameter_count_changed",
        "parameter_schema_changed",
        "state_dict_schema_changed",
        "checkpoint_bytes_changed",
        "soft_training_path_changed",
        "gpu_authorized",
        "scientific_training_authorized",
        "systems_pass_earned",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False, key


def test_issue706_hard_reasoner_preserves_choices_sparse_updates_and_output():
    baseline, candidate = _reasoner_pair()
    assert tuple(baseline.state_dict()) == tuple(candidate.state_dict())
    for (left_name, left), (right_name, right) in zip(
        baseline.state_dict().items(), candidate.state_dict().items()
    ):
        assert left_name == right_name
        assert torch.equal(left, right)

    summary = torch.randn(6, 8)
    chosen = torch.tensor([4, 2, 3, 1, 4, 2])
    depth_logits = torch.full((6, 4), -7.0)
    depth_logits[torch.arange(6), chosen - 1] = 7.0

    with torch.inference_mode():
        baseline_out = baseline(summary.clone(), depth_logits.clone(), hard=True)
        candidate_out = candidate(summary.clone(), depth_logits.clone(), hard=True)

    assert torch.equal(baseline.last_steps, chosen)
    assert torch.equal(candidate.last_steps, chosen)
    assert candidate.last_active_prefix_counts == (6, 5, 3, 2)
    assert candidate.last_dense_masked_execution is False
    assert torch.allclose(baseline.last_expected, candidate.last_expected, atol=0, rtol=0)
    assert torch.allclose(baseline_out, candidate_out, atol=1e-2, rtol=1e-2)


def test_issue706_soft_path_is_identical_to_frozen_reasoner():
    baseline, candidate = _reasoner_pair()
    summary = torch.randn(5, 8)
    depth_logits = torch.randn(5, 4)
    with torch.inference_mode():
        baseline_out = baseline(summary.clone(), depth_logits.clone(), hard=False)
        candidate_out = candidate(summary.clone(), depth_logits.clone(), hard=False)
    assert candidate.last_active_prefix_counts is None
    assert candidate.last_dense_masked_execution is False
    assert candidate.last_steps is None
    assert torch.allclose(baseline_out, candidate_out, atol=1e-6, rtol=1e-6)


def test_issue706_installer_reuses_exact_cells_and_state_dict_paths():
    class Stage(nn.Module):
        def __init__(self, reasoner: nn.Module):
            super().__init__()
            self.reasoner = reasoner

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = nn.ModuleList(
                [Stage(_reasoner_pair()[0]), Stage(_reasoner_pair()[0])]
            )

    model = Model()
    keys_before = tuple(model.state_dict())
    values_before = {name: value.clone() for name, value in model.state_dict().items()}
    cells_before = tuple(stage.reasoner.cell for stage in model.stages)
    installed = install_latent_depth_sync_coalescing_v26_10(model)
    assert installed == (0, 1)
    assert tuple(model.state_dict()) == keys_before
    for name, value in model.state_dict().items():
        assert torch.equal(value, values_before[name])
    for stage, old_cell in zip(model.stages, cells_before):
        assert isinstance(stage.reasoner, CoalescedDepthPrefixLatentReasonerV2610)
        assert stage.reasoner.cell is old_cell


def test_issue706_impl_has_one_count_materialization_and_no_per_depth_nonzero():
    text = IMPL.read_text()
    assert "chosen = depth_logits.argmax(dim=-1) + 1" in text
    assert "torch.argsort(chosen, dim=0, descending=True, stable=True)" in text
    assert "counts_device = torch.stack(" in text
    assert 'counts_device.to(device="cpu").tolist()' in text
    assert ".nonzero(" not in text
    assert "selected = ordered[:count]" in text
    assert "self.cell(selected, selected)" in text
    assert "current = summary.index_copy(0, order, ordered)" in text


def test_issue706_launcher_freezes_correctness_latency_and_sync_gates():
    text = LAUNCHER.read_text()
    for required in (
        'RESULT_PATH = "/vol/aera-v26/issue706-v26-10-latent-depth-sync-coalescing/result.json"',
        'PARENT_RESULT_PATH = "/vol/aera-v26/issue702-cuda-timeline-dispatch-gap-attribution/result.json"',
        "PARENT_TRIGGER = 705",
        "PARENT_EVIDENCE_COMMENT = 5572472746",
        "PARENT_RUN = 34135719600",
        "PARENT_JOB = 101786085046",
        "PARENT_ATTEMPT = 1",
        "BATCHES = (8, 64)",
        "TOKEN_SEED_BASE = 138471 + 10000",
        "WARMUP_CALLS = 3",
        "TIMED_CALLS_PER_CONDITION = 20",
        "PROFILE_CALLS_PER_CONDITION = 1",
        "MIN_LATENCY_IMPROVEMENT = 0.05",
        "MIN_STREAM_SYNCHRONIZE_REDUCTION = 20",
        "INTEGRATED_ATOL = 1e-2",
        "INTEGRATED_RTOL = 1e-2",
        '"route_mode": "hard_sparse"',
        '"hard": True',
        '"update_memory": True',
        "candidate_prefix_counts_exact",
        "dense_masked_execution",
        "cudaStreamSynchronize_reduction",
        "fresh_full_e2e_systems_gate_authorized",
        '"systems_pass_earned": False',
    ):
        assert required in text
    for forbidden in (
        ".backward(",
        "torch.optim.",
        ".step(",
        "optimizer=",
        "run_end_to_end_systems_v26_9(",
        "run_end_to_end_systems_v26_8(",
    ):
        assert forbidden not in text


def test_issue706_launcher_packages_root_dependency_and_separates_preflight_l4():
    text = LAUNCHER.read_text()
    assert "import modal_aera_v26_9_issue665_frozen_throughput_component_attribution as issue665" in text
    assert 'image = issue665.image.add_local_file(' in text
    assert 'ISSUE665_LAUNCHER, f"/root/{ISSUE665_LAUNCHER}"' in text
    assert "def preflight()" in text
    assert "def run_microbenchmark()" in text
    assert "preauth_main" in text and "l4_main" in text


def test_issue706_workflow_is_owner_only_single_attempt_and_separately_gated():
    text = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert "workflow_dispatch:" not in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert "[aera-v26-10-issue706-latent-depth-sync-coalescing-preauth]" in text
    assert "[aera-v26-10-issue706-latent-depth-sync-coalescing-l4]" in text
    assert "## #706 sole L4 latent-depth sync coalescing authorization" in text
    assert "preauth_main" in text and "l4_main" in text
    assert "V26_10_IMPL_BLOB=d8f691c198eed1fa96bcbb78a4e76cad82d18779" in text
    assert "MIN_LATENCY_IMPROVEMENT=0.05" in text
    assert "MIN_STREAM_SYNCHRONIZE_REDUCTION=20" in text
