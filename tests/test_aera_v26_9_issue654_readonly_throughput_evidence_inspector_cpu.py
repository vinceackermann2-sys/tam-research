from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "modal_aera_v26_9_issue654_readonly_throughput_evidence_inspector.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-9-issue654-readonly-throughput-evidence-inspector.yml"
SCIENTIFIC_ADAPTER = ROOT / "tam_research" / "aera_v26_9_issue643_bounded_memory_end_to_end_systems.py"

SOURCE_MAIN = "1eebb678609523aa67401e482dcc63e0ea858aa2"
SOURCE_TREE = "0e81102d35a58bae1920ed861f2729d711be7132"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue650-e2e-cli-guard-continuation/result.json"
SOURCE_RESULT_SHA256 = "914615db5267565563dcc9e82bfc31f444a656a68bd560f50447a8fd03588431"
SCIENTIFIC_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
INSPECTOR_BLOB = "9d60d9dddf42982cdc7d03681abc329c84eb4511"
WORKFLOW_BLOB = "5546d4fcfc0d704b65d0ea77b946ed0f2b01ae80"

FROZEN_ROWS = {
    "8": {
        "routing_exact": True,
        "logit_equivalence_pass": True,
        "logit_max_abs": 0.03125,
        "state_equivalence_pass": True,
        "physical_sparse_pass": True,
        "write_geometry_pass": True,
        "finite": True,
        "persistent_state_bytes_pass": True,
        "candidate_full_vs_transformer_speed_ratio": 0.20816584116044815,
        "required_full_speed_ratio": 0.25,
        "throughput_pass": False,
        "reference_full_latency_ms": 67.08428955078125,
        "candidate_full_latency_ms": 54.38617515563965,
        "candidate_vs_reference_latency_ratio": 0.8107140363239679,
        "no_reference_full_latency_regression": True,
    },
    "64": {
        "routing_exact": True,
        "logit_equivalence_pass": True,
        "logit_max_abs": 0.0625,
        "state_equivalence_pass": True,
        "physical_sparse_pass": True,
        "write_geometry_pass": True,
        "finite": True,
        "persistent_state_bytes_pass": True,
        "candidate_full_vs_transformer_speed_ratio": 0.9196618814832817,
        "required_full_speed_ratio": 1.25,
        "throughput_pass": False,
        "reference_full_latency_ms": 115.4748306274414,
        "candidate_full_latency_ms": 99.48311996459961,
        "candidate_vs_reference_latency_ratio": 0.8615134521007772,
        "no_reference_full_latency_regression": True,
    },
}


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _literal_constant(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant {name}")


def test_issue654_freezes_exact_consumed_source_authority() -> None:
    assert _literal_constant(INSPECTOR, "RESEARCH_ISSUE") == 654
    assert _literal_constant(INSPECTOR, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal_constant(INSPECTOR, "SOURCE_TREE") == SOURCE_TREE
    assert _literal_constant(INSPECTOR, "SOURCE_RESULT_PATH") == SOURCE_RESULT_PATH
    assert _literal_constant(INSPECTOR, "SOURCE_RESULT_SHA256") == SOURCE_RESULT_SHA256
    assert _literal_constant(INSPECTOR, "SOURCE_TRIGGER") == 653
    assert _literal_constant(INSPECTOR, "SOURCE_RUN") == 34022331841
    assert _literal_constant(INSPECTOR, "SOURCE_JOB") == 101457058965
    assert _literal_constant(INSPECTOR, "SOURCE_ATTEMPT") == 1
    assert _literal_constant(INSPECTOR, "SOURCE_DECISION") == "FAIL_FROZEN_E2E_SYSTEMS_GATE"
    assert _literal_constant(INSPECTOR, "SOURCE_OVERALL_PASS") is False
    assert _literal_constant(INSPECTOR, "SOURCE_DEVICE") == "NVIDIA L4"
    assert _literal_constant(INSPECTOR, "FROZEN_ROWS") == FROZEN_ROWS


def test_issue654_exact_artifact_and_scientific_blobs() -> None:
    assert _blob(SCIENTIFIC_ADAPTER) == SCIENTIFIC_ADAPTER_BLOB
    assert _blob(INSPECTOR) == INSPECTOR_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB


def test_issue654_inspector_is_strict_read_only_cpu_evidence_extraction() -> None:
    source = INSPECTOR.read_text()
    assert 'volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)' in source
    assert "volume.reload()" in source
    assert "path.read_bytes()" in source
    assert "hashlib.sha256(raw).hexdigest()" in source
    assert "digest != SOURCE_RESULT_SHA256" in source
    assert "json.loads(raw)" in source
    assert '"volume_mutated": False' in source
    assert '"gpu_used": False' in source
    assert '"model_constructed": False' in source
    assert '"checkpoint_read": False' in source
    assert '"new_benchmark_performed": False' in source
    assert '"threshold_changed": False' in source
    assert "gpu=" not in source
    assert "volume.commit" not in source
    assert "import torch" not in source
    assert "from torch" not in source
    for forbidden in (
        "write_text(",
        "write_bytes(",
        ".unlink(",
        ".rename(",
        ".replace(",
        "torch.load",
        "load_models(",
        "run_end_to_end_systems",
        "optimizer.step",
        ".backward(",
        "open(",
    ):
        assert forbidden not in source


def test_issue654_extracts_only_persisted_throughput_diagnostics() -> None:
    source = INSPECTOR.read_text()
    for required in (
        '"timings": timings',
        '"top_cuda_operators": top_ops',
        '"fragmentation_operator_calls": fragmentation',
        '"peak_vram": peak_vram',
        '"physical_sparse_counters": physical_sparse',
        '"routing_accounting": {',
        '"reference": routing_reference',
        '"candidate": routing_candidate',
        '"candidate_full_vs_transformer_speed_ratio"',
        '"required_full_speed_ratio"',
        '"throughput_pass"',
        '"candidate_full_latency_ms"',
        '"reference_full_latency_ms"',
        '"candidate_vs_reference_latency_ratio"',
    ):
        assert required in source
    assert _literal_constant(INSPECTOR, "EXPECTED_TIMING_CONDITIONS") == (
        "transformer",
        "v26_torch_reference_full_ficem",
        "v26_4_triton_full_ficem",
    )


def test_issue654_reverifies_all_correctness_gates_before_extraction() -> None:
    source = INSPECTOR.read_text()
    for key in (
        "routing_exact",
        "logit_equivalence_pass",
        "logit_max_abs",
        "state_equivalence_pass",
        "physical_sparse_pass",
        "write_geometry_pass",
        "finite",
        "persistent_state_bytes_pass",
        "candidate_full_vs_transformer_speed_ratio",
        "required_full_speed_ratio",
        "throughput_pass",
        "reference_full_latency_ms",
        "candidate_full_latency_ms",
        "candidate_vs_reference_latency_ratio",
        "no_reference_full_latency_regression",
    ):
        assert f'"{key}"' in source
    assert "observed != FROZEN_ROWS[batch]" in source


def test_issue654_workflow_is_owner_opened_one_shot_and_exact_bound_main() -> None:
    source = WORKFLOW.read_text()
    lowered = source.lower()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "cancel-in-progress: false" in source
    assert "[aera-v26-9-issue654-readonly-throughput-evidence-inspector]" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${#matching_triggers[@]}" = "1"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${matching_triggers[0]}"' in source
    assert "bound_lines=" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor " + SOURCE_MAIN in source
    assert "34022331841" in source
    assert "101457058965" in source
    assert SOURCE_RESULT_SHA256 in source
    assert "## #654 sole read-only throughput inspector authorization" in source
    assert source.count("modal run modal_aera_v26_9_issue654_readonly_throughput_evidence_inspector.py") == 1
    assert 'gpu="' not in lowered
    assert "gpu: " not in lowered
    assert "modal deploy" not in lowered
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "redispatch" not in lowered


def test_issue654_workflow_verifies_preregistered_dynamic_blobs() -> None:
    source = WORKFLOW.read_text()
    for required in (
        "INSPECTOR_BLOB=",
        "WORKFLOW_BLOB=",
        "CPU_TEST_BLOB=",
        "git hash-object modal_aera_v26_9_issue654_readonly_throughput_evidence_inspector.py",
        "git hash-object .github/workflows/aera-v26-9-issue654-readonly-throughput-evidence-inspector.yml",
        "git hash-object tests/test_aera_v26_9_issue654_readonly_throughput_evidence_inspector_cpu.py",
        SCIENTIFIC_ADAPTER_BLOB,
    ):
        assert required in source


def test_issue654_higher_authorizations_and_credit_remain_false() -> None:
    source = INSPECTOR.read_text()
    for key in (
        "optimization_authorized",
        "systems_pass_earned",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{key}": False' in source
