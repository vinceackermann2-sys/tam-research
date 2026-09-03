from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "modal_aera_v26_8_issue574_issue571_result_inspector.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue574-issue571-result-inspector.yml"
ISSUE571_LAUNCHER = ROOT / "modal_aera_v26_8_issue571_memory_safe_end_to_end_systems_l4_app.py"
ISSUE571_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue571-memory-safe-e2e-systems-l4.yml"
ISSUE571_TEST = ROOT / "tests" / "test_aera_v26_8_issue571_memory_safe_end_to_end_systems_l4_cpu.py"

SOURCE_MAIN = "c913d817ee1c1a1fca2d7c7622f4c8ca5353772f"
SOURCE_TREE = "646ffe7f96ed2f1322408fcd2dc3eee2ff886161"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue571-memory-safe-end-to-end-systems/result.json"
SOURCE_RESULT_SHA256 = "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
INSPECTOR_BLOB = "e5ff20ff60b0b00e70c1e8d7b1ca8516459d2749"
WORKFLOW_BLOB = "e29d851b7c8baae07dcbc17f0779b7b391c4c2b7"

FROZEN_BLOBS = {
    ROOT / "tam_research" / "aera_v26_5_end_to_end_systems.py": "c9731cae7e386f09b2a190b045532591c4fa00be",
    ROOT / "tam_research" / "aera_v26_5_end_to_end_systems_repair1.py": "b3f7082b188644007b873db3733492f424d4941a",
    ROOT / "tam_research" / "aera_v26_6_issue530_end_to_end_systems.py": "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b",
    ROOT / "tam_research" / "aera_v26_8_issue562_end_to_end_systems.py": "3534103eea21f7c4d9d31798ad34601fd47090d6",
    ROOT / "tam_research" / "aera_v26_8_issue569_end_to_end_systems_memory_safe.py": "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203",
    ROOT / "tam_research" / "aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py": "3575c58d1cd730be77649f087908c51dbf3e6088",
    ROOT / "tam_research" / "aera_hardware_core_v26_7_ficem_read_mixed_dtype.py": "d8133c6b204b1ee5f23955255fb2fb09d09bd723",
    ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py": "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
    ROOT / "tam_research" / "aera_hardware_core_v26_6_ficem_write_materialize_cast.py": "d45c262314a0b4691f26812a279937a225043ad9",
    ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py": "e54570292489bd17570038dca7518419ac00418c",
    ROOT / "tam_research" / "aera_hardware_core_v26.py": "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py": "4e336b6e1a6238dac782fa320751d68281493ee1",
}

FROZEN_BATCH_PARTITION = {
    "8": {
        "routing_exact": True,
        "logit_equivalence_pass": True,
        "logit_max_abs": 0.0625,
        "state_equivalence_pass": False,
        "physical_sparse_pass": False,
        "write_geometry_pass": True,
        "finite": True,
        "persistent_state_bytes_pass": True,
        "candidate_full_vs_transformer_speed_ratio": 0.2665054349604248,
        "required_full_speed_ratio": 0.25,
        "throughput_pass": True,
        "reference_full_latency_ms": 43.729408264160156,
        "candidate_full_latency_ms": 35.385873794555664,
        "no_reference_full_latency_regression": True,
    },
    "64": {
        "routing_exact": False,
        "logit_equivalence_pass": False,
        "logit_max_abs": 2.34765625,
        "state_equivalence_pass": False,
        "physical_sparse_pass": False,
        "write_geometry_pass": True,
        "finite": True,
        "persistent_state_bytes_pass": True,
        "candidate_full_vs_transformer_speed_ratio": 1.164232659436335,
        "required_full_speed_ratio": 1.25,
        "throughput_pass": False,
        "reference_full_latency_ms": 90.45303726196289,
        "candidate_full_latency_ms": 78.88516616821289,
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


def test_issue574_freezes_exact_source_and_consumed_issue571_authority() -> None:
    assert _literal_constant(INSPECTOR, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal_constant(INSPECTOR, "SOURCE_TREE") == SOURCE_TREE
    assert _literal_constant(INSPECTOR, "SOURCE_RESULT_PATH") == SOURCE_RESULT_PATH
    assert _literal_constant(INSPECTOR, "SOURCE_RESULT_SHA256") == SOURCE_RESULT_SHA256
    assert _literal_constant(INSPECTOR, "SOURCE_TRIGGER") == 573
    assert _literal_constant(INSPECTOR, "SOURCE_RUN") == 33741700781
    assert _literal_constant(INSPECTOR, "SOURCE_JOB") == 100604889696
    assert _literal_constant(INSPECTOR, "SOURCE_ATTEMPT") == 1
    assert _literal_constant(INSPECTOR, "SOURCE_DECISION") == "FAIL_FROZEN_E2E_SYSTEMS_GATE"
    assert _literal_constant(INSPECTOR, "SOURCE_OVERALL_PASS") is False
    assert _literal_constant(INSPECTOR, "FROZEN_BATCH_PARTITION") == FROZEN_BATCH_PARTITION


def test_issue574_preserves_all_frozen_lineage_blobs_and_issue571_harness() -> None:
    for path, expected in FROZEN_BLOBS.items():
        assert _blob(path) == expected, path
    assert _blob(ISSUE571_LAUNCHER) == "d0e88d2ebab5a2df340325b2951ed1517c6945b7"
    assert _blob(ISSUE571_WORKFLOW) == "7077f54353bd5b57dd33d47e55b5d65edc664290"
    assert _blob(ISSUE571_TEST) == "7955ae4387f598b945bbf2a0c46d5c9634178bba"
    assert _blob(INSPECTOR) == INSPECTOR_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB


def test_issue574_inspector_is_strictly_read_only_cpu_and_sha_guarded() -> None:
    source = INSPECTOR.read_text()
    assert 'volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)' in source
    assert "volume.reload()" in source
    assert "path.read_bytes()" in source
    assert "hashlib.sha256(raw).hexdigest()" in source
    assert "digest != SOURCE_RESULT_SHA256" in source
    assert "json.loads(raw)" in source
    assert '"volume_mutated": False' in source
    assert '"gpu_used": False' in source
    assert '"experiment_rerun": False' in source
    assert "gpu=" not in source
    assert "volume.commit" not in source
    for forbidden in (
        "write_text(",
        "write_bytes(",
        ".unlink(",
        ".rename(",
        ".replace(",
        "torch.",
        "torch.load",
        "load_models(",
        "run_end_to_end_systems",
        "optimizer.step",
        ".backward(",
    ):
        assert forbidden not in source


def test_issue574_retains_complete_stored_diagnostics_without_inventing_routes() -> None:
    source = INSPECTOR.read_text()
    for required in (
        '"reference_accounting": routing_reference',
        '"candidate_accounting": routing_candidate',
        '"accounting_diff": _json_diff(routing_reference, routing_candidate)',
        '"raw_gate_signatures_persisted": False',
        '"logit_equivalence": logit',
        '"state_equivalence": state',
        '"physical_sparse": sparse',
        '"write_geometry": write',
        '"timings": timings',
        '"peak_vram_diagnostic_only": peak_vram',
        '"profiler_candidate_full_diagnostic_only": profiler',
        '"failed_gates": failed_gates',
        '"parameter_versions_before": result.get("parameter_versions_before")',
        '"checkpoint_hashes_before": result.get("checkpoint_hashes_before")',
        '"issue571_gate_metadata": metadata',
    ):
        assert required in source
    base = (ROOT / "tam_research" / "aera_v26_5_end_to_end_systems.py").read_text()
    for stored_state_field in (
        '"max_stream_abs"',
        '"max_keys_abs"',
        '"max_values_abs"',
        '"max_strengths_abs"',
        '"max_continuous_abs"',
        '"continuous_allclose"',
        '"validity_exact"',
        '"dtype_device_shape_exact"',
    ):
        assert stored_state_field in base


def test_issue574_classifies_only_observed_gate_categories_and_backend_contract() -> None:
    source = INSPECTOR.read_text()
    for category in (
        "evaluator_contract_incompatibility",
        "routing_mismatch",
        "continuous_state_numerical_mismatch",
        "validity_or_state_schema_mismatch",
        "output_logit_numerical_mismatch",
        "output_logit_metadata_mismatch",
        "performance_threshold_miss",
    ):
        assert f'"{category}"' in source
    assert _literal_constant(INSPECTOR, "FROZEN_501_BACKEND_NAME") == "triton-ficem-read-repair5-write-v26.4"
    assert _literal_constant(INSPECTOR, "SOURCE_CANDIDATE_BACKEND") == "triton-ficem-read-v26.8-mixed-strength-precision-write-v26.6-materialize-cast"
    assert '"identity_incompatibility_proven": frozen_backend_identity_incompatible' in source
    assert '"retroactive_pass_granted": False' in source


def test_issue574_workflow_is_one_shot_cpu_only_exact_bound_main() -> None:
    source = WORKFLOW.read_text()
    lowered = source.lower()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "cancel-in-progress: false" in source
    assert "[aera-v26-8-issue574-issue571-result-inspector]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert "bind_lines=" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert "33741700781" in source
    assert "100604889696" in source
    assert SOURCE_RESULT_SHA256 in source
    assert INSPECTOR_BLOB in source
    assert source.count("modal run modal_aera_v26_8_issue574_issue571_result_inspector.py") == 1
    assert 'gpu="' not in lowered
    assert "gpu: " not in lowered
    assert "modal deploy" not in lowered
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "redispatch" not in lowered
    assert "grep -c 'volume.commit' modal_aera_v26_8_issue574_issue571_result_inspector.py" in source


def test_issue574_higher_authorizations_remain_false() -> None:
    source = INSPECTOR.read_text()
    for key in (
        "repair_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{key}": False' in source
