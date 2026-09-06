from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-9-issue665-frozen-throughput-component-attribution.yml"
SCIENTIFIC_ADAPTER = ROOT / "tam_research" / "aera_v26_9_issue643_bounded_memory_end_to_end_systems.py"
RUNTIME_INTERFACE = ROOT / "tam_research" / "aera_hardware_core_v26.py"
BASE_SYSTEMS = ROOT / "tam_research" / "aera_v26_5_end_to_end_systems.py"
V26_9_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py"

SOURCE_MAIN = "6a87dd8c3d3f9e73d34aa7a3c1e2ed991b53a002"
SOURCE_TREE = "3daf8e0fce277a65f9ae3daa56d8252dd15ff664"
SCIENTIFIC_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
RUNTIME_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
LAUNCHER_BLOB = "72f27391ff2f0a7bff8d4532f307ddc4869cf494"
WORKFLOW_BLOB = "d2642917ed1e9009edfe1b255131d0e55723ea6a"
SOURCE_RESULT_SHA256 = "914615db5267565563dcc9e82bfc31f444a656a68bd560f50447a8fd03588431"


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


def test_issue665_freezes_exact_source_and_immutable_scientific_bytes() -> None:
    assert _literal_constant(LAUNCHER, "RESEARCH_ISSUE") == 665
    assert _literal_constant(LAUNCHER, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal_constant(LAUNCHER, "SOURCE_TREE") == SOURCE_TREE
    assert _literal_constant(LAUNCHER, "SOURCE_TRIGGER") == 653
    assert _literal_constant(LAUNCHER, "SOURCE_RUN") == 34022331841
    assert _literal_constant(LAUNCHER, "SOURCE_JOB") == 101457058965
    assert _literal_constant(LAUNCHER, "SOURCE_ATTEMPT") == 1
    assert _literal_constant(LAUNCHER, "READONLY_TRIGGER") == 664
    assert _literal_constant(LAUNCHER, "READONLY_RUN") == 34039744132
    assert _literal_constant(LAUNCHER, "READONLY_JOB") == 101504212730
    assert _literal_constant(LAUNCHER, "READONLY_ATTEMPT") == 1
    assert _literal_constant(LAUNCHER, "READONLY_EVIDENCE_COMMENT") == 5559943198
    assert _literal_constant(LAUNCHER, "SOURCE_RESULT_SHA256") == SOURCE_RESULT_SHA256
    assert _literal_constant(LAUNCHER, "SOURCE_DECISION") == "FAIL_FROZEN_E2E_SYSTEMS_GATE"

    assert _blob(SCIENTIFIC_ADAPTER) == SCIENTIFIC_ADAPTER_BLOB
    assert _blob(RUNTIME_INTERFACE) == RUNTIME_INTERFACE_BLOB
    assert _blob(BASE_SYSTEMS) == BASE_SYSTEMS_BLOB
    assert _blob(V26_9_BACKEND) == V26_9_BACKEND_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB


def test_issue665_uses_exact_frozen_candidate_workload_but_not_gate_timing_protocol() -> None:
    source = LAUNCHER.read_text()
    assert _literal_constant(LAUNCHER, "SYSTEM_BATCH_SIZES") == (8, 64)
    assert _literal_constant(LAUNCHER, "TOKEN_SEED_BASE") == 138471
    assert _literal_constant(LAUNCHER, "TOKEN_SEED_OFFSET") == 10000
    assert _literal_constant(LAUNCHER, "DIAGNOSTIC_WARMUP_CALLS") == 2
    assert _literal_constant(LAUNCHER, "DIAGNOSTIC_MEASURED_CALLS") == 12
    assert 'TOKEN_SEED_BASE + TOKEN_SEED_OFFSET + batch_size' in source
    assert '(batch_size, triage.SEQ_LEN)' in source
    assert 'base._model_call(candidate, tokens, update_memory=True)' in source
    assert 'load_models_v26_9(' in source
    assert 'route_mode": "hard_sparse"' in source
    assert '"hard": True' in source
    assert '"update_memory": True' in source
    assert "_timed_summaries" not in source
    assert "SYSTEM_TIMED_CALLS_PER_ROUND" not in source
    assert "SYSTEM_ROUNDS" not in source
    assert '"frozen_gate_timing_protocol_reused": False' in source
    assert '"comparative_gate_rerun": False' in source
    assert "run_end_to_end_systems_v26_9_bounded_memory(" not in source


def test_issue665_constructs_reference_and_transformer_only_for_loader_identity() -> None:
    source = LAUNCHER.read_text()
    assert "reference, candidate, transformer, candidate_backend_names = systems.load_models_v26_9(" in source
    assert "del reference" in source
    assert "del transformer" in source
    assert "base._model_call(reference" not in source
    assert "base._model_call(transformer" not in source
    assert '"reference_model_executed": False' in source
    assert '"transformer_model_executed": False' in source


def test_issue665_same_stream_event_contract_and_existing_boundaries_only() -> None:
    source = LAUNCHER.read_text()
    assert "torch.cuda.Event(enable_timing=True)" in source
    assert source.count("torch.cuda.synchronize()") == 1
    for required in (
        'candidate._route_one_stage',
        'router.forward',
        'stage.forward_chunk',
        'runtime.pack_ephemeral_epi_state',
        'runtime.select_packed_epi_state',
        'runtime.merge_packed_epi_state',
        '("read", "update", "update_from_projected")',
        'f"route.{stage_name}"',
        'f"router.{_name}"',
        'f"stage_forward.{_name}"',
        'f"ficem_{_method}.{_name}"',
        'f"state_pack.{stage_name}"',
        'f"state_select.{stage_name}"',
        'f"state_merge.{stage_name}"',
    ):
        assert required in source
    assert "finally:" in source
    assert "for restore in reversed(restored):" in source
    assert "runtime.pack_ephemeral_epi_state = original_pack" in source
    assert "runtime.select_packed_epi_state = original_select" in source
    assert "runtime.merge_packed_epi_state = original_merge" in source


def test_issue665_nested_timings_are_not_naively_summed() -> None:
    source = LAUNCHER.read_text()
    for required in (
        '"route_inclusive_ms"',
        '"stage_forward_inclusive_ms"',
        '"ficem_backend_inclusive_ms"',
        '"derived_stage_compute_excluding_ficem_ms"',
        '"derived_route_glue_excluding_router_state_stage_ms"',
        '"model_glue_outside_stage_routes"',
        '"route_stage_ficem_measurements_are_nested_inclusive": True',
        '"exclusive_fields_are_derived_by_parent_minus_immediate_children": True',
        '"component_level_synchronization": False',
        '"synchronize_only_after_complete_diagnostic_call": True',
        '"naive_sum_of_nested_inclusive_timings_forbidden": True',
    ):
        assert required in source
    assert "max(stage_forward_ms - ficem_ms, 0.0)" in source
    assert "route_ms - router_ms - movement_ms - stage_forward_ms" in source
    assert "max(full_call_ms - route_total, 0.0)" in source


def test_issue665_records_foundation_optional_routes_and_diagnostic_target_only() -> None:
    source = LAUNCHER.read_text()
    assert '"foundation" if index == 0 else f"optional_{index}"' in source
    assert '"route_fractions": first_routes' in source
    assert '"routing_accounting": first_accounting' in source
    assert '"dominant_measured_category": dominant' in source
    assert '"diagnostic_next_target_label": mapping[dominant]' in source
    assert '"optimization_authorized": False' in source


def test_issue665_durable_result_is_fresh_commit_before_result_marker() -> None:
    source = LAUNCHER.read_text()
    assert _literal_constant(LAUNCHER, "RESULT_PATH") == (
        "/vol/aera-v26/issue665-frozen-throughput-component-attribution/result.json"
    )
    assert _literal_constant(LAUNCHER, "SOURCE_RESULT_PATH") == (
        "/vol/aera-v26/issue650-e2e-cli-guard-continuation/result.json"
    )
    assert "if result_path.exists():" in source
    assert "result_path.write_text(" in source
    assert "volume.commit()" in source
    assert "print(RESULT_MARKER" in source
    assert source.index("result_path.write_text(") < source.index("volume.commit()")
    assert source.index("volume.commit()") < source.index("print(RESULT_MARKER")


def test_issue665_defines_exactly_one_l4_and_no_scientific_training_path() -> None:
    source = LAUNCHER.read_text()
    assert source.count('gpu="L4"') == 1
    assert '"gpu": "L4"' in source
    for forbidden in (
        "optimizer.step(",
        ".backward(",
        "loss.backward(",
        "train_loader",
        "TokenBin(",
        "validate_production_data(",
    ):
        assert forbidden not in source
    for false_key in (
        "systems_pass_earned",
        "optimization_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{false_key}": False' in source
        assert f'"{false_key}": True' not in source


def test_issue665_workflow_is_owner_opened_canonical_one_shot() -> None:
    source = WORKFLOW.read_text()
    lowered = source.lower()
    assert "issues:\n    types: [opened]" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "[aera-v26-9-issue665-frozen-throughput-component-attribution]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${#matching_triggers[@]}" = "1"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${matching_triggers[0]}"' in source
    assert "## #665 sole diagnostic L4 authorization" in source
    assert "34039744132" in source
    assert "101504212730" in source
    assert "5559943198" in source
    assert "workflow_dispatch" not in lowered
    assert "modal deploy" not in lowered
    assert "gh run " not in lowered
    assert "rerun" not in lowered
    assert "retry" not in lowered
    assert "redispatch" not in lowered
    assert source.count(
        "modal run modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py"
    ) == 1


def test_issue665_workflow_verifies_frozen_artifact_and_lineage_blobs() -> None:
    source = WORKFLOW.read_text()
    for required in (
        "LAUNCHER_BLOB=",
        "WORKFLOW_BLOB=",
        "CPU_TEST_BLOB=",
        "FROZEN_TREE=",
        "FROZEN_COMMIT=",
        SCIENTIFIC_ADAPTER_BLOB,
        RUNTIME_INTERFACE_BLOB,
        BASE_SYSTEMS_BLOB,
        V26_9_BACKEND_BLOB,
        "git hash-object modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py",
        "git hash-object .github/workflows/aera-v26-9-issue665-frozen-throughput-component-attribution.yml",
        "git hash-object tests/test_aera_v26_9_issue665_frozen_throughput_component_attribution_cpu.py",
    ):
        assert required in source


def test_issue665_result_marker_cannot_be_interpreted_as_systems_pass() -> None:
    source = WORKFLOW.read_text()
    assert "AERA_V26_9_ISSUE665_COMPONENT_ATTRIBUTION_RESULT_JSON=" in source
    assert '.systems_pass_earned == false' in source
    assert '.optimization_authorized == false' in source
    assert '.source_decision == "FAIL_FROZEN_E2E_SYSTEMS_GATE"' in source
    assert '.source_decision_changed == false' in source
