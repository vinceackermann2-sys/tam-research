from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCALIZER = ROOT / "tam_research" / "aera_v26_8_issue578_first_divergence_localizer.py"
LAUNCHER = ROOT / "modal_aera_v26_8_issue578_first_divergence_localizer_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-8-issue578-first-divergence-localizer-l4.yml"

SOURCE_MAIN = "bbf548edf47fd91948c54819f1cee47c4c567ed6"
SOURCE_TREE = "ed4020eec5a51819c5fed3cc02f2895c2c8d8821"
LOCALIZER_BLOB = "5933d1113b950348bb63e5f3eb8713ad36dbf580"
LAUNCHER_BLOB = "cd47e1252bed5617556998659eadfe6a61637d39"
WORKFLOW_BLOB = "b76282733903d220e7118ede283f789db0eb56ba"

FROZEN_BLOBS = {
    ROOT / "tam_research" / "aera_v26_5_end_to_end_systems.py": "c9731cae7e386f09b2a190b045532591c4fa00be",
    ROOT / "tam_research" / "aera_v26_5_end_to_end_systems_repair1.py": "b3f7082b188644007b873db3733492f424d4941a",
    ROOT / "tam_research" / "aera_v26_6_issue530_end_to_end_systems.py": "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b",
    ROOT / "tam_research" / "aera_v26_8_issue562_end_to_end_systems.py": "3534103eea21f7c4d9d31798ad34601fd47090d6",
    ROOT / "tam_research" / "aera_v26_8_issue569_end_to_end_systems_memory_safe.py": "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203",
    ROOT / "tam_research" / "aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py": "3575c58d1cd730be77649f087908c51dbf3e6088",
    ROOT / "tam_research" / "aera_hardware_core_v26_6_ficem_write_materialize_cast.py": "d45c262314a0b4691f26812a279937a225043ad9",
    ROOT / "tam_research" / "aera_hardware_core_v26.py": "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py": "4e336b6e1a6238dac782fa320751d68281493ee1",
    ROOT / "tam_research" / "aera_hardware_core_v25_1.py": "1c3456d8040455b4cd1194db4c8586f77d0f3e43",
    ROOT / "tam_research" / "aera_hardware_core_v19.py": "98008bceb8c68af3bc346e5dfcc7a8218875661e",
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
    raise AssertionError(f"missing literal constant {name}")


def test_issue578_freezes_exact_source_and_harness_blobs() -> None:
    assert _literal_constant(LOCALIZER, "RESEARCH_ISSUE") == 578
    assert _literal_constant(LOCALIZER, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal_constant(LOCALIZER, "SOURCE_TREE") == SOURCE_TREE
    assert _literal_constant(LOCALIZER, "BATCH_SIZES") == (8, 64)
    assert _literal_constant(LOCALIZER, "INTEGRATED_ATOL") == 1e-2
    assert _literal_constant(LOCALIZER, "INTEGRATED_RTOL") == 1e-2
    assert _literal_constant(LOCALIZER, "DUPLICATE_THRESHOLD") == 0.95
    assert _literal_constant(LOCALIZER, "MAX_GPU_SECONDS") == 300
    assert _blob(LOCALIZER) == LOCALIZER_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB


def test_issue578_preserves_frozen_lineage_bytes() -> None:
    for path, expected in FROZEN_BLOBS.items():
        assert _blob(path) == expected, path


def test_issue578_cpu_preflight_authorizes_no_gpu_or_measurement() -> None:
    source = LOCALIZER.read_text()
    assert "def cpu_contract_preflight_issue578()" in source
    assert '"gpu_authorized_by_cpu_preflight": False' in source
    assert '"model_construction_performed": False' in source
    assert '"checkpoint_loaded": False' in source
    assert '"localization_measurement_performed": False' in source
    assert 'triage.DIAGNOSTIC_SEED + 10_000 + batch_size' in source
    assert 'issue562.load_models_v26_8(' in source
    assert 'del transformer' in source
    assert "SYSTEM_WARMUP_CALLS" not in source
    assert "SYSTEM_TIMED_CALLS_PER_ROUND" not in source


def test_issue578_instrumentation_is_observational_and_restored() -> None:
    source = LOCALIZER.read_text()
    assert "@contextmanager\ndef _capture_model" in source
    assert 'object.__setattr__(model, "_route_one_stage"' in source
    assert 'object.__setattr__(router, "forward"' in source
    assert 'object.__setattr__(controller, "forward"' in source
    assert 'object.__setattr__(memory, "read_with_reuse"' in source
    assert 'object.__setattr__(memory, "update_block_from_projected"' in source
    assert 'object.__setattr__(memory, "update_block"' in source
    assert "yield records\n    finally:" in source
    assert "for obj, name, original in reversed(restorations):" in source
    assert "object.__setattr__(obj, name, original)" in source
    assert "return gate, logits" in source
    assert "return output" in source
    assert "return recalled, projected_query, normalized_old_keys" in source
    assert "return result" in source
    for forbidden in (
        "optimizer.step(",
        ".backward(",
        "model.train(",
        "threshold = 0.94",
        "DUPLICATE_THRESHOLD = 0.94",
    ):
        assert forbidden not in source


def test_issue578_captures_required_first_divergence_boundaries() -> None:
    source = LOCALIZER.read_text()
    for required in (
        '"route_call"',
        '"selected_population"',
        '"logits"',
        '"recalled"',
        '"projected_query"',
        '"normalized_old_keys"',
        '"applied_read"',
        '"novelty"',
        '"memory_write"',
        '"depth_logits"',
        '"reasoner_steps"',
        '"selected_write_indices"',
        '"selected_write_strengths"',
        '"projected_new_keys"',
        '"payload_source"',
        '"write_strength"',
        '"initial_new_valid"',
        '"incoming_similarity"',
        '"shadowed_incoming"',
        '"old_similarity"',
        '"duplicate_old"',
        '"keep_old"',
        '"post_memory"',
        '"first_bitwise_difference"',
        '"first_integrated_tolerance_or_metadata_failure"',
        '"first_discrete_decision_difference"',
        '"threshold_margin_diagnostics"',
    ):
        assert required in source
    assert '"actual_candidate_backend_internal_decisions_available": False' in source
    assert '"adjudication_replay_is_diagnostic_only": True' in source
    assert '"kind": "torch_reference_formula_replay_from_exact_update_inputs"' in source


def test_issue578_launcher_is_unique_one_l4_result_boundary() -> None:
    source = LAUNCHER.read_text()
    assert _literal_constant(LAUNCHER, "APP_NAME") == "tam-research-aera-v26-8-issue578-first-divergence-localizer"
    assert _literal_constant(LAUNCHER, "RESULT_PATH") == "/vol/aera-v26/issue578-first-divergence-localizer/result.json"
    assert _literal_constant(LAUNCHER, "MAX_GPU_SECONDS") == 300
    assert _literal_constant(LAUNCHER, "LOCALIZER_BLOB") == LOCALIZER_BLOB
    assert source.count('gpu="L4"') == 1
    assert "timeout=MAX_GPU_SECONDS" in source
    assert "volume.reload()" in source
    assert "if path.exists():" in source
    assert "run_first_divergence_localization(" in source
    assert "tmp.replace(path)" in source
    assert "volume.commit()" in source
    assert "RESULT_MARKER + json.dumps" in source
    assert "SUMMARY_MARKER + json.dumps" in source
    assert '"repair_authorized"] = False' in source
    assert '"architecture_freeze_authorized"] = False' in source
    assert '"100m_authorized"] = False' in source
    assert '"breakthrough_proven"] = False' in source


def test_issue578_workflow_requires_later_authorization_and_canonical_attempt1() -> None:
    source = WORKFLOW.read_text()
    lowered = source.lower()
    assert "issues:\n    types: [opened]" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "[aera-v26-8-issue578-first-divergence-localizer-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "sort -n" in source
    assert 'test "${TRIGGER_ISSUE}" = "${eligible[0]}"' in source
    assert "## #578 sole L4 diagnostic authorization" in source
    assert "Authorize main:" in source
    assert "33741700781" in source and "100604889696" in source
    assert "33744802059" in source and "100614716963" in source
    assert LOCALIZER_BLOB in source
    assert LAUNCHER_BLOB in source
    assert source.count("modal run modal_aera_v26_8_issue578_first_divergence_localizer_app.py") == 1
    assert "workflow_dispatch" not in source
    assert "gh run rerun" not in lowered
    assert "rerun-failed" not in lowered
    assert "modal deploy" not in lowered
    assert "cancel-in-progress: true" not in lowered


def test_issue578_higher_authorizations_remain_false() -> None:
    source = LOCALIZER.read_text() + "\n" + LAUNCHER.read_text()
    for key in (
        "repair_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{key}": False' in source or f'"{key}"] = False' in source
