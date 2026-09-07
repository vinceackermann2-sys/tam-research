from __future__ import annotations

import hashlib
from pathlib import Path

LAUNCHER = Path("modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py")
PREFLIGHT = Path("modal_aera_v26_9_issue687_stage_internal_preflight.py")
WORKFLOW = Path(".github/workflows/aera-v26-9-issue687-stage-internal-throughput-attribution.yml")
TEST_FILE = Path("tests/test_aera_v26_9_issue687_stage_internal_throughput_attribution_cpu.py")
ISSUE665 = Path("modal_aera_v26_9_issue665_frozen_throughput_component_attribution.py")
ADAPTER = Path("tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py")
RUNTIME = Path("tam_research/aera_hardware_core_v26.py")
STAGE_V25_1 = Path("tam_research/aera_hardware_core_v25_1.py")
TOKENWISE_V19 = Path("tam_research/aera_hardware_core_v19.py")
NOHOST = Path("tam_research/aera_hardware_core_v25_1_nohost.py")
BACKEND = Path("tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py")

EXPECTED_LAUNCHER_BLOB = "b9950c032686a496c6d944c30c441428499cb367"
EXPECTED_PREFLIGHT_BLOB = "d36345b3f9fdd62411b3fa7b5d43538abc67dbdf"
EXPECTED_WORKFLOW_BLOB = "5698f15d2ccdbe982383f1dde9cc269abef0fb5e"
EXPECTED_ISSUE665_BLOB = "72f27391ff2f0a7bff8d4532f307ddc4869cf494"
EXPECTED_ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
EXPECTED_RUNTIME_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
EXPECTED_STAGE_V25_1_BLOB = "1c3456d8040455b4cd1194db4c8586f77d0f3e43"
EXPECTED_TOKENWISE_V19_BLOB = "98008bceb8c68af3bc346e5dfcc7a8218875661e"
EXPECTED_NOHOST_BLOB = "237e5615cf32f644e8675808a6b3e9adaf04fb23"
EXPECTED_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"

PREAUTH_PREFIX = "[aera-v26-9-issue687-stage-internal-preauth]"
L4_PREFIX = "[aera-v26-9-issue687-stage-internal-l4]"
MANIFEST = "## #687 pre-implementation stage-internal attribution freeze"
AUTH = "## #687 sole L4 stage-internal attribution authorization"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue687_exact_candidate_blobs_and_frozen_lineage() -> None:
    for path in (LAUNCHER, PREFLIGHT, WORKFLOW, TEST_FILE):
        assert path.is_file()
    assert _blob(LAUNCHER) == EXPECTED_LAUNCHER_BLOB
    assert _blob(PREFLIGHT) == EXPECTED_PREFLIGHT_BLOB
    assert _blob(WORKFLOW) == EXPECTED_WORKFLOW_BLOB
    assert _blob(ISSUE665) == EXPECTED_ISSUE665_BLOB
    assert _blob(ADAPTER) == EXPECTED_ADAPTER_BLOB
    assert _blob(RUNTIME) == EXPECTED_RUNTIME_BLOB
    assert _blob(STAGE_V25_1) == EXPECTED_STAGE_V25_1_BLOB
    assert _blob(TOKENWISE_V19) == EXPECTED_TOKENWISE_V19_BLOB
    assert _blob(NOHOST) == EXPECTED_NOHOST_BLOB
    assert _blob(BACKEND) == EXPECTED_BACKEND_BLOB


def test_launcher_freezes_source_evidence_and_diagnostic_design() -> None:
    text = LAUNCHER.read_text()
    for value in (
        "bc74136533f27d9f39bf437f52975efd7a929fd4",
        "c8835002261c1b249f90468085ad0c868afb2d90",
        "SOURCE_TRIGGER = 686",
        "SOURCE_EVIDENCE_COMMENT = 5562153718",
        "50.84833526611328",
        "29.77371195331216",
        "117.23713684082031",
        "59.66491297632456",
        "FAIL_FROZEN_E2E_SYSTEMS_GATE",
        "/vol/aera-v26/issue687-stage-internal-throughput-attribution/result.json",
    ):
        assert value in text
    assert "SYSTEM_BATCH_SIZES = tuple(issue665.SYSTEM_BATCH_SIZES)" in text
    assert "TOKEN_SEED_BASE = issue665.TOKEN_SEED_BASE" in text
    assert "TOKEN_SEED_OFFSET = issue665.TOKEN_SEED_OFFSET" in text
    assert "DIAGNOSTIC_WARMUP_CALLS = issue665.DIAGNOSTIC_WARMUP_CALLS" in text
    assert "DIAGNOSTIC_MEASURED_CALLS = issue665.DIAGNOSTIC_MEASURED_CALLS" in text


def test_launcher_instruments_without_replacing_model_operations() -> None:
    text = LAUNCHER.read_text()
    for label in (
        "stage_forward.",
        "context.",
        "norm",
        "controller_start",
        "controller_end",
        "state_to_chunk",
        "attention",
        "experts",
        "reasoner",
        "reason_to_chunk",
        "out_norm",
        "stream_input_norm",
        "stream_cell",
        "pair_write_gate",
        "event_pair_select",
        "ficem_read",
        "ficem_update",
        "ficem_update_from_projected",
    ):
        assert label in text
    assert "for restore in reversed(restored):" in text
    assert "v25_1.select_budgeted_event_pairs = original_select" in text
    assert "same_stream_cuda_events" in text
    assert "component_level_synchronization" in text
    assert "naive_sum_of_nested_inclusive_timings_forbidden" in text
    assert "context_exclusive_derived_by_subtracting_ficem_read" in text
    assert "stage_compute_derived_by_subtracting_all_ficem_backend_time" in text


def test_launcher_categories_and_authority_are_frozen() -> None:
    text = LAUNCHER.read_text()
    categories = (
        "normalization_and_start_controller",
        "context_integration_outside_ficem",
        "attention",
        "sparse_experts",
        "end_controller",
        "latent_reasoner",
        "reason_to_chunk_and_output_norm",
        "recurrent_stream_update",
        "write_gate_and_event_pair_selection",
        "residual_stage_glue",
    )
    for category in categories:
        assert category in text
    assert 'gpu="L4"' in text
    assert text.count('gpu="L4"') == 1
    assert "torch.cuda.synchronize()" in text
    assert "reference_model_executed\": False" in text
    assert "transformer_model_executed\": False" in text
    assert "scientific_seed_consumed\": False" in text
    assert "optimization_authorized\": False" in text
    assert "systems_pass_earned\": False" in text
    assert "100m_authorized\": False" in text
    assert "breakthrough_proven\": False" in text


def test_preauthorization_wrapper_is_cpu_preflight_only() -> None:
    text = PREFLIGHT.read_text()
    assert text.count("diagnostic.preflight.remote()") == 1
    assert "run_diagnostic" not in text
    assert "gpu=" not in text
    assert "volume.commit" not in text
    assert "torch" not in text
    assert "AERA_V26_9_ISSUE687_STAGE_INTERNAL_PREAUTH_JSON=" in text


def test_workflow_has_two_disjoint_owner_only_modes() -> None:
    text = WORKFLOW.read_text()
    assert "on:\n  issues:\n    types: [opened]" in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert PREAUTH_PREFIX in text
    assert L4_PREFIX in text
    assert PREAUTH_PREFIX != L4_PREFIX
    assert MANIFEST in text
    assert AUTH in text
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in text
    assert "SOURCE_EVIDENCE_COMMENT=5562153718" in text
    assert "issues/686" in text
    assert "5562153718" in text


def test_workflow_executes_only_expected_modal_targets() -> None:
    text = WORKFLOW.read_text()
    preauth = "modal run modal_aera_v26_9_issue687_stage_internal_preflight.py::main"
    l4 = "modal run modal_aera_v26_9_issue687_stage_internal_throughput_attribution.py::main"
    modal_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("modal run ")]
    assert len(modal_lines) == 2
    assert modal_lines[0].startswith(preauth)
    assert modal_lines[1].startswith(l4)
    assert text.index("Re-verify issue687 trigger immediately before Modal") < text.index(preauth)
    assert text.index("Re-verify issue687 trigger immediately before Modal") < text.index(l4)
    assert text.index("Authenticate Modal") < text.index(preauth)
    assert text.index("Authenticate Modal") < text.index(l4)


def test_workflow_requires_preauth_before_l4_authorization_path() -> None:
    text = WORKFLOW.read_text()
    assert "#687 preauthorization result-path absence evidence" in text
    assert "#687 sole L4 stage-internal attribution authorization" in text
    assert "preauth_count" in text
    assert 'test "${preauth_count}" = "1"' in text
    assert "auth_count" in text
    assert 'test "${auth_count}" = "1"' in text
    assert "steps.guard.outputs.mode == 'preauth'" in text
    assert "steps.guard.outputs.mode == 'l4'" in text


def test_workflow_yaml_block_shape_and_no_alternate_dispatch() -> None:
    text = WORKFLOW.read_text()
    allowed = ("name:", "on:", "permissions:", "concurrency:", "jobs:")
    unexpected = [
        (number, line)
        for number, line in enumerate(text.splitlines(), 1)
        if line and not line[0].isspace() and not line.startswith(allowed)
    ]
    assert unexpected == []
    assert "\nPY\n" not in text
    assert text.count("          PY\n") == 2
    lowered = text.lower()
    assert "workflow_dispatch" not in lowered
    assert "modal deploy" not in lowered
    assert "rerun" not in lowered
    assert "redispatch" not in lowered


def test_no_higher_stage_true_authority_in_issue687_files() -> None:
    combined = LAUNCHER.read_text().lower() + "\n" + WORKFLOW.read_text().lower()
    for phrase in (
        "optimization_authorized=true",
        "systems_pass_earned=true",
        "architecture_freeze_authorized=true",
        "s2_authorized=true",
        "fresh_scientific_seed_authorized=true",
        "independent_replication_credit=true",
        "100m_authorized=true",
        "breakthrough_proven=true",
    ):
        assert phrase not in combined
