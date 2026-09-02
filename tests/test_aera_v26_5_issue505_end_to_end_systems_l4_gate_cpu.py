from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_5_end_to_end_systems as historical
from tam_research import aera_v26_5_end_to_end_systems_repair1 as repaired

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "tam_research" / "aera_v26_5_end_to_end_systems.py"
REPAIRED = ROOT / "tam_research" / "aera_v26_5_end_to_end_systems_repair1.py"
V26 = ROOT / "tam_research" / "aera_hardware_core_v26.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
WRITE_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
LAUNCHER = ROOT / "modal_aera_v26_5_issue505_end_to_end_systems_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-5-issue505-e2e-systems-l4.yml"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue505_freezes_source_evaluator_backend_and_harness_blobs() -> None:
    assert _blob(HISTORICAL) == "c9731cae7e386f09b2a190b045532591c4fa00be"
    assert _blob(REPAIRED) == "b3f7082b188644007b873db3733492f424d4941a"
    assert _blob(V26) == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert _blob(READ_BACKEND) == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert _blob(WRITE_BACKEND) == "e54570292489bd17570038dca7518419ac00418c"
    assert _blob(STABLE_REFERENCE) == "4e336b6e1a6238dac782fa320751d68281493ee1"
    assert _blob(LAUNCHER) == "491fe31b0f701c1a67e91a9bd877069eeba42e55"
    assert _blob(WORKFLOW) == "06240f93cf779050a8615957fe431bc7e5fd2ccd"


def test_issue505_freezes_exact_systems_workload_and_pass_thresholds() -> None:
    assert historical.CHECKPOINT_RELATIVE_DIR == "/vol/aera-real-language/v25-dev-seed8471"
    assert historical.SOURCE_CHECKPOINT_SEED == 8471
    assert historical.SYSTEM_BATCH_SIZES == (8, 64)
    assert (
        historical.SYSTEM_WARMUP_CALLS,
        historical.SYSTEM_TIMED_CALLS_PER_ROUND,
        historical.SYSTEM_ROUNDS,
    ) == (3, 20, 5)
    assert historical.BATCH8_MIN_FULL_SPEED_RATIO == 0.25
    assert historical.BATCH64_MIN_FULL_SPEED_RATIO == 1.25
    assert (historical.INTEGRATED_ATOL, historical.INTEGRATED_RTOL) == (1e-2, 1e-2)
    assert historical.EXPECTED_STATE_BYTES == 77_760
    assert (
        historical.EXPECTED_SELECTED_WRITES,
        historical.EXPECTED_CANDIDATES,
        historical.EXPECTED_VECTOR_UPDATES,
    ) == (16, 255, 1)
    assert historical.MAX_GPU_SECONDS == 600

    protocol = repaired.repair1_protocol()
    assert protocol["batch_sizes"] == [8, 64]
    assert protocol["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert protocol["timing_order"] == "rotated interleaved conditions per issue381"
    assert protocol["timing_clock"] == "CUDA events with synchronize before/after"
    assert protocol["hard"] is True
    assert protocol["route_mode"] == "hard_sparse"
    assert protocol["physically_real_sparse_required"] is True
    assert protocol["dense_masked_sparse_credit"] is False
    assert protocol["integrated_atol"] == 1e-2
    assert protocol["integrated_rtol"] == 1e-2
    assert protocol["persistent_state_bytes_per_session"] == 77_760
    assert protocol["production_write_geometry"] == [16, 255, 1]
    assert protocol["batch8_min_full_speed_ratio"] == 0.25
    assert protocol["batch64_min_full_speed_ratio"] == 1.25
    historical_source = HISTORICAL.read_text()
    assert "no_reference_latency_regression = candidate_ms <= reference_ms" in historical_source


def test_issue505_freezes_issue503_version_tracking_repair_semantics() -> None:
    protocol = repaired.repair1_protocol()
    assert protocol["repair_issue"] == 503
    assert protocol["predecessor_module_blob"] == "c9731cae7e386f09b2a190b045532591c4fa00be"
    assert protocol["top_level_inference_decorated"] is False
    assert protocol["model_construction_outside_inference_mode"] is True
    assert protocol["parameter_version_snapshots_outside_inference_mode"] is True
    assert protocol["measurements_inside_explicit_inference_mode"] is True
    assert protocol["historical_issue501_module_mutated"] is False


def test_issue505_launcher_is_one_l4_one_durable_result_and_exact_evaluator_call() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-5-issue505-end-to-end-systems"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue505-end-to-end-systems/result.json"' in source
    assert "MAX_GPU_SECONDS = 600" in source
    assert 'SOURCE_MAIN = "4dece4ef767d8de58b74acd091e1adb55009c5ab"' in source
    assert source.count('gpu="L4"') == 1
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("run_end_to_end_systems_repair1()") == 1
    assert source.count("if result_path.exists():") == 2
    assert "volume.reload()" in source
    assert "result_path.write_text(durable_json)" in source
    assert "volume.commit()" in source
    assert source.index("result_path.write_text(durable_json)") < source.index("volume.commit()")
    assert source.index("volume.commit()") < source.index(
        '"AERA_V26_5_ISSUE505_END_TO_END_SYSTEMS_RESULT_JSON="'
    )
    assert '"AERA_V26_5_ISSUE505_END_TO_END_SYSTEMS_SUMMARY_JSON="' in source
    assert '"AERA_V26_5_ISSUE505_END_TO_END_SYSTEMS_PREFLIGHT_JSON="' in source
    assert '"AERA_V26_5_ISSUE505_END_TO_END_SYSTEMS_L4_START_JSON="' in source
    for forbidden in ("torch.optim", ".backward(", "torch.save(", "modal.Function.lookup", "modal deploy"):
        assert forbidden not in source


def test_issue505_launcher_preflight_freezes_repair_and_primitive_evidence() -> None:
    source = LAUNCHER.read_text()
    for required in (
        'REPAIR_CPU_HEAD = "58d04a12edaef50caf5daa24eb0386e8c624c6ca"',
        "REPAIR_CPU_RUN = 33657968851",
        "REPAIR_CPU_JOB = 100341171002",
        "READ_PASS_RUN = 33618950619",
        "READ_PASS_JOB = 100211244996",
        "WRITE_PASS_RUN = 33651216734",
        "WRITE_PASS_JOB = 100318422299",
        '"model_construction_outside_inference_mode": True',
        '"parameter_version_snapshots_outside_inference_mode": True',
        '"measurements_inside_explicit_inference_mode": True',
        '"physically_real_sparse_required": True',
        '"dense_masked_sparse_credit": False',
    ):
        assert required in source


def test_issue505_workflow_is_issue_open_only_attempt1_bound_main_and_duplicate_safe() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "cancel-in-progress: false" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "startsWith(github.event.issue.title, '[aera-v26-5-issue505-e2e-systems-l4]')" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'startswith("🔬 **AERA-v26.5 #505 end-to-end systems result**")' in source
    assert 'test "${report_count}" = "0"' in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor 4dece4ef767d8de58b74acd091e1adb55009c5ab HEAD" in source
    assert source.count("modal run modal_aera_v26_5_issue505_end_to_end_systems_app.py") == 1
    assert "modal deploy" not in source
    assert "continue-on-error: true" in source


def test_issue505_workflow_freezes_exact_cpu_green_and_blob_guards() -> None:
    source = WORKFLOW.read_text()
    assert "pulls/504" in source
    assert "58d04a12edaef50caf5daa24eb0386e8c624c6ca" in source
    assert "4dece4ef767d8de58b74acd091e1adb55009c5ab" in source
    assert "actions/runs/33657968851" in source
    assert "actions/jobs/100341171002" in source
    assert "actions/runs/33618950619" in source
    assert "actions/jobs/100211244996" in source
    assert "actions/runs/33651216734" in source
    assert "actions/jobs/100318422299" in source
    for blob in (
        "c9731cae7e386f09b2a190b045532591c4fa00be",
        "b3f7082b188644007b873db3733492f424d4941a",
        "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
        "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
        "e54570292489bd17570038dca7518419ac00418c",
        "4e336b6e1a6238dac782fa320751d68281493ee1",
        "491fe31b0f701c1a67e91a9bd877069eeba42e55",
    ):
        assert blob in source


def test_issue505_higher_authorizations_remain_false() -> None:
    protocol = repaired.repair1_protocol()
    for key in (
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False
    contract = repaired.cpu_contract_preflight_repair1()
    assert contract["scientific_seed_consumed"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False
