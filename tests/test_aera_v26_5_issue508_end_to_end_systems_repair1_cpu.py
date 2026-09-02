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
EXHAUSTED_LAUNCHER = ROOT / "modal_aera_v26_5_issue505_end_to_end_systems_app.py"
EXHAUSTED_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-5-issue505-e2e-systems-l4.yml"
SUCCESSOR_LAUNCHER = ROOT / "modal_aera_v26_5_issue508_end_to_end_systems_repair1_app.py"
SUCCESSOR_WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-5-issue508-e2e-systems-l4-repair1.yml"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue508_freezes_source_and_historical_attempt_blobs() -> None:
    assert _blob(HISTORICAL) == "c9731cae7e386f09b2a190b045532591c4fa00be"
    assert _blob(REPAIRED) == "b3f7082b188644007b873db3733492f424d4941a"
    assert _blob(V26) == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert _blob(READ_BACKEND) == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert _blob(WRITE_BACKEND) == "e54570292489bd17570038dca7518419ac00418c"
    assert _blob(STABLE_REFERENCE) == "4e336b6e1a6238dac782fa320751d68281493ee1"
    assert _blob(EXHAUSTED_LAUNCHER) == "491fe31b0f701c1a67e91a9bd877069eeba42e55"
    assert _blob(EXHAUSTED_WORKFLOW) == "06240f93cf779050a8615957fe431bc7e5fd2ccd"
    assert _blob(SUCCESSOR_LAUNCHER) == "5597dbbd79c782420d48ed538ef2669aebfe5fae"
    assert _blob(SUCCESSOR_WORKFLOW) == "556ea59ebc2d95272caa774a9fef62efbf66a302"


def test_issue508_root_cause_is_only_checkpoint_hash_inventory_key_names() -> None:
    historical_source = HISTORICAL.read_text()
    exhausted_source = EXHAUSTED_LAUNCHER.read_text()
    successor_source = SUCCESSOR_LAUNCHER.read_text()

    assert 'paths = {"aera": root / "aera.pt", "transformer": root / "transformer.pt"}' in historical_source
    assert 'return {name: _sha256(path) for name, path in paths.items()}' in historical_source
    assert 'if set(hashes) != {"aera.pt", "transformer.pt"}:' in exhausted_source

    assert 'CHECKPOINT_HASH_KEYS = frozenset({"aera", "transformer"})' in successor_source
    assert "if set(hashes) != CHECKPOINT_HASH_KEYS:" in successor_source
    assert '{"aera.pt", "transformer.pt"}' not in successor_source
    assert 'historical.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471"' in successor_source
    assert "hashes = historical.checkpoint_hashes(historical.CHECKPOINT_RELATIVE_DIR)" in successor_source
    assert '"checkpoint_hash_key_repair": True' in successor_source


def test_issue508_preserves_exact_systems_workload_thresholds_and_repair_semantics() -> None:
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
    assert protocol["repair_issue"] == 503
    assert protocol["top_level_inference_decorated"] is False
    assert protocol["model_construction_outside_inference_mode"] is True
    assert protocol["parameter_version_snapshots_outside_inference_mode"] is True
    assert protocol["measurements_inside_explicit_inference_mode"] is True
    assert protocol["batch_sizes"] == [8, 64]
    assert protocol["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert protocol["route_mode"] == "hard_sparse"
    assert protocol["physically_real_sparse_required"] is True
    assert protocol["dense_masked_sparse_credit"] is False
    assert protocol["integrated_atol"] == 1e-2
    assert protocol["integrated_rtol"] == 1e-2
    assert protocol["persistent_state_bytes_per_session"] == 77_760
    assert protocol["production_write_geometry"] == [16, 255, 1]


def test_issue508_successor_launcher_is_one_l4_one_result_and_exact_evaluator_call() -> None:
    source = SUCCESSOR_LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-5-issue508-end-to-end-systems-repair1"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue508-end-to-end-systems-repair1/result.json"' in source
    assert "MAX_GPU_SECONDS = 600" in source
    assert 'SOURCE_MAIN = "371c97380c1488689a6a1ddacfb89f47a64aabfc"' in source
    assert source.count('gpu="L4"') == 1
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("run_end_to_end_systems_repair1()") == 1
    assert source.count("if result_path.exists():") == 2
    assert "result_path.write_text(durable_json)" in source
    assert "volume.commit()" in source
    assert source.index("result_path.write_text(durable_json)") < source.index("volume.commit()")
    assert source.index("volume.commit()") < source.index(
        '"AERA_V26_5_ISSUE508_END_TO_END_SYSTEMS_REPAIR1_RESULT_JSON="'
    )
    assert '"AERA_V26_5_ISSUE508_END_TO_END_SYSTEMS_REPAIR1_SUMMARY_JSON="' in source
    assert '"AERA_V26_5_ISSUE508_END_TO_END_SYSTEMS_REPAIR1_PREFLIGHT_JSON="' in source
    assert '"AERA_V26_5_ISSUE508_END_TO_END_SYSTEMS_REPAIR1_L4_START_JSON="' in source
    for forbidden in (
        "torch.optim",
        ".backward(",
        "torch.save(",
        "modal.Function.lookup",
        "modal deploy",
    ):
        assert forbidden not in source


def test_issue508_successor_launcher_records_exhausted_attempt_and_no_science_change() -> None:
    source = SUCCESSOR_LAUNCHER.read_text()
    for required in (
        "EXHAUSTED_ISSUE = 505",
        "EXHAUSTED_TRIGGER = 507",
        "EXHAUSTED_RUN = 33660377370",
        "EXHAUSTED_JOB = 100349177580",
        'EXHAUSTED_LAUNCHER_BLOB = "491fe31b0f701c1a67e91a9bd877069eeba42e55"',
        'EXHAUSTED_WORKFLOW_BLOB = "06240f93cf779050a8615957fe431bc7e5fd2ccd"',
        '"model_construction_outside_inference_mode": True',
        '"parameter_version_snapshots_outside_inference_mode": True',
        '"measurements_inside_explicit_inference_mode": True',
        '"physically_real_sparse_required": True',
        '"dense_masked_sparse_credit": False',
        '"scientific_seed_consumed": False',
        '"architecture_freeze_authorized": False',
        '"100m_authorized": False',
        '"breakthrough_proven": False',
    ):
        assert required in source


def test_issue508_workflow_is_distinct_attempt1_bound_main_and_duplicate_safe() -> None:
    source = SUCCESSOR_WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "cancel-in-progress: false" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "startsWith(github.event.issue.title, '[aera-v26-5-issue508-e2e-systems-l4-repair1]')" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'startswith("🔬 **AERA-v26.5 #508 end-to-end systems repair1 result**")' in source
    assert 'test "${report_count}" = "0"' in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor 371c97380c1488689a6a1ddacfb89f47a64aabfc HEAD" in source
    assert source.count("modal run modal_aera_v26_5_issue508_end_to_end_systems_repair1_app.py") == 1
    assert "modal deploy" not in source
    assert "gh run rerun" not in source
    assert "/rerun" not in source


def test_issue508_workflow_freezes_exhausted_failure_and_all_source_blobs() -> None:
    source = SUCCESSOR_WORKFLOW.read_text()
    assert "issues/507" in source
    assert "actions/runs/33660377370" in source
    assert "actions/jobs/100349177580" in source
    assert "'.conclusion')\" = \"failure\"" in source
    assert "actions/runs/33659021797" in source
    assert "actions/jobs/100344666507" in source
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
        "06240f93cf779050a8615957fe431bc7e5fd2ccd",
        "5597dbbd79c782420d48ed538ef2669aebfe5fae",
    ):
        assert blob in source


def test_issue508_higher_authorizations_remain_false() -> None:
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
