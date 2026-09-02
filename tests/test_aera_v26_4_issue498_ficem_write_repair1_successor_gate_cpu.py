from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_hardware_core_v26_4_ficem_write_triton as write_backend
from tam_research import aera_v26_4_ficem_write_probe as probe

ROOT = Path(__file__).resolve().parents[1]
WRITE_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
V26_INTERFACE = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
WRITE_PROBE = ROOT / "tam_research" / "aera_v26_4_ficem_write_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_4_ficem_write_repair1_successor_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-4-ficem-write-repair1-successor-l4.yml"

SOURCE_MAIN = "852edbe5261f75f56f45e52b33d6acf3bea02f2b"
WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
WRITE_PROBE_BLOB = "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
LAUNCHER_BLOB = "a0de551981ebc833149839ae9362e156163b21f5"
WORKFLOW_BLOB = "766b6af4347e8fdb49e73a79e3640cf56c598bef"
WRITE_REPAIR_PR = 497
WRITE_REPAIR_CPU_RUN = 33647126135
WRITE_REPAIR_CPU_JOB = 100304584393
WRITE_REPAIR_CPU_HEAD = "515d3fad8463c8e61da664b2667b7a4616ab3864"
EXHAUSTED_WRITE_RUN = 33638047466
EXHAUSTED_WRITE_JOB = 100273784137
READ_PASS_RUN = 33618950619
READ_PASS_JOB = 100211244996
TRIGGER_PREFIX = "[aera-v26-4-ficem-write-l4-repair1-successor]"
RESULT_HEADING = "🔬 **AERA-v26.4 #498 repaired FICEM WRITE successor result**"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue498_freezes_candidate_reference_probe_launcher_and_workflow_blobs() -> None:
    assert _blob(WRITE_BACKEND) == WRITE_BACKEND_BLOB
    assert _blob(READ_BACKEND) == READ_BACKEND_BLOB
    assert _blob(V26_INTERFACE) == V26_INTERFACE_BLOB
    assert _blob(STABLE_REFERENCE) == STABLE_REFERENCE_BLOB
    assert _blob(WRITE_PROBE) == WRITE_PROBE_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB


def test_issue498_repair_protocol_is_exact_and_higher_authorizations_are_false() -> None:
    protocol = write_backend.fused_ficem_read_write_v26_4_protocol()
    assert protocol["duplicate_similarity"] == 0.95
    assert protocol["write_tail_triton_launches_target"] == 2
    assert protocol["write_threshold_input_dtype_visibility_repair1"] is True
    assert protocol["write_numeric_duplicate_threshold_changed_by_repair1"] is False
    assert protocol["float32_write_threshold_semantics_changed_by_repair1"] is False
    assert protocol["write_tail_kernel_count_changed_by_repair1"] is False
    assert protocol["write_threshold_repair1_issue"] == 495
    assert protocol["write_pre_repair_backend_blob"] == "5d703bbba296328ca2f49407e56192d10541349d"
    assert protocol["repair5_read_backend_blob"] == READ_BACKEND_BLOB
    assert protocol["v26_interface_blob"] == V26_INTERFACE_BLOB
    assert protocol["stable_compaction_reference_blob"] == STABLE_REFERENCE_BLOB
    for key in (
        "write_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue498_inherits_frozen_issue488_design_and_thresholds_exactly() -> None:
    assert probe.DESIGN_SEED == 408_487
    assert (probe.D_MODEL, probe.WRITE_COUNT, probe.CAPACITY, probe.MEMORY_DIM) == (
        200,
        16,
        48,
        50,
    )
    assert probe.DUPLICATE_THRESHOLD == 0.95
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert probe.EDGE_FIXTURES == (
        "empty_old_all_new_valid",
        "mixed_incoming_validity",
        "incoming_duplicate_newest_wins",
        "threshold_inclusive_and_below_control",
        "surviving_new_suppresses_old",
        "shadowed_new_does_not_suppress_old",
        "over_capacity_truncation",
        "invalid_retained_storage_order",
    )
    assert (probe.WARMUP_CALLS, probe.TIMED_ROUNDS, probe.CALLS_PER_ROUND) == (10, 5, 100)
    assert (probe.FP32_ATOL, probe.FP32_RTOL) == (1e-5, 1e-5)
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert probe.MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert probe.MAX_ROW_LATENCY_RATIO == 1.05
    assert probe.MAX_TAIL_EVENT_RATIO == 0.75
    contract = probe.cpu_contract_preflight()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["synthetic_only"] is True
    assert contract["scientific_seed_consumed"] is False
    protocol = probe.issue488_protocol()
    for key in (
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue498_launcher_is_duplicate_safe_one_l4_one_probe_and_persists_first() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-4-issue498-ficem-write-repair1-successor"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue498-ficem-write-repair1-successor/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("result = run_ficem_write_probe()") == 1
    assert source.count("result_path.exists()") >= 2
    assert "refusing duplicate issue498 repaired WRITE successor run because result exists" in source
    assert SOURCE_MAIN in source
    assert WRITE_BACKEND_BLOB in source
    assert READ_BACKEND_BLOB in source
    assert V26_INTERFACE_BLOB in source
    assert STABLE_REFERENCE_BLOB in source
    assert WRITE_PROBE_BLOB in source
    assert str(WRITE_REPAIR_CPU_RUN) in source
    assert str(WRITE_REPAIR_CPU_JOB) in source
    assert WRITE_REPAIR_CPU_HEAD in source
    assert str(EXHAUSTED_WRITE_RUN) in source
    assert str(EXHAUSTED_WRITE_JOB) in source
    assert str(READ_PASS_RUN) in source
    assert str(READ_PASS_JOB) in source
    write = source.index("result_path.write_text(durable_json)")
    commit = source.index("volume.commit()")
    digest = source.index("digest = hashlib.sha256")
    marker = source.index("AERA_V26_ISSUE498_FICEM_WRITE_REPAIR1_SUCCESSOR_RESULT_JSON=")
    summary = source.index("AERA_V26_ISSUE498_FICEM_WRITE_REPAIR1_SUCCESSOR_SUMMARY_JSON=")
    assert write < commit < digest < marker < summary
    assert '"bfloat16_threshold_inclusive_and_below_control"' in source
    assert '"two_kernel_topology_pass"' in source
    assert '"geomean_latency_ratio_by_dtype"' in source
    assert '"synthetic_only": True' in source
    assert '"scientific_seed_consumed": False' in source
    for forbidden in (
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        "seed8471",
        "workflow_dispatch",
    ):
        assert forbidden not in source


def test_issue498_workflow_is_unique_attempt1_bound_and_uses_actual_report_heading() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert TRIGGER_PREFIX in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert f'startswith("{RESULT_HEADING}")' in source
    assert 'test "${report_count}" = "0"' in source
    assert 'contains("AERA_V26_ISSUE498_FICEM_WRITE_REPAIR1_SUCCESSOR_RESULT_JSON=")' not in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert str(WRITE_REPAIR_PR) in source
    assert WRITE_REPAIR_CPU_HEAD in source
    assert str(WRITE_REPAIR_CPU_RUN) in source
    assert str(WRITE_REPAIR_CPU_JOB) in source
    assert str(EXHAUSTED_WRITE_RUN) in source
    assert str(EXHAUSTED_WRITE_JOB) in source
    assert str(READ_PASS_RUN) in source
    assert str(READ_PASS_JOB) in source
    assert WRITE_BACKEND_BLOB in source
    assert READ_BACKEND_BLOB in source
    assert V26_INTERFACE_BLOB in source
    assert STABLE_REFERENCE_BLOB in source
    assert WRITE_PROBE_BLOB in source
    assert LAUNCHER_BLOB in source
    assert source.count("modal run modal_aera_v26_4_ficem_write_repair1_successor_app.py") == 1
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source
    assert "automatic retry" in source
    assert "redispatch" in source
    assert "alternate trigger" in source
    assert "timeout increase" in source


def test_issue498_workflow_permissions_and_reporting_are_narrow() -> None:
    source = WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests: read" in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert "pull-requests: write" not in permissions
    assert source.count("continue-on-error: true") == 3
    assert "Durable Modal result + authoritative marker are the experiment record" in source
    assert "No end-to-end/freeze/S2/scientific-seed/100M/breakthrough authorization" in source
