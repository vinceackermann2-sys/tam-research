from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_5_end_to_end_systems_repair1 as repair1
from tam_research import aera_v26_6_issue530_end_to_end_systems as systems
from tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast import (
    MaterializeCastTritonFICEMReadWriteBackend,
)


RESEARCH_ISSUE = 534
SOURCE_MAIN = "67b9559cafaf72d08261ff5c988233f2bc20932b"
SYSTEMS_ADAPTER_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
LAUNCHER_BLOB = "b8fe5715375a1a7eb87311cca053ee071d5736c0"
WORKFLOW_BLOB = "ecf67b64768836dcc548bf5390912bab1f4e8249"
V26_6_CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
ISSUE527_RESULT_SHA256 = "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
ISSUE527_ORACLE_BLOB = "8f472451af4024bb3faacb56d814f7d6bdb25cc9"
ISSUE527_PROBE_BLOB = "bcfeb6a93ed062b7d00359603dc9fbc7aca5767f"

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_6_issue534_end_to_end_systems_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-6-issue534-e2e-systems-l4.yml"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue534_freezes_exact_lineage_and_new_harness_blobs() -> None:
    assert _git_blob_sha(Path(systems.__file__)) == SYSTEMS_ADAPTER_BLOB
    assert _git_blob_sha(Path(base.__file__)) == "c9731cae7e386f09b2a190b045532591c4fa00be"
    assert _git_blob_sha(Path(repair1.__file__)) == "b3f7082b188644007b873db3733492f424d4941a"
    candidate_module = inspect.getmodule(MaterializeCastTritonFICEMReadWriteBackend)
    assert candidate_module is not None
    assert _git_blob_sha(Path(candidate_module.__file__)) == V26_6_CANDIDATE_BLOB

    assert _git_blob_sha(
        ROOT / "tam_research/aera_v26_6_issue525_mixed_dtype_write_oracle.py"
    ) == ISSUE527_ORACLE_BLOB
    assert _git_blob_sha(
        ROOT / "tam_research/aera_v26_6_issue527_ficem_write_repaired_oracle_probe.py"
    ) == ISSUE527_PROBE_BLOB

    assert _git_blob_sha(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob_sha(WORKFLOW) == WORKFLOW_BLOB


def test_issue534_freezes_issue530_merge_and_authoritative_issue527_pass() -> None:
    protocol = systems.issue530_systems_protocol()

    assert systems.RESEARCH_ISSUE == 530
    assert systems.SOURCE_MAIN == "2c0c28005bff8d9b4f36a96de86144dd74107e39"
    assert systems.V26_6_CANDIDATE_BLOB == V26_6_CANDIDATE_BLOB

    assert systems.ISSUE527_TRIGGER == 529
    assert systems.ISSUE527_RUN == 33680028132
    assert systems.ISSUE527_JOB == 100414089065
    assert systems.ISSUE527_ATTEMPT == 1
    assert systems.ISSUE527_BOUND_MAIN == "2c0c28005bff8d9b4f36a96de86144dd74107e39"
    assert systems.ISSUE527_RESULT_SHA256 == ISSUE527_RESULT_SHA256
    assert systems.ISSUE527_ORACLE_BLOB == ISSUE527_ORACLE_BLOB
    assert systems.ISSUE527_PROBE_BLOB == ISSUE527_PROBE_BLOB
    assert protocol["issue527_decision"] == "PASS"
    assert protocol["issue527_direct_pass"] == [256, 256]
    assert protocol["issue527_edge_pass"] == [32, 32]
    assert protocol["issue527_public_pass"] == [6, 6]
    assert protocol["issue527_topology_pass"] == [4, 4]

    launcher = LAUNCHER.read_text()
    assert 'SYSTEMS_ADAPTER_PR = 533' in launcher
    assert 'SYSTEMS_ADAPTER_PR_HEAD = "af3ae4d721ccf218f2d5ebcd41458fd7ff5b8ad3"' in launcher
    assert "SYSTEMS_ADAPTER_CPU_RUN = 33682266234" in launcher
    assert "SYSTEMS_ADAPTER_CPU_JOB = 100421371756" in launcher
    assert 'SYSTEMS_ADAPTER_MERGE = SOURCE_MAIN' in launcher
    assert f'SOURCE_MAIN = "{SOURCE_MAIN}"' in launcher


def test_issue534_preserves_consumed_issue508_failure_without_rerun() -> None:
    protocol = systems.issue530_systems_protocol()
    assert systems.ISSUE508_TRIGGER == 510
    assert systems.ISSUE508_RUN == 33661498305
    assert systems.ISSUE508_JOB == 100352870198
    assert systems.ISSUE508_ATTEMPT == 1
    assert systems.ISSUE508_BOUND_MAIN == "1d475a199cfd2b14d5e94e5cffa29e05ac868ab1"
    assert systems.ISSUE508_FAILURE == "FICEM write state/value floating dtypes must match"
    assert protocol["issue508_authoritative_result_emitted"] is False
    assert protocol["issue508_integrated_failure"] == systems.ISSUE508_FAILURE

    workflow = WORKFLOW.read_text()
    assert 'issues/510" --jq' in workflow
    assert 'actions/runs/33661498305' in workflow
    assert 'actions/jobs/100352870198' in workflow
    assert '"failure"' in workflow


def test_issue534_reuses_entire_frozen_systems_decision_surface() -> None:
    protocol = systems.issue530_systems_protocol()
    base_protocol = base.systems_protocol()

    assert base.SYSTEM_BATCH_SIZES == (8, 64)
    assert base.SYSTEM_WARMUP_CALLS == 3
    assert base.SYSTEM_TIMED_CALLS_PER_ROUND == 20
    assert base.SYSTEM_ROUNDS == 5
    assert base.BATCH8_MIN_FULL_SPEED_RATIO == 0.25
    assert base.BATCH64_MIN_FULL_SPEED_RATIO == 1.25
    assert (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) == (1e-2, 1e-2)
    assert base.EXPECTED_STATE_BYTES == 77_760
    assert (
        base.EXPECTED_SELECTED_WRITES,
        base.EXPECTED_CANDIDATES,
        base.EXPECTED_VECTOR_UPDATES,
    ) == (16, 255, 1)
    assert base.SOURCE_CHECKPOINT_SEED == 8471
    assert base.CHECKPOINT_RELATIVE_DIR == "/vol/aera-real-language/v25-dev-seed8471"

    assert protocol["batch_sizes"] == [8, 64]
    assert protocol["warmup_calls"] == 3
    assert protocol["timed_calls_per_round"] == 20
    assert protocol["rounds"] == 5
    assert protocol["batch8_min_full_speed_ratio"] == 0.25
    assert protocol["batch64_min_full_speed_ratio"] == 1.25
    assert protocol["integrated_atol"] == 1e-2
    assert protocol["integrated_rtol"] == 1e-2
    assert protocol["persistent_state_bytes_per_session"] == 77_760
    assert protocol["production_write_geometry"] == [16, 255, 1]
    assert protocol["source_checkpoint_seed"] == 8471
    assert protocol["checkpoint_relative_dir"] == base.CHECKPOINT_RELATIVE_DIR
    assert protocol["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert protocol["random_token_seed_rule"] == base_protocol["random_token_seed_rule"]
    assert protocol["timing_order"] == "rotated interleaved conditions per issue381"
    assert protocol["timing_clock"] == "CUDA events with synchronize before/after"
    assert protocol["hard"] is True
    assert protocol["route_mode"] == "hard_sparse"
    assert protocol["physically_real_sparse_required"] is True
    assert protocol["dense_masked_sparse_credit"] is False


def test_issue534_exact_evaluator_keeps_all_decision_bearing_checks() -> None:
    source = inspect.getsource(systems.run_end_to_end_systems_v26_6)
    required = (
        "triage.DIAGNOSTIC_SEED + 10_000 + batch_size",
        "base.SYSTEM_BATCH_SIZES",
        "base._timed_summaries(calls, batch_size=batch_size)",
        "base._route_signature",
        "triage._routing_accounting",
        "base._logit_equivalence",
        "base._state_equivalence",
        "base._physical_sparse_proof",
        "base._write_geometry",
        "base._finite_output",
        "base._episodic_state_bytes_per_session",
        "base._threshold_for_batch",
        "candidate_ms <= reference_ms",
        "row[\"throughput_pass\"]",
        "row[\"no_reference_full_latency_regression\"]",
        "base._parameter_versions",
        "base.checkpoint_hashes",
        "with torch.inference_mode():",
    )
    for fragment in required:
        assert fragment in source

    assert "@torch.inference_mode" not in source
    assert '"v26_6_triton_full_ficem"' in source
    assert '"PASS_E2E_SYSTEMS" if overall_pass else "FAIL_FROZEN_E2E_SYSTEMS_GATE"' in source


def test_issue534_launcher_is_duplicate_safe_one_l4_and_durable_before_marker() -> None:
    source = LAUNCHER.read_text()

    assert f"RESEARCH_ISSUE = {RESEARCH_ISSUE}" in source
    assert f'SOURCE_MAIN = "{SOURCE_MAIN}"' in source
    assert 'APP_NAME = "tam-research-aera-v26-6-issue534-end-to-end-systems"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue534-end-to-end-systems/result.json"' in source
    assert 'PRIMITIVE_RESULT_PATH = "/vol/aera-v26/issue527-ficem-write-repaired-oracle/result.json"' in source
    assert "MAX_GPU_SECONDS = 600" in source
    assert source.count('gpu="L4"') == 1
    assert "timeout=MAX_GPU_SECONDS" in source
    assert 'create_if_missing=False' in source
    assert source.count("run_end_to_end_systems_v26_6()") == 1
    assert "refusing duplicate issue534" in source
    assert "volume.reload()" in source
    assert "primitive_sha != ISSUE527_RESULT_SHA256" in source

    run_gate_pos = source.index("def run_gate()")
    durable_pos = source.index("result_path.write_text(durable_json)", run_gate_pos)
    commit_pos = source.index("volume.commit()", durable_pos)
    digest_pos = source.index("digest = hashlib.sha256", commit_pos)
    marker_pos = source.index("print(RESULT_MARKER", digest_pos)
    assert durable_pos < commit_pos < digest_pos < marker_pos

    local_pos = source.index("def main()")
    preflight_pos = source.index("check = preflight.remote()", local_pos)
    gpu_pos = source.index("result = run_gate.remote()", local_pos)
    assert preflight_pos < gpu_pos

    assert "modal.deploy" not in source
    assert "retry" not in source.lower()
    assert '"scientific_seed_consumed": False' in source
    assert '"breakthrough_proven": False' in source


def test_issue534_workflow_is_owner_only_attempt1_exact_bound_and_single_dispatch() -> None:
    source = WORKFLOW.read_text()

    assert "issues:" in source
    assert "types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "startsWith(github.event.issue.title, '[aera-v26-6-issue534-e2e-systems-l4]')" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "trigger_count=" in source
    assert 'test "${trigger_count}" = "1"' in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert SOURCE_MAIN in source
    assert SYSTEMS_ADAPTER_BLOB in source
    assert LAUNCHER_BLOB in source
    assert "33682266234" in source
    assert "100421371756" in source
    assert "33680028132" in source
    assert "100414089065" in source
    assert "33661498305" in source
    assert "100352870198" in source
    assert source.count("modal run modal_aera_v26_6_issue534_end_to_end_systems_app.py") == 1
    assert "modal deploy" not in source
    assert "/rerun" not in source
    assert "gh run rerun" not in source
    assert "workflow_dispatch" not in source
    assert "AERA_V26_6_ISSUE534_END_TO_END_SYSTEMS_RESULT_JSON=" in source
    assert "AERA_V26_6_ISSUE534_END_TO_END_SYSTEMS_SUMMARY_JSON=" in source


def test_issue534_cpu_preflight_and_protocol_authorize_no_higher_claims() -> None:
    contract = systems.cpu_contract_preflight_issue530()
    protocol = systems.issue530_systems_protocol()

    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_construction_performed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False

    assert protocol["systems_gpu_authorized_by_issue530"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["independent_replication_credit"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False
