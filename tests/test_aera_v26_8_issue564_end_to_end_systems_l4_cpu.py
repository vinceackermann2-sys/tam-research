from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_8_issue562_end_to_end_systems as adapter
from tam_research.aera_hardware_core_v26 import TorchFICEMReferenceBackend
from tam_research.aera_hardware_core_v26_8_ficem_read_mixed_strength_precision import (
    StrengthPrecisionTritonFICEMReadWriteBackend,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_8_issue564_end_to_end_systems_l4_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-8-issue564-e2e-systems-l4.yml"
ADAPTER = ROOT / "tam_research/aera_v26_8_issue562_end_to_end_systems.py"
ADAPTER_CPU = ROOT / "tests/test_aera_v26_8_issue562_end_to_end_systems_cpu.py"
ISSUE530 = ROOT / "tam_research/aera_v26_6_issue530_end_to_end_systems.py"
V26_8 = ROOT / "tam_research/aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py"

SOURCE_MAIN = "4277ae6e1f8267be3256c1c49c41835f78fe3147"
SOURCE_TREE = "56b61b639fd2ce3616b672d74b4fdf0f7736e278"
LAUNCHER_BLOB = "3b6268a905b4fd2707d1deaf5950c7a89682b4bd"
WORKFLOW_BLOB = "376e693b2116c027d29211374928a8379667fa87"
ADAPTER_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ADAPTER_CPU_BLOB = "2893a86cbdf767cdfa92601503d107d5ca3912fb"
ISSUE530_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
V26_8_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue564_exact_three_file_harness_and_frozen_blobs() -> None:
    assert _git_blob(LAUNCHER) == LAUNCHER_BLOB
    assert _git_blob(WORKFLOW) == WORKFLOW_BLOB
    assert _git_blob(ADAPTER) == ADAPTER_BLOB
    assert _git_blob(ADAPTER_CPU) == ADAPTER_CPU_BLOB
    assert _git_blob(ISSUE530) == ISSUE530_BLOB
    assert _git_blob(V26_8) == V26_8_BLOB

    issue564_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*issue564*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert issue564_paths == sorted(
        [
            ".github/workflows/aera-v26-8-issue564-e2e-systems-l4.yml",
            "modal_aera_v26_8_issue564_end_to_end_systems_l4_app.py",
            "tests/test_aera_v26_8_issue564_end_to_end_systems_l4_cpu.py",
        ]
    )


def test_issue564_preserves_frozen_systems_surface() -> None:
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
    assert base.CHECKPOINT_RELATIVE_DIR == "/vol/aera-real-language/v25-dev-seed8471"
    assert base.SOURCE_CHECKPOINT_SEED == 8471

    protocol = adapter.issue562_systems_protocol()
    assert protocol["batch_sizes"] == [8, 64]
    assert protocol["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert protocol["timing_order"] == "rotated interleaved conditions per issue381"
    assert protocol["timing_clock"] == "CUDA events with synchronize before/after"
    assert protocol["hard"] is True
    assert protocol["route_mode"] == "hard_sparse"
    assert protocol["physically_real_sparse_required"] is True
    assert protocol["dense_masked_sparse_credit"] is False
    assert protocol["persistent_state_bytes_per_session"] == 77_760
    assert protocol["production_write_geometry"] == [16, 255, 1]
    assert protocol["candidate_backend"] == StrengthPrecisionTritonFICEMReadWriteBackend.name
    assert protocol["reference_backend"] == TorchFICEMReferenceBackend.name
    assert protocol["frozen_issue530_run_function_reused"] is True
    assert protocol["frozen_issue530_loader_replaced_before_parameter_snapshot"] is True
    assert protocol["frozen_issue530_loader_replaced_before_any_model_call"] is True
    assert protocol["candidate_v26_8_read_mixed_strength_precision"] is True
    assert protocol["candidate_v26_6_write_inherited"] is True
    assert protocol["candidate_write_backend_changed_by_v26_8"] is False
    assert protocol["candidate_training_backend_changed_by_v26_8"] is False


def test_issue564_freezes_authoritative_predecessor_evidence() -> None:
    protocol = adapter.issue562_systems_protocol()
    assert protocol["issue558_trigger"] == 561
    assert protocol["issue558_run"] == 33733085825
    assert protocol["issue558_job"] == 100577290103
    assert protocol["issue558_result_sha256"] == (
        "e1fdc7e6b69a33084ca4b419b5489e755d7a98b12c367775ef19d1127700aa7e"
    )
    assert protocol["issue558_decision"] == "PASS"
    assert protocol["issue558_overall_pass"] is True
    assert protocol["issue558_mixed_rows_pass"] == [8, 8]
    assert protocol["issue545_trigger"] == 550
    assert protocol["issue545_run"] == 33686037672
    assert protocol["issue545_job"] == 100433658768
    assert protocol["issue545_authoritative_result_emitted"] is False
    assert protocol["issue545_failure"] == "FICEM read-tail floating dtypes must match"
    assert protocol["issue553_decision"] == "FAIL"
    assert protocol["issue529_result_sha256"] == (
        "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
    )


def test_issue564_launcher_one_cpu_preflight_one_l4_one_evaluator_call() -> None:
    source = LAUNCHER.read_text()
    assert f'SOURCE_MAIN = "{SOURCE_MAIN}"' in source
    assert f'SOURCE_TREE = "{SOURCE_TREE}"' in source
    assert 'APP_NAME = "tam-research-aera-v26-8-issue564-end-to-end-systems"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue564-end-to-end-systems/result.json"' in source
    assert 'CHECKPOINT_HASH_KEYS = frozenset({"aera", "transformer"})' in source
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 600" in source
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("adapter.cpu_contract_preflight_issue562()") == 1
    assert source.count("result = run_end_to_end_systems_v26_8()") == 1
    assert source.count("volume.commit()") == 1
    assert source.index("volume.commit()") < source.index("print(RESULT_MARKER +")
    assert source.count("result_path.exists()") == 2
    assert "refusing duplicate issue564 systems run" in source
    assert source.index("check = preflight.remote()") < source.index("result = run_gate.remote()")
    assert 'PRECHECK_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_PRECHECK_JSON="' in source
    assert 'PREFLIGHT_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_PREFLIGHT_JSON="' in source
    assert 'L4_START_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_L4_START_JSON="' in source
    assert 'RESULT_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_RESULT_JSON="' in source
    assert 'SUMMARY_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_SUMMARY_JSON="' in source
    assert "modal.deploy" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source


def test_issue564_workflow_is_canonical_lowest_attempt1_without_concurrency_or_retry() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "[aera-v26-8-issue564-e2e-systems-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "\nconcurrency:" not in source
    assert "trigger_count=" not in source
    assert source.count("mapfile -t matching_triggers") == 2
    assert 'canonical_trigger="${matching_triggers[0]}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${canonical_trigger}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${matching_triggers[0]}"' in source
    assert source.count('test "${#matching_triggers[@]}" -ge 1') == 2
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert "33734411109" in source
    assert "100581497823" in source
    assert "33733085825" in source
    assert "100577290103" in source
    assert "33686037672" in source
    assert "100433658768" in source
    assert f'git hash-object modal_aera_v26_8_issue564_end_to_end_systems_l4_app.py)" = "{LAUNCHER_BLOB}"' in source
    assert source.count("modal run modal_aera_v26_8_issue564_end_to_end_systems_l4_app.py") == 1
    assert "modal deploy" not in source
    assert "gh run rerun" not in source.lower()
    assert "redispatch" not in source.lower()


def test_issue564_cpu_contract_and_protocol_keep_higher_authorizations_false() -> None:
    contract = adapter.cpu_contract_preflight_issue562()
    protocol = adapter.issue562_systems_protocol()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_construction_performed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["systems_measurement_performed"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False
    assert protocol["systems_gpu_authorized_by_issue562"] is False
    assert protocol["end_to_end_systems_executed_by_issue562"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["independent_replication_credit"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False
