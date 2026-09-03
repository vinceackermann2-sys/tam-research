from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_8_issue569_end_to_end_systems_memory_safe as memory_safe
from tam_research.aera_hardware_core_v26 import TorchFICEMReferenceBackend
from tam_research.aera_hardware_core_v26_8_ficem_read_mixed_strength_precision import (
    StrengthPrecisionTritonFICEMReadWriteBackend,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_8_issue571_memory_safe_end_to_end_systems_l4_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-8-issue571-memory-safe-e2e-systems-l4.yml"
CPU_TEST = ROOT / "tests/test_aera_v26_8_issue571_memory_safe_end_to_end_systems_l4_cpu.py"
MEMORY_SAFE = ROOT / "tam_research/aera_v26_8_issue569_end_to_end_systems_memory_safe.py"
MEMORY_SAFE_CPU = ROOT / "tests/test_aera_v26_8_issue569_end_to_end_systems_memory_safe_cpu.py"
ISSUE562 = ROOT / "tam_research/aera_v26_8_issue562_end_to_end_systems.py"
ISSUE562_CPU = ROOT / "tests/test_aera_v26_8_issue562_end_to_end_systems_cpu.py"
ISSUE530 = ROOT / "tam_research/aera_v26_6_issue530_end_to_end_systems.py"
BASE_SYSTEMS = ROOT / "tam_research/aera_v26_5_end_to_end_systems.py"
REPAIR1_SYSTEMS = ROOT / "tam_research/aera_v26_5_end_to_end_systems_repair1.py"
V26_8 = ROOT / "tam_research/aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py"

SOURCE_MAIN = "f9d10b71e2d4b8035bbd954d710d48febd616b48"
SOURCE_TREE = "4b23e3be4f7f2c8a607eba13fa24c526c9683187"
LAUNCHER_BLOB = "d0e88d2ebab5a2df340325b2951ed1517c6945b7"
WORKFLOW_BLOB = "7077f54353bd5b57dd33d47e55b5d65edc664290"
MEMORY_SAFE_BLOB = "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203"
MEMORY_SAFE_CPU_BLOB = "a6b3fc81239b4e7b4c6d2185f9d21465384876a0"
ISSUE562_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ISSUE562_CPU_BLOB = "2893a86cbdf767cdfa92601503d107d5ca3912fb"
ISSUE530_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
V26_8_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue571_exact_three_file_harness_and_frozen_lineage() -> None:
    expected = {
        LAUNCHER: LAUNCHER_BLOB,
        WORKFLOW: WORKFLOW_BLOB,
        MEMORY_SAFE: MEMORY_SAFE_BLOB,
        MEMORY_SAFE_CPU: MEMORY_SAFE_CPU_BLOB,
        ISSUE562: ISSUE562_BLOB,
        ISSUE562_CPU: ISSUE562_CPU_BLOB,
        ISSUE530: ISSUE530_BLOB,
        BASE_SYSTEMS: BASE_SYSTEMS_BLOB,
        REPAIR1_SYSTEMS: REPAIR1_SYSTEMS_BLOB,
        V26_8: V26_8_BLOB,
    }
    assert all(path.exists() for path in expected)
    for path, blob in expected.items():
        assert _git_blob(path) == blob

    issue571_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*issue571*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    assert issue571_paths == sorted(
        [
            ".github/workflows/aera-v26-8-issue571-memory-safe-e2e-systems-l4.yml",
            "modal_aera_v26_8_issue571_memory_safe_end_to_end_systems_l4_app.py",
            "tests/test_aera_v26_8_issue571_memory_safe_end_to_end_systems_l4_cpu.py",
        ]
    )


def test_issue571_preserves_exact_memory_safe_systems_surface() -> None:
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

    protocol = memory_safe.issue569_systems_protocol()
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
    assert protocol["chunk_batch_rows"] == 1
    assert protocol["all_elements_covered"] is True
    assert protocol["sampling_or_approximation"] is False
    assert protocol["result_dependent_chunk_sizing"] is False


def test_issue571_freezes_566_consumed_and_567_inert_evidence() -> None:
    protocol = memory_safe.issue569_systems_protocol()
    assert protocol["issue566_trigger"] == 566
    assert protocol["issue566_bound_main"] == "6e9471ca86fed0438bd25dd314040a16e637f2be"
    assert protocol["issue566_run"] == 33737873193
    assert protocol["issue566_job"] == 100592625418
    assert protocol["issue566_attempt"] == 1
    assert protocol["issue566_authoritative_result_emitted"] is False
    assert protocol["issue566_l4_started"] is True
    assert protocol["issue566_failure_type"] == "torch.OutOfMemoryError"
    assert protocol["issue566_failure"] == "CUDA out of memory. Tried to allocate 6.14 GiB."
    assert protocol["issue566_failure_site"].endswith("::_logit_equivalence")

    assert protocol["issue567_trigger"] == 567
    assert protocol["issue567_run"] == 33737887818
    assert protocol["issue567_job"] == 100592679436
    assert protocol["issue567_canonical"] is False
    assert protocol["issue567_gpu_started"] is False

    assert protocol["checkpoint_hashes_issue566"] == {
        "aera": "f8aa92421801e8f190247e420632be5f0c20bc5ea8bf6bdeefe06686b3a31b30",
        "transformer": "cdd5cab4439a709468d6607d45d82081b33e876b2e40d91d4a38ba139b219dd7",
    }


def test_issue571_launcher_is_one_preflight_one_l4_one_memory_safe_run() -> None:
    source = LAUNCHER.read_text()
    assert f'SOURCE_MAIN = "{SOURCE_MAIN}"' in source
    assert f'SOURCE_TREE = "{SOURCE_TREE}"' in source
    assert 'APP_NAME = "aera-v26-8-issue571-memory-safe-e2e-systems-l4"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert (
        'RESULT_PATH = "/vol/aera-v26/issue571-memory-safe-end-to-end-systems/result.json"'
        in source
    )
    assert 'CHECKPOINT_HASH_KEYS = frozenset(CHECKPOINT_HASHES)' in source
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 600" in source
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("memory_safe.cpu_contract_preflight_issue569()") == 1
    assert source.count("result = run_end_to_end_systems_v26_8_memory_safe()") == 1
    assert source.count("volume.commit()") == 1
    assert source.index("volume.commit()") < source.index("print(RESULT_MARKER +")
    assert source.count("result_path.exists()") == 2
    assert "refusing duplicate issue571 memory-safe systems run" in source
    assert source.index("check = preflight.remote()") < source.index("result = run_gate.remote()")
    assert (
        'PRECHECK_MARKER = "AERA_V26_8_ISSUE571_MEMORY_SAFE_END_TO_END_SYSTEMS_PRECHECK_JSON="'
        in source
    )
    assert (
        'PREFLIGHT_MARKER = "AERA_V26_8_ISSUE571_MEMORY_SAFE_END_TO_END_SYSTEMS_PREFLIGHT_JSON="'
        in source
    )
    assert (
        'L4_START_MARKER = "AERA_V26_8_ISSUE571_MEMORY_SAFE_END_TO_END_SYSTEMS_L4_START_JSON="'
        in source
    )
    assert (
        'RESULT_MARKER = "AERA_V26_8_ISSUE571_MEMORY_SAFE_END_TO_END_SYSTEMS_RESULT_JSON="'
        in source
    )
    assert (
        'SUMMARY_MARKER = "AERA_V26_8_ISSUE571_MEMORY_SAFE_END_TO_END_SYSTEMS_SUMMARY_JSON="'
        in source
    )
    assert "modal.deploy" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "workflow_dispatch" not in source


def test_issue571_workflow_is_canonical_lowest_attempt1_and_future_merge_safe() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "[aera-v26-8-issue571-memory-safe-e2e-systems-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "\nconcurrency:" not in source
    assert "trigger_count=" not in source
    assert source.count("mapfile -t matching_triggers") == 2
    assert 'canonical_trigger="${matching_triggers[0]}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${canonical_trigger}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${matching_triggers[0]}"' in source
    assert source.count('test "${#matching_triggers[@]}" -ge 1') == 2
    assert "Bind main:" in source
    assert 'git checkout --detach "${bound_main}"' in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert "HEAD^{tree}" not in source

    assert "06cb135a6ec92f59ffbd1a2a215da8367092c079" in source
    assert "33740176638" in source
    assert "100599997754" in source
    assert "33737873193" in source
    assert "100592625418" in source
    assert "33737887818" in source
    assert "100592679436" in source
    assert "33733085825" in source
    assert "100577290103" in source
    assert "33686037672" in source
    assert "100433658768" in source
    assert "issue566_result_count" in source
    assert 'test "${issue566_result_count}" = "0"' in source
    assert "issue566_failure_count" in source
    assert 'test "${issue566_failure_count}" = "1"' in source
    assert 'select(.name == "Authenticate Modal") | .conclusion' in source
    assert 'select(.name == "Run canonical issue564 end-to-end systems L4 gate") | .conclusion' in source
    assert (
        f'git hash-object modal_aera_v26_8_issue571_memory_safe_end_to_end_systems_l4_app.py)" = "{LAUNCHER_BLOB}"'
        in source
    )
    assert source.count(
        "modal run modal_aera_v26_8_issue571_memory_safe_end_to_end_systems_l4_app.py"
    ) == 1
    assert "modal deploy" not in source
    assert "gh run rerun" not in source.lower()
    assert "No rerun, retry, redispatch" in source


def test_issue571_cpu_contract_and_protocol_keep_all_higher_authorizations_false() -> None:
    contract = memory_safe.cpu_contract_preflight_issue569()
    protocol = memory_safe.issue569_systems_protocol()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_construction_performed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["systems_measurement_performed"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False
    assert protocol["gpu_authorized_by_issue569"] is False
    assert protocol["end_to_end_systems_executed_by_issue569"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["independent_replication_credit"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_issue571_static_harness_contains_no_scientific_or_retry_surface() -> None:
    launcher = LAUNCHER.read_text()
    workflow = WORKFLOW.read_text()
    combined = launcher + "\n" + workflow
    for forbidden in (
        "torch.optim",
        ".backward(",
        "optimizer.step",
        "workflow_dispatch",
        "modal deploy",
        "gh run rerun",
        "fresh scientific seed authorized: true",
    ):
        assert forbidden not in combined.lower()

    assert "chunk_batch_rows" in launcher
    assert "sampling_or_approximation" in launcher
    assert "result_dependent_chunk_sizing" in launcher
    assert "issue566_consumed_pre_result" in launcher
    assert "issue567_inert_pre_gpu" in launcher
    assert CPU_TEST.exists()
