from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_8_issue569_end_to_end_systems_memory_safe as memory_safe
from tam_research import aera_v26_9_issue641_physical_sparse_backend_identity_compat as issue641
from tam_research import aera_v26_9_issue643_bounded_memory_end_to_end_systems as systems
from tam_research.aera_hardware_core_v26 import TorchFICEMReferenceBackend
from tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility import (
    IdentityWeightVisibilityTritonFICEMReadWriteBackend,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py"
LAUNCHER = ROOT / "modal_aera_v26_9_issue643_bounded_memory_end_to_end_systems_l4_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-9-issue643-bounded-memory-end-to-end-systems-l4.yml"
CPU_TEST = ROOT / "tests/test_aera_v26_9_issue643_bounded_memory_end_to_end_systems_l4_cpu.py"

BASE_SYSTEMS = ROOT / "tam_research/aera_v26_5_end_to_end_systems.py"
REPAIR1_SYSTEMS = ROOT / "tam_research/aera_v26_5_end_to_end_systems_repair1.py"
ISSUE530 = ROOT / "tam_research/aera_v26_6_issue530_end_to_end_systems.py"
ISSUE562 = ROOT / "tam_research/aera_v26_8_issue562_end_to_end_systems.py"
ISSUE569 = ROOT / "tam_research/aera_v26_8_issue569_end_to_end_systems_memory_safe.py"
V26_9 = ROOT / "tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py"
ISSUE625 = ROOT / "tam_research/aera_v26_9_issue625_schema_guard_repair1.py"
ISSUE641 = ROOT / "tam_research/aera_v26_9_issue641_physical_sparse_backend_identity_compat.py"
ISSUE641_CPU = ROOT / "tests/test_aera_v26_9_issue641_physical_sparse_backend_identity_compat_cpu.py"

SOURCE_MAIN = "ef93e787e6d01585307f05f046d7fd3806374511"
SOURCE_TREE = "a44bcdb61b3124494e58902cad3d233cf7926cff"
ADAPTER_BLOB = "512572340cc09e2e7ad6729712258c12cb377ef2"
LAUNCHER_BLOB = "aee3916cb87d48296e63f36d77d6541e0cf4d474"
WORKFLOW_BLOB = "3cbc65ae5a53eb1b518481f8ed119ec974c59fbc"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
ISSUE530_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
ISSUE562_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ISSUE569_BLOB = "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203"
V26_9_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
ISSUE625_BLOB = "92d06a4954bca1b302355e81f5bf09b06fcee222"
ISSUE641_BLOB = "5ea1919d15904add0f9e0fb714757f32b11442cb"
ISSUE641_CPU_BLOB = "e620a9874958bda78d586269f597095f5cf70670"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue643_exact_four_file_harness_and_frozen_lineage() -> None:
    expected = {
        ADAPTER: ADAPTER_BLOB,
        LAUNCHER: LAUNCHER_BLOB,
        WORKFLOW: WORKFLOW_BLOB,
        BASE_SYSTEMS: BASE_SYSTEMS_BLOB,
        REPAIR1_SYSTEMS: REPAIR1_SYSTEMS_BLOB,
        ISSUE530: ISSUE530_BLOB,
        ISSUE562: ISSUE562_BLOB,
        ISSUE569: ISSUE569_BLOB,
        V26_9: V26_9_BLOB,
        ISSUE625: ISSUE625_BLOB,
        ISSUE641: ISSUE641_BLOB,
        ISSUE641_CPU: ISSUE641_CPU_BLOB,
    }
    assert all(path.exists() for path in expected)
    for path, blob in expected.items():
        assert _git_blob(path) == blob

    issue643_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*issue643*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    assert issue643_paths == sorted(
        [
            ".github/workflows/aera-v26-9-issue643-bounded-memory-end-to-end-systems-l4.yml",
            "modal_aera_v26_9_issue643_bounded_memory_end_to_end_systems_l4_app.py",
            "tam_research/aera_v26_9_issue643_bounded_memory_end_to_end_systems.py",
            "tests/test_aera_v26_9_issue643_bounded_memory_end_to_end_systems_l4_cpu.py",
        ]
    )


def test_issue643_preserves_exact_frozen_systems_surface() -> None:
    assert systems.SOURCE_MAIN == SOURCE_MAIN
    assert systems.SOURCE_TREE == SOURCE_TREE
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

    inherited = memory_safe.issue569_systems_protocol()
    assert inherited["batch_sizes"] == [8, 64]
    assert inherited["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert inherited["timing_order"] == "rotated interleaved conditions per issue381"
    assert inherited["timing_clock"] == "CUDA events with synchronize before/after"
    assert inherited["hard"] is True
    assert inherited["route_mode"] == "hard_sparse"
    assert inherited["physically_real_sparse_required"] is True
    assert inherited["dense_masked_sparse_credit"] is False
    assert inherited["persistent_state_bytes_per_session"] == 77_760
    assert inherited["production_write_geometry"] == [16, 255, 1]
    assert inherited["chunk_batch_rows"] == 1
    assert inherited["all_elements_covered"] is True
    assert inherited["sampling_or_approximation"] is False
    assert inherited["result_dependent_chunk_sizing"] is False


def test_issue643_candidate_protocol_changes_only_versioned_candidate_identity() -> None:
    protocol = systems.issue643_candidate_protocol()
    assert protocol["research_issue"] == 643
    assert protocol["candidate_backend"] == IdentityWeightVisibilityTritonFICEMReadWriteBackend.name
    assert protocol["reference_backend"] == TorchFICEMReferenceBackend.name
    assert protocol["v26_9_candidate_blob"] == V26_9_BLOB
    assert protocol["issue569_systems_blob"] == ISSUE569_BLOB
    assert protocol["issue641_adapter_blob"] == ISSUE641_BLOB
    assert protocol["only_candidate_semantic_change_issue643"] == (
        "execution_backend_v26_8_to_v26_9_identity_weight_visibility"
    )
    assert protocol["frozen_issue530_run_function_reused"] is True
    assert protocol["frozen_issue530_loader_replaced_before_parameter_snapshot"] is True
    assert protocol["frozen_issue530_loader_replaced_before_any_model_call"] is True
    assert protocol["frozen_issue569_memory_safe_reductions_reused"] is True
    assert protocol["issue641_physical_sparse_identity_adapter_scoped"] is True
    assert protocol["candidate_v26_6_write_inherited"] is True
    assert protocol["candidate_write_backend_changed_by_v26_9"] is False
    assert protocol["candidate_training_backend_changed_by_v26_9"] is False


def test_issue643_freezes_historical_fail_and_primitive_pass_authority() -> None:
    assert systems.ISSUE571_TRIGGER == 573
    assert systems.ISSUE571_RUN == 33741700781
    assert systems.ISSUE571_JOB == 100604889696
    assert systems.ISSUE571_RESULT_PATH == (
        "/vol/aera-v26/issue571-memory-safe-end-to-end-systems/result.json"
    )
    assert systems.ISSUE571_RESULT_SHA256 == (
        "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
    )
    assert systems.ISSUE571_DECISION == "FAIL_FROZEN_E2E_SYSTEMS_GATE"

    assert systems.ISSUE630_TRIGGER == 640
    assert systems.ISSUE630_RUN == 33985543569
    assert systems.ISSUE630_JOB == 101358253857
    assert systems.ISSUE630_RESULT_SHA256 == (
        "ef92c85b55484b3ce191cd4016be86bf52da997a153f737194976164b29554b4"
    )
    assert systems.ISSUE630_DECISION == "PASS"
    assert systems.ISSUE630_DESIGN_SEED == 891475817
    assert issue641.ISSUE630_RESULT_SHA256 == systems.ISSUE630_RESULT_SHA256
    assert issue641.ISSUE571_RESULT_SHA256 == systems.ISSUE571_RESULT_SHA256


def test_issue643_cpu_preflight_executes_no_systems_or_checkpoint_work() -> None:
    contract = systems.cpu_contract_preflight_issue643()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["model_construction_performed"] is False
    assert contract["checkpoint_loaded"] is False
    assert contract["systems_measurement_performed"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["s2_authorized"] is False
    assert contract["fresh_scientific_seed_authorized"] is False
    assert contract["independent_replication_credit"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False
    assert contract["checkpoint_path"] == "/vol/aera-real-language/v25-dev-seed8471"
    assert contract["checkpoint_hashes_expected"] == systems.CHECKPOINT_HASHES


def test_issue643_adapter_scopes_exact_loader_protocol_and_issue641_wrapper() -> None:
    source = ADAPTER.read_text()

    assert source.count(
        "issue562.load_models_v26_8 = load_models_v26_9"
    ) == 1
    assert source.count(
        "issue562.issue562_systems_protocol = issue643_candidate_protocol"
    ) == 1
    assert source.count(
        "issue562.load_models_v26_8 = _FROZEN_ISSUE562_LOADER"
    ) == 1
    assert source.count(
        "issue562.issue562_systems_protocol = _FROZEN_ISSUE562_PROTOCOL"
    ) == 1
    assert source.count("_FROZEN_ISSUE641_WRAPPER(") == 1
    assert source.count("_FROZEN_ISSUE569_RUN(run_dir=run_dir)") == 1

    install = source.index("issue562.load_models_v26_8 = load_models_v26_9")
    wrapper = source.index("_FROZEN_ISSUE641_WRAPPER(")
    frozen_run = source.index("_FROZEN_ISSUE569_RUN(run_dir=run_dir)")
    restore_loader = source.index(
        "issue562.load_models_v26_8 = _FROZEN_ISSUE562_LOADER"
    )
    restore_protocol = source.index(
        "issue562.issue562_systems_protocol = _FROZEN_ISSUE562_PROTOCOL"
    )
    assert install < wrapper < frozen_run < restore_loader
    assert frozen_run < restore_protocol

    load_call = source.index("reference, candidate, transformer = frozen530.base.load_models(")
    backend_install = source.index(
        "candidate_backend_names = _install_v26_9_candidate_backend(candidate)"
    )
    assert load_call < backend_install
    assert "memory._execution_backend = IdentityWeightVisibilityTritonFICEMReadWriteBackend()" in source
    assert "TorchFICEMReferenceBackend.name" in source
    assert "substring" not in source.lower()
    assert "startswith(" not in source


def test_issue643_launcher_is_one_preflight_one_l4_one_systems_run() -> None:
    source = LAUNCHER.read_text()
    assert f'SOURCE_MAIN = "{SOURCE_MAIN}"' in source
    assert f'SOURCE_TREE = "{SOURCE_TREE}"' in source
    assert f'ISSUE643_ADAPTER_BLOB = "{ADAPTER_BLOB}"' in source
    assert 'APP_NAME = "aera-v26-9-issue643-bounded-memory-e2e-systems-l4"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert (
        'RESULT_PATH = "/vol/aera-v26/issue643-bounded-memory-end-to-end-systems/result.json"'
        in source
    )
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 600" in source
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("contract = systems.cpu_contract_preflight_issue643()") == 1
    assert source.count("result = run_end_to_end_systems_v26_9_bounded_memory()") == 1
    assert source.count("volume.commit()") == 1
    assert source.index("volume.commit()") < source.index("print(RESULT_MARKER +")
    assert source.count("result_path.exists()") == 2
    assert "refusing duplicate issue643 bounded-memory systems run" in source
    assert source.index("check = preflight.remote()") < source.index("result = run_gate.remote()")
    assert '_sha256_file(historical571) != ISSUE571_RESULT_SHA256' in source
    assert '_sha256_file(primitive630) != ISSUE630_RESULT_SHA256' in source
    assert "hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)" in source
    assert 'device != "NVIDIA L4"' in source
    assert (
        'PRECHECK_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_PRECHECK_JSON="'
        in source
    )
    assert (
        'PREFLIGHT_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_PREFLIGHT_JSON="'
        in source
    )
    assert (
        'L4_START_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_L4_START_JSON="'
        in source
    )
    assert (
        'RESULT_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_RESULT_JSON="'
        in source
    )
    assert (
        'SUMMARY_MARKER = "AERA_V26_9_ISSUE643_BOUNDED_MEMORY_END_TO_END_SYSTEMS_SUMMARY_JSON="'
        in source
    )


def test_issue643_workflow_is_canonical_lowest_attempt1_and_post_merge_authorized() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "[aera-v26-9-issue643-bounded-memory-e2e-systems-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "\nconcurrency:" not in source
    assert source.count("mapfile -t matching_triggers") == 2
    assert 'canonical_trigger="${matching_triggers[0]}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${canonical_trigger}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${matching_triggers[0]}"' in source
    assert source.count('test "${#matching_triggers[@]}" -ge 1') == 2
    assert "Bind main:" in source
    assert 'git checkout --detach "${bound_main}"' in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert "## #643 sole L4 bounded-memory E2E authorization" in source
    assert "Authorize main:" in source

    assert "504b5b3fdd254645eeb31bcb059831a8a6ee3164" in source
    assert "33989770634" in source
    assert "101369805445" in source
    assert "33989555924" in source
    assert "101369221062" in source
    assert "33985543569" in source
    assert "101358253857" in source
    assert "33741700781" in source
    assert "100604889696" in source
    assert "ef92c85b55484b3ce191cd4016be86bf52da997a153f737194976164b29554b4" in source
    assert "FAIL_FROZEN_E2E_SYSTEMS_GATE" in source
    assert f'" = "{ADAPTER_BLOB}"' in source
    assert f'" = "{LAUNCHER_BLOB}"' in source
    assert source.count(
        "modal run modal_aera_v26_9_issue643_bounded_memory_end_to_end_systems_l4_app.py"
    ) == 1
    assert "modal deploy" not in source
    assert "gh run rerun" not in source.lower()


def test_issue643_all_higher_authorizations_remain_false() -> None:
    protocol = systems.issue643_candidate_protocol()
    for key in (
        "systems_gpu_authorized_by_issue643",
        "end_to_end_systems_executed_by_issue643",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue643_static_surface_contains_no_scientific_or_retry_path() -> None:
    adapter = ADAPTER.read_text()
    launcher = LAUNCHER.read_text()
    workflow = WORKFLOW.read_text()
    combined = adapter + "\n" + launcher + "\n" + workflow
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

    assert "chunk_batch_rows" in adapter
    assert "chunk_batch_rows" in launcher
    assert "dense_masked_sparse_credit" in adapter
    assert "issue630_design_seed_consumed" in adapter
    assert CPU_TEST.exists()
