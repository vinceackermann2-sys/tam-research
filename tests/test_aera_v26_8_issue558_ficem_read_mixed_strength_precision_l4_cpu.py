from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from tam_research import aera_v26_7_issue553_ficem_read_mixed_dtype_probe as frozen553
from tam_research import aera_v26_8_issue558_ficem_read_mixed_strength_precision_probe as probe


ROOT = Path(__file__).resolve().parents[1]
V26_8 = ROOT / "tam_research/aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py"
V26_8_CPU = ROOT / "tests/test_aera_v26_8_issue556_ficem_read_mixed_strength_precision_cpu.py"
V26_7 = ROOT / "tam_research/aera_hardware_core_v26_7_ficem_read_mixed_dtype.py"
REPAIR5 = ROOT / "tam_research/aera_hardware_core_v26_3_ficem_read_triton.py"
WRITE_V26_6 = ROOT / "tam_research/aera_hardware_core_v26_6_ficem_write_materialize_cast.py"
FROZEN_ISSUE553_PROBE = ROOT / "tam_research/aera_v26_7_issue553_ficem_read_mixed_dtype_probe.py"
FROZEN_ISSUE553_LAUNCHER = ROOT / "modal_aera_v26_7_issue553_ficem_read_mixed_dtype_l4_app.py"
FROZEN_ISSUE553_WORKFLOW = ROOT / ".github/workflows/aera-v26-7-issue553-ficem-read-mixed-dtype-l4.yml"
HISTORICAL_PROBE = ROOT / "tam_research/aera_v26_3_ficem_read_probe.py"
REPAIR5_PROBE = ROOT / "tam_research/aera_v26_3_ficem_read_probe_repair5.py"
SYSTEMS_530 = ROOT / "tam_research/aera_v26_6_issue530_end_to_end_systems.py"
V26_INTERFACE = ROOT / "tam_research/aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research/aera_hardware_core_v25_1_compact.py"
ISSUE558_PROBE = ROOT / "tam_research/aera_v26_8_issue558_ficem_read_mixed_strength_precision_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_8_issue558_ficem_read_mixed_strength_precision_l4_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-8-issue558-ficem-read-mixed-strength-precision-l4.yml"

SOURCE_MAIN = "ae25cb4133c1ff94bec1cdfa9aa58e4081c05c73"
SOURCE_TREE = "3b59aa070a98f873d728d2c30ab08156f73bec23"
EXPECTED_BLOBS = {
    V26_8: "3575c58d1cd730be77649f087908c51dbf3e6088",
    V26_8_CPU: "443d36dcc61eb72f8a2f406a6f2ae1abfeb365c4",
    V26_7: "d8133c6b204b1ee5f23955255fb2fb09d09bd723",
    REPAIR5: "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
    WRITE_V26_6: "d45c262314a0b4691f26812a279937a225043ad9",
    FROZEN_ISSUE553_PROBE: "ff9a47f510be07e8adeff018f327338147163cdb",
    FROZEN_ISSUE553_LAUNCHER: "b3630e065c56c93a1b7f6f164416f068ccb2ecac",
    FROZEN_ISSUE553_WORKFLOW: "eef7826f1a76a853d9cf745243612dd457d79a10",
    HISTORICAL_PROBE: "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b",
    REPAIR5_PROBE: "6fd6518e10ed1ef4115863f98ac591ffd77ce903",
    SYSTEMS_530: "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b",
    V26_INTERFACE: "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
    STABLE_REFERENCE: "4e336b6e1a6238dac782fa320751d68281493ee1",
    ISSUE558_PROBE: "99ab8252f2b594404aae1ca86752eaa902eb80a5",
    LAUNCHER: "634758b2293d982055521c2584771499a3b28149",
    WORKFLOW: "152343fb2f98ccbe6265fd9d520a205e87d61f75",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue558_frozen_blobs_and_source_boundary() -> None:
    for path, expected in EXPECTED_BLOBS.items():
        assert path.exists(), path
        assert _git_blob(path) == expected, path
    assert probe.RESEARCH_ISSUE == 558
    assert probe.SOURCE_MAIN == SOURCE_MAIN
    assert probe.SOURCE_TREE == SOURCE_TREE
    assert probe.V26_8_BACKEND_BLOB == EXPECTED_BLOBS[V26_8]
    assert probe.V26_8_CPU_TEST_BLOB == EXPECTED_BLOBS[V26_8_CPU]
    assert probe.FROZEN_ISSUE553_PROBE_BLOB == EXPECTED_BLOBS[FROZEN_ISSUE553_PROBE]


def test_issue558_cpu_preflight_preserves_exact_553_decision_surface() -> None:
    contract = probe.cpu_contract_preflight_issue558()
    protocol = contract["protocol"]
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["synthetic_only"] is True
    assert contract["scientific_seed_consumed"] is False

    assert probe.DESIGN_SEED == frozen553.DESIGN_SEED == 408_411
    assert (probe.D_MODEL, probe.TIME, probe.CAPACITY, probe.MEMORY_DIM) == (
        200,
        256,
        48,
        50,
    )
    assert probe.BATCH_SIZES == frozen553.BATCH_SIZES == (8, 64)
    assert probe.DTYPE_NAMES == frozen553.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.VALIDITY_KINDS == frozen553.VALIDITY_KINDS == ("mixed", "full")
    assert probe.MIXED_LAYOUTS == frozen553.MIXED_LAYOUTS == (
        ("bfloat16", "float32"),
        ("float32", "bfloat16"),
    )
    assert (probe.WARMUP_CALLS, probe.TIMED_ROUNDS, probe.CALLS_PER_ROUND) == (
        10,
        5,
        100,
    )
    assert (
        probe.FP32_ATOL,
        probe.FP32_RTOL,
        probe.BF16_ATOL,
        probe.BF16_RTOL,
    ) == (1e-5, 1e-5, 1e-2, 1e-2)
    assert probe.MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert probe.MAX_ROW_LATENCY_RATIO == 1.05
    assert probe.MAX_FULL_EVENT_RATIO == 0.75

    assert protocol["frozen_issue553_probe_logic_reused"] is True
    assert protocol["candidate_substitution_only"] is True
    assert protocol["mixed_tail_profiler_acceptance_changed"] is False
    assert protocol["mixed_tail_profiler_event_name_updated_only"] is True
    assert protocol["historical_surface_preserved"] is True
    assert protocol["historical_surface_candidate_is_v26_8"] is True
    assert protocol["historical_timing_decision_bearing"] is True
    assert protocol["mixed_timing_decision_bearing"] is False


def test_issue558_probe_is_a_narrow_runtime_substitution_not_fixture_fork() -> None:
    source = ISSUE558_PROBE.read_text()
    assert "def _make_mixed_case(" not in source
    assert "def _mixed_correctness(" not in source
    assert "def _mixed_near_tie(" not in source
    assert "def _mixed_known_empty(" not in source
    assert "def _historical_surface(" not in source
    assert "torch.Generator().manual_seed" not in source
    assert "frozen553.run_ficem_read_mixed_dtype_probe_v26_7()" in source

    run_source = inspect.getsource(probe.run_ficem_read_probe_v26_8_issue558)
    assert run_source.count("frozen553.run_ficem_read_mixed_dtype_probe_v26_7()") == 1
    assert "original_backend = frozen553.MixedDtypeTritonFICEMReadWriteBackend" in run_source
    assert "original_tail = frozen553.fused_ficem_read_tail_mixed_dtype" in run_source
    assert "original_profiler = frozen553._tail_profile_with_cast_accounting" in run_source
    assert "v26_8.StrengthPrecisionTritonFICEMReadWriteBackend" in run_source
    assert "v26_8.fused_ficem_read_tail_v26_8" in run_source
    assert "_tail_profile_with_v26_8_event_name" in run_source
    assert "finally:" in run_source
    assert "= original_backend" in run_source
    assert "= original_tail" in run_source
    assert "= original_profiler" in run_source


def test_issue558_profiler_changes_event_name_only_not_acceptance_fields() -> None:
    successor = inspect.getsource(probe._tail_profile_with_v26_8_event_name)
    predecessor = inspect.getsource(frozen553._tail_profile_with_cast_accounting)

    for token in (
        "torch.profiler.ProfilerActivity.CPU",
        "torch.profiler.ProfilerActivity.CUDA",
        '"topk": 0',
        '"softmax": 0',
        '"gather": 0',
        '"_to_copy": 0',
        '"copy_": 0',
        '"cuda_device_events"',
        '"triton_read_tail_events"',
        '"relevant_operator_calls"',
    ):
        assert token in predecessor
        assert token in successor
    assert "ficem_read_tail_kernel" in predecessor
    assert "mixed_strength_precision_kernel" in successor
    assert "ficem_read_tail_kernel" not in successor

    frozen_source = FROZEN_ISSUE553_PROBE.read_text()
    assert 'tail_profile["cuda_device_events"] == 1' in frozen_source
    assert 'tail_profile["triton_read_tail_events"] == 1' in frozen_source
    assert 'tail_profile["relevant_operator_calls"]["_to_copy"] == 0' in frozen_source
    assert 'tail_profile["relevant_operator_calls"]["copy_"] == 0' in frozen_source


def test_issue558_frozen_predecessor_and_candidate_evidence_are_explicit() -> None:
    protocol = probe.issue558_protocol()
    assert protocol["issue557_head"] == "783c5e2921d1d7fc7f598948d1bb968e51260440"
    assert protocol["issue557_cpu_run"] == 33730039451
    assert protocol["issue557_cpu_job"] == 100567632346
    assert protocol["issue557_merge"] == SOURCE_MAIN

    assert protocol["issue553_trigger"] == 555
    assert protocol["issue553_run"] == 33727540468
    assert protocol["issue553_job"] == 100559866985
    assert protocol["issue553_decision"] == "FAIL"
    assert protocol["issue553_result_sha256"] == (
        "009af31baf70e46eb93b6e7489d62f356a02b727521d3fabe4a7dab2dcf5ab47"
    )
    assert protocol["issue553_consumed"] is True

    assert protocol["issue545_trigger"] == 550
    assert protocol["issue545_run"] == 33686037672
    assert protocol["issue545_job"] == 100433658768
    assert protocol["issue545_failure"] == "FICEM read-tail floating dtypes must match"

    assert protocol["issue479_trigger"] == 484
    assert protocol["issue479_run"] == 33618950619
    assert protocol["issue479_job"] == 100211244996
    assert protocol["issue529_trigger"] == 529
    assert protocol["issue529_run"] == 33680028132
    assert protocol["issue529_job"] == 100414089065
    assert protocol["issue529_result_sha256"] == (
        "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
    )


def test_issue558_launcher_is_duplicate_safe_one_l4_and_durable_before_marker() -> None:
    source = LAUNCHER.read_text()
    assert f'SOURCE_MAIN = "{SOURCE_MAIN}"' in source
    assert f'SOURCE_TREE = "{SOURCE_TREE}"' in source
    assert (
        'APP_NAME = "tam-research-aera-v26-8-issue558-ficem-read-mixed-strength-precision"'
        in source
    )
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert (
        'RESULT_PATH = "/vol/aera-v26/issue558-ficem-read-mixed-strength-precision/result.json"'
        in source
    )
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("run_ficem_read_probe_v26_8_issue558()") == 1
    assert "if result_path.exists():" in source
    assert "refusing duplicate issue558 run because durable result exists" in source

    write_index = source.index("result_path.write_text(durable_json)")
    commit_index = source.index("volume.commit()", write_index)
    marker_index = source.index("print(RESULT_MARKER", commit_index)
    assert write_index < commit_index < marker_index
    preflight_index = source.index("check = preflight.remote()")
    gate_index = source.index("result = run_gate.remote()")
    assert preflight_index < gate_index

    for marker in (
        "AERA_V26_8_ISSUE558_FICEM_READ_MIXED_STRENGTH_PRECISION_PRECHECK_JSON=",
        "AERA_V26_8_ISSUE558_FICEM_READ_MIXED_STRENGTH_PRECISION_PREFLIGHT_JSON=",
        "AERA_V26_8_ISSUE558_FICEM_READ_MIXED_STRENGTH_PRECISION_L4_START_JSON=",
        "AERA_V26_8_ISSUE558_FICEM_READ_MIXED_STRENGTH_PRECISION_RESULT_JSON=",
        "AERA_V26_8_ISSUE558_FICEM_READ_MIXED_STRENGTH_PRECISION_SUMMARY_JSON=",
    ):
        assert marker in source
    assert "modal deploy" not in source
    assert "run_end_to_end_systems" not in source


def test_issue558_workflow_is_canonical_lowest_attempt1_without_concurrency() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "\nconcurrency:" not in source
    assert (
        "startsWith(github.event.issue.title, '[aera-v26-8-issue558-ficem-read-mixed-strength-precision-l4]')"
        in source
    )
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${#matching_triggers[@]}" -ge 1' in source
    assert 'canonical_trigger="${matching_triggers[0]}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${canonical_trigger}"' in source
    assert 'test "${TRIGGER_ISSUE}" = "${matching_triggers[0]}"' in source
    assert 'test "${#matching_triggers[@]}" = "1"' not in source
    assert "Bind main:" in source
    assert "git rev-parse HEAD" in source

    reverify = "\n      - name: Re-verify issue558 canonical trigger immediately before Modal\n"
    authenticate = "\n      - name: Authenticate Modal\n"
    assert reverify in source
    assert authenticate in source
    assert source.index(reverify) < source.index(authenticate)
    assert source.count(
        "modal run modal_aera_v26_8_issue558_ficem_read_mixed_strength_precision_l4_app.py"
    ) == 1
    assert "modal deploy" not in source


def test_issue558_workflow_hard_guards_cpu_merge_and_predecessor_results() -> None:
    source = WORKFLOW.read_text()
    for token in (
        "pulls/557",
        "783c5e2921d1d7fc7f598948d1bb968e51260440",
        "actions/runs/33730039451",
        "actions/jobs/100567632346",
        "issues/555",
        "actions/runs/33727540468",
        "actions/jobs/100559866985",
        '"decision":"FAIL"',
        '"historical":{"decision":"PASS","overall_pass":true',
        '"mixed":{"overall_pass":false,"rows_pass":false',
        "issues/550",
        "actions/runs/33686037672",
        "actions/jobs/100433658768",
        "5522486484",
    ):
        assert token in source

    for path, blob in EXPECTED_BLOBS.items():
        if path in (WORKFLOW,):
            continue
        relative = path.relative_to(ROOT).as_posix()
        # Only repository files explicitly frozen in the workflow guard are required here.
        if relative.startswith("tam_research/") or relative.startswith("tests/") or relative.startswith(".github/") or relative.startswith("modal_"):
            if relative in {
                "tam_research/aera_v26_8_issue558_ficem_read_mixed_strength_precision_probe.py",
                "modal_aera_v26_8_issue558_ficem_read_mixed_strength_precision_l4_app.py",
                "tam_research/aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py",
                "tests/test_aera_v26_8_issue556_ficem_read_mixed_strength_precision_cpu.py",
                "tam_research/aera_hardware_core_v26_7_ficem_read_mixed_dtype.py",
                "tam_research/aera_hardware_core_v26_3_ficem_read_triton.py",
                "tam_research/aera_hardware_core_v26_6_ficem_write_materialize_cast.py",
                "tam_research/aera_v26_7_issue553_ficem_read_mixed_dtype_probe.py",
                "modal_aera_v26_7_issue553_ficem_read_mixed_dtype_l4_app.py",
                ".github/workflows/aera-v26-7-issue553-ficem-read-mixed-dtype-l4.yml",
                "tam_research/aera_v26_3_ficem_read_probe.py",
                "tam_research/aera_v26_3_ficem_read_probe_repair5.py",
                "tam_research/aera_v26_6_issue530_end_to_end_systems.py",
                "tam_research/aera_hardware_core_v26.py",
                "tam_research/aera_hardware_core_v25_1_compact.py",
            }:
                assert relative in source
                assert blob in source


def test_issue558_higher_authorizations_remain_false() -> None:
    protocol = probe.issue558_protocol()
    for key in (
        "gpu_authorized_by_probe_module",
        "mixed_dtype_read_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
        "scientific_seed_consumed",
    ):
        assert protocol[key] is False


def test_issue558_harness_contains_no_model_checkpoint_training_or_scientific_path() -> None:
    probe_source = ISSUE558_PROBE.read_text().lower()
    launcher_source = LAUNCHER.read_text().lower()
    assert "run_end_to_end_systems" not in probe_source
    assert "run_end_to_end_systems" not in launcher_source
    assert "optimizer(" not in probe_source
    assert "backward(" not in probe_source
    assert "checkpoint_loaded\": true" not in probe_source
    assert "scientific_seed_consumed\": true" not in probe_source
    assert "100m_authorized\": true" not in probe_source
