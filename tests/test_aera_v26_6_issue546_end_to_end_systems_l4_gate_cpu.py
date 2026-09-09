from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_6_issue530_end_to_end_systems as evaluator
from tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast import (
    MaterializeCastTritonFICEMReadWriteBackend,
    materialize_cast_ficem_read_write_v26_6_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_6_issue546_end_to_end_systems_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-6-issue546-e2e-systems-l4.yml"
HISTORICAL_LAUNCHER = ROOT / "modal_aera_v26_6_issue539_end_to_end_systems_app.py"
HISTORICAL_WORKFLOW = ROOT / ".github/workflows/aera-v26-6-issue539-e2e-systems-l4.yml"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue546_freezes_exact_evaluator_historical_and_successor_blobs() -> None:
    assert _git_blob_sha(Path(evaluator.__file__)) == "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
    assert _git_blob_sha(HISTORICAL_LAUNCHER) == "4663d86e9dc2f8eedb68213cf673ce1f80e15574"
    assert _git_blob_sha(HISTORICAL_WORKFLOW) == "4e03f6e42c7635f4779f17aaa8d30488d968bc1f"
    assert _git_blob_sha(LAUNCHER) == "1c4ad1bd314007ac31e61a68e844b61969320f50"
    assert _git_blob_sha(WORKFLOW) == "21f509e1653b0214f4f03257d27aedc6b9e045e7"


def test_issue546_preserves_entire_frozen_systems_surface() -> None:
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

    protocol = evaluator.issue530_systems_protocol()
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


def test_issue546_candidate_is_exact_primitive_passed_v26_6_backend() -> None:
    protocol = evaluator.issue530_systems_protocol()
    candidate = materialize_cast_ficem_read_write_v26_6_protocol()
    assert evaluator.V26_6_CANDIDATE_BLOB == "d45c262314a0b4691f26812a279937a225043ad9"
    assert protocol["candidate_backend"] == MaterializeCastTritonFICEMReadWriteBackend.name
    assert protocol["issue527_trigger"] == 529
    assert protocol["issue527_run"] == 33680028132
    assert protocol["issue527_job"] == 100414089065
    assert protocol["issue527_result_sha256"] == (
        "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
    )
    assert protocol["issue527_decision"] == "PASS"
    assert protocol["issue527_direct_pass"] == [256, 256]
    assert protocol["issue527_edge_pass"] == [32, 32]
    assert protocol["issue527_public_pass"] == [6, 6]
    assert protocol["issue527_topology_pass"] == [4, 4]
    assert candidate["write_fieldwise_mixed_dtype_supported"] is True
    assert candidate["write_global_cross_field_dtype_equality_required"] is False
    assert candidate["write_supported_float_dtypes"] == ["float32", "bfloat16"]
    assert candidate["write_tail_triton_launches_target"] == 2


def test_issue546_launcher_is_new_duplicate_safe_one_l4_result_before_marker() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-6-issue546-end-to-end-systems"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue546-end-to-end-systems/result.json"' in source
    assert 'SOURCE_MAIN = "8db5422929f66f89f323f2fc518c1d6b9b224581"' in source
    assert 'SOURCE_TREE = "bb20291652c5a60ac99e16d6e4ba0e53ffa27475"' in source
    assert "MAX_GPU_SECONDS = 600" in source
    assert 'gpu="L4"' in source
    assert "timeout=MAX_GPU_SECONDS" in source
    assert source.count("result = run_end_to_end_systems_v26_6()") == 1
    assert source.count("volume.commit()") == 1
    assert source.index("volume.commit()") < source.index("print(RESULT_MARKER +")
    assert "result_path.exists()" in source
    assert "refusing duplicate issue546" in source
    assert "ISSUE539_TRIGGERS = (542, 543, 544)" in source
    assert "ISSUE539_RUNS = (33684268881, 33684274250, 33684290786)" in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "modal.deploy" not in source


def test_issue546_workflow_uses_canonical_lowest_trigger_without_concurrency() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "\nconcurrency:" not in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "[aera-v26-6-issue546-e2e-systems-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert "matching=" in source
    assert "sort -n -u" in source
    assert 'canonical="$(printf' in source
    assert 'test "${TRIGGER_ISSUE}" = "${canonical}"' in source
    assert "trigger_count=" not in source
    assert 'test "${trigger_count}" = "1"' not in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor 8db5422929f66f89f323f2fc518c1d6b9b224581 HEAD" in source
    assert "modal run modal_aera_v26_6_issue546_end_to_end_systems_app.py" in source
    assert "gh run rerun" not in source.lower()
    assert "modal deploy" not in source.lower()


def test_issue546_workflow_freezes_consumed_issue539_race_as_pre_gpu() -> None:
    source = WORKFLOW.read_text()
    for issue in (542, 543, 544):
        assert f"issue in 542 543 544" in source or str(issue) in source
    assert "33684268881" in source
    assert "100427851578" in source
    assert "33684274250" in source
    assert "33684290786" in source
    assert "100428014282" in source
    assert 'select(.name == "Authenticate Modal") | .conclusion' in source
    assert 'select(.name == "Run sole issue539 end-to-end systems L4 gate") | .conclusion' in source
    assert 'actions/runs/33684274250/jobs" --jq \'.total_count\'' in source


def test_issue546_keeps_all_higher_authorizations_false() -> None:
    contract = evaluator.cpu_contract_preflight_issue530()
    protocol = evaluator.issue530_systems_protocol()
    assert contract["gpu_authorized_by_cpu_preflight"] is False
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
