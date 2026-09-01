from __future__ import annotations

from pathlib import Path

from tam_research.aera_v26_state_movement_probe import (
    BATCH_SIZES,
    CALLS_PER_ROUND,
    CAPACITY,
    D_MODEL,
    DESIGN_SEED,
    MAX_BATCH64_GEOMEAN_LATENCY_RATIO,
    MAX_BATCH64_ROW_LATENCY_RATIO,
    MAX_BATCH8_ROW_LATENCY_RATIO,
    MAX_KERNEL_RATIO,
    MEMORY_DIM,
    PACKED_FLOAT_WIDTH,
    SELECTED_FRACTIONS,
    TIMED_ROUNDS,
    WARMUP_CALLS,
    cpu_contract_preflight,
    issue400_protocol,
    run_cpu_correctness_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tam_research" / "aera_v26_state_movement_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_state_movement_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-state-movement-l4.yml"


def _text(path: Path) -> str:
    return path.read_text()


def test_issue400_cpu_contract_freezes_production_geometry_and_thresholds():
    check = cpu_contract_preflight()
    protocol = issue400_protocol()
    assert check["synthetic_only"] is True
    assert check["gpu_authorized_by_cpu_preflight"] is False
    assert protocol["research_issue"] == 400
    assert DESIGN_SEED == 398_400
    assert protocol["design_seed_is_scientific_seed"] is False
    assert (D_MODEL, MEMORY_DIM, CAPACITY, PACKED_FLOAT_WIDTH) == (200, 50, 48, 5_048)
    assert BATCH_SIZES == (8, 64)
    assert SELECTED_FRACTIONS == (0.25, 0.50, 0.75, 1.00)
    assert (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) == (10, 5, 200)
    assert MAX_KERNEL_RATIO == 0.80
    assert MAX_BATCH64_GEOMEAN_LATENCY_RATIO == 0.90
    assert MAX_BATCH64_ROW_LATENCY_RATIO == 1.05
    assert MAX_BATCH8_ROW_LATENCY_RATIO == 1.10
    assert protocol["model_loaded"] is False
    assert protocol["checkpoint_loaded"] is False
    assert protocol["corpus_accessed"] is False
    assert protocol["training_performed"] is False
    assert protocol["scientific_seed_consumed"] is False
    assert protocol["end_to_end_systems_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["100m_authorized"] is False


def test_issue400_cpu_correctness_matrix_is_exact_for_all_eight_rows():
    rows = run_cpu_correctness_matrix()
    assert len(rows) == 8
    assert set(rows) == {
        "batch8_fraction025",
        "batch8_fraction050",
        "batch8_fraction075",
        "batch8_fraction100",
        "batch64_fraction025",
        "batch64_fraction050",
        "batch64_fraction075",
        "batch64_fraction100",
    }
    for row in rows.values():
        assert row == {
            "pass": True,
            "selected_exact": True,
            "merged_exact": True,
            "source_unchanged": True,
            "finite": True,
            "output_device_matches_index_device": True,
        }


def test_issue400_harness_times_actual_merged_legacy_and_v26_helpers():
    src = _text(HARNESS)
    assert "selected = _select_epi_state(case.base, case.run_idx)" in src
    assert "merged = _merge_epi_state(case.base, case.update, case.run_idx)" in src
    assert "base_packed = pack_ephemeral_epi_state(case.base)" in src
    assert "selected_packed = select_packed_epi_state(base_packed, case.run_idx)" in src
    assert "selected = unpack_ephemeral_epi_state(selected_packed)" in src
    assert "update_packed = pack_ephemeral_epi_state(case.update)" in src
    assert "merged_packed = merge_packed_epi_state(" in src
    assert "merged = unpack_ephemeral_epi_state(merged_packed)" in src
    # Packing is inside v26_coalesced_movement, which is the callable passed to
    # the timed loop; there is no pre-packed benchmark-only fast path.
    assert '"v26": lambda case=case: v26_coalesced_movement(case)' in src
    assert "CUDA events" in src
    assert "torch.cuda.Event(enable_timing=True)" in src
    assert "torch.profiler.ProfilerActivity.CUDA" in src
    assert "cuda_device_events" in src


def test_issue400_harness_has_no_checkpoint_model_or_corpus_execution_path():
    src = _text(HARNESS)
    launcher = _text(LAUNCHER)
    combined = src + "\n" + launcher
    forbidden = (
        "torch.load(",
        "load_state_dict(",
        "SOURCE_RUN_DIR",
        "aera.pt",
        "transformer.pt",
        "ResearchLM(",
        "HardwareAwareAERATextLM",
        "aera_real_language",
        "seed8471",
    )
    for token in forbidden:
        assert token not in combined
    assert ".cpu()" not in src
    assert ".item()" not in src
    assert "run_state_movement_probe" in launcher


def test_issue400_modal_launcher_is_unique_one_l4_and_duplicate_safe():
    src = _text(LAUNCHER)
    assert 'APP_NAME = "tam-research-aera-v26-issue400-state-movement"' in src
    assert 'RESULT_PATH = "/vol/aera-v26/issue400-state-movement/result.json"' in src
    assert "MAX_GPU_SECONDS = 300" in src
    assert 'gpu="L4"' in src
    assert "refusing duplicate issue400 state-movement run because result exists" in src
    assert "AERA_V26_ISSUE400_STATE_MOVEMENT_PREFLIGHT_JSON=" in src
    assert "AERA_V26_ISSUE400_STATE_MOVEMENT_RESULT_JSON=" in src
    assert '"synthetic_only": True' in src
    assert '"scientific_seed_consumed": False' in src
    assert "run_cpu_correctness_matrix()" in src


def test_issue400_workflow_freezes_single_attempt_exact_bound_main_and_best_effort_reporting():
    src = _text(WORKFLOW)
    assert "[aera-v26-state-movement-l4]" in src
    assert "GITHUB_RUN_ATTEMPT" in src
    assert "Issue #400 permits Actions attempt 1 only" in src
    assert (
        "[research] AERA-v26 synthetic L4 coalesced-state movement gate after #399 CPU proof"
        in src
    )
    assert "94550b5d0844b61b8f62ee9f44063aece7785c36" in src
    assert "8730e79b696288ef40951a72a826f98dbc536499" in src
    assert "trigger_count" in src
    assert "Expected exactly one issue400 state-movement trigger" in src
    assert "Bind main:" in src
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in src
    assert "cancel-in-progress: false" in src
    assert "Record frozen issue400 guard (best effort)" in src
    assert "Record issue400 result (best effort)" in src
    assert "Record issue400 workflow failure (best effort)" in src
    assert src.count("continue-on-error: true") == 3
    assert 'gh api "repos/${GITHUB_REPOSITORY}/issues/${STATE_TRIGGER}/comments"' in src


def test_issue400_workflow_has_no_rerun_or_redispatch_mechanism():
    src = _text(WORKFLOW).lower()
    forbidden = (
        "gh run rerun",
        "workflow_dispatch",
        "/rerun-failed-jobs",
        "/rerun",
        "rerun_workflow",
        "modal deploy",
    )
    for token in forbidden:
        assert token not in src
    assert "modal run modal_aera_v26_state_movement_app.py" in src
    assert "aera_v26_issue400_state_movement_result_json=" in src
    assert 'if [ "${rc}" = "0" ]; then exit 1; fi' in src


def test_issue400_workflow_reporting_is_non_authoritative_and_gpu_scope_stays_narrow():
    src = _text(WORKFLOW)
    assert "contents: read" in src
    assert "issues: write" in src
    assert "actions: write" not in src
    assert "Synthetic production-shaped tensors only" in src
    assert "One NVIDIA L4 <=300 GPU seconds" in src
    assert "No model/checkpoint/corpus/seed8471" in src
    assert "no end-to-end systems run" in src
    assert "reporting is non-authoritative" in src
    assert "Durable Modal result + Actions marker are authoritative" in src
