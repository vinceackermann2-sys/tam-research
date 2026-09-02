from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_3_ficem_read_probe as historical
from tam_research import aera_v26_3_ficem_read_probe_repair5 as successor
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    fused_ficem_read_v26_3_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
HISTORICAL_PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
SUCCESSOR_PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe_repair5.py"
LAUNCHER = ROOT / "modal_aera_v26_3_ficem_read_repair5_successor_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair5-successor.yml"

SOURCE_MAIN = "fb1cc86f51f1b012cf2f74bfaf30d6d9b389ee34"
BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
SUCCESSOR_PROBE_BLOB = "6fd6518e10ed1ef4115863f98ac591ffd77ce903"
LAUNCHER_BLOB = "bc0e14c56530e713d3638cd96431329e254a4fcb"
SUCCESSOR_PR = 478
SUCCESSOR_CPU_RUN = 33611259063
SUCCESSOR_CPU_JOB = 100186694663
SUCCESSOR_CPU_HEAD = "d092464d3b62a0703c53b238164ee16e975142ce"
EXHAUSTED_RUN = 33608906596
EXHAUSTED_JOB = 100179200965


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue479_freezes_candidate_historical_and_successor_probe_blobs() -> None:
    assert _blob(BACKEND) == BACKEND_BLOB
    assert _blob(HISTORICAL_PROBE) == HISTORICAL_PROBE_BLOB
    assert _blob(SUCCESSOR_PROBE) == SUCCESSOR_PROBE_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB

    backend_protocol = fused_ficem_read_v26_3_protocol()
    for key in (
        "bf16_reference_rounding_repair3",
        "bf16_product_rounding_repair4",
        "bf16_actual_autocast_tail_repair5",
        "bf16_strength_bias_fp32_repair5",
        "bf16_logits_fp32_repair5",
        "bf16_final_weights_fp32_repair5",
        "bf16_recalled_fp32_repair5",
    ):
        assert backend_protocol[key] is True
    assert backend_protocol["bf16_product_rounding_active_after_repair5"] is False
    assert backend_protocol["float32_path_changed_by_repair5"] is False

    successor_protocol = successor.issue477_protocol()
    assert successor_protocol["diagnostic_reference_precision_context_corrected"] is True
    assert successor_protocol["historical_probe_modified"] is False
    assert successor_protocol["candidate_path_changed_by_probe_successor"] is False
    assert successor_protocol["fixtures_changed_by_probe_successor"] is False
    assert successor_protocol["thresholds_changed_by_probe_successor"] is False
    assert successor_protocol["timing_changed_by_probe_successor"] is False

    for protocol in (backend_protocol, successor_protocol):
        for key in (
            "end_to_end_systems_authorized",
            "architecture_freeze_authorized",
            "s2_authorized",
            "fresh_scientific_seed_authorized",
            "100m_authorized",
            "breakthrough_proven",
        ):
            assert protocol[key] is False


def test_issue479_inherits_original_design_tolerances_and_thresholds_exactly() -> None:
    assert successor.DESIGN_SEED is historical.DESIGN_SEED
    assert successor.DESIGN_SEED == 408_411
    assert (successor.D_MODEL, successor.TIME, successor.CAPACITY, successor.MEMORY_DIM) == (
        200,
        256,
        48,
        50,
    )
    assert successor.BATCH_SIZES == (8, 64)
    assert successor.DTYPE_NAMES == ("float32", "bfloat16")
    assert successor.VALIDITY_KINDS == ("mixed", "full")
    assert (successor.WARMUP_CALLS, successor.TIMED_ROUNDS, successor.CALLS_PER_ROUND) == (
        10,
        5,
        100,
    )
    assert (successor.FP32_ATOL, successor.FP32_RTOL) == (1e-5, 1e-5)
    assert (successor.BF16_ATOL, successor.BF16_RTOL) == (1e-2, 1e-2)
    assert successor.MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert successor.MAX_ROW_LATENCY_RATIO == 1.05
    assert successor.MAX_FULL_EVENT_RATIO == 0.75

    source = SUCCESSOR_PROBE.read_text()
    assert "generator = torch.Generator().manual_seed(DESIGN_SEED)" in source
    assert "case = frozen.make_case(" in source
    assert "known_empty[key] = frozen.known_empty_case(" in source
    assert "selection = frozen._tie_aware_top4_equivalence(" in source
    assert '"pre_out_recalled_close": recalled_close' in source
    assert '"final_out_close": final_close' in source
    assert '"query_and_normalized_keys_bit_exact": reuse_exact' in source
    assert "timing = frozen._timed_summaries(calls)" in source
    assert "frozen._cuda_profile(call)" in source
    assert "frozen._peak_vram(call)" in source
    assert "candidate_ordinal" not in source
    assert "eligible_case" not in source
    assert "resampl" not in source.lower()
    assert "nudge" not in source.lower()


def test_issue479_launcher_is_duplicate_safe_one_l4_and_successor_once() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-3-issue479-ficem-read-repair5-successor"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue479-ficem-read-repair5-successor/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("def run_probe()") == 1
    assert source.count("result_path.exists()") >= 2
    assert "refusing duplicate issue479 FICEM read repair5 successor run" in source
    assert SOURCE_MAIN in source
    assert BACKEND_BLOB in source
    assert HISTORICAL_PROBE_BLOB in source
    assert SUCCESSOR_PROBE_BLOB in source
    assert f"SOURCE_SUCCESSOR_PR = {SUCCESSOR_PR}" in source
    assert f"SOURCE_SUCCESSOR_CPU_RUN = {SUCCESSOR_CPU_RUN}" in source
    assert f"SOURCE_SUCCESSOR_CPU_JOB = {SUCCESSOR_CPU_JOB}" in source
    assert SUCCESSOR_CPU_HEAD in source
    assert f"EXHAUSTED_GATE_RUN = {EXHAUSTED_RUN}" in source
    assert f"EXHAUSTED_GATE_JOB = {EXHAUSTED_JOB}" in source
    assert source.count("result = run_ficem_read_probe_repair5()") == 1
    assert "run_ficem_read_probe_repair5(" not in source.replace(
        "result = run_ficem_read_probe_repair5()", ""
    )


def test_issue479_launcher_persists_before_authoritative_marker() -> None:
    source = LAUNCHER.read_text()
    write = source.index("result_path.write_text(durable_json)")
    commit = source.index("volume.commit()")
    digest = source.index("result_sha256 = hashlib.sha256")
    marker = source.index("AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_RESULT_JSON=")
    summary = source.index("AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_SUMMARY_JSON=")
    assert write < commit < digest < marker < summary
    assert '"result_sha256": result_sha256' in source
    assert '"result_path": RESULT_PATH' in source
    assert 'result["rows"]["bfloat16_batch8_mixed"]["correctness"]' in source
    for key in (
        "selection_semantically_equivalent",
        "distinct_selected_set_exact",
        "tied_selection_semantically_valid",
        "pre_out_recalled_close",
        "final_out_close",
        "query_and_normalized_keys_bit_exact",
        "source_unchanged",
        "finite",
        "dtype_device_shape_exact",
        "pre_out_max_abs_diff",
        "final_out_max_abs_diff",
    ):
        assert f'"{key}"' in source


def test_issue479_launcher_has_no_model_training_or_alternate_acceptance_path() -> None:
    source = LAUNCHER.read_text()
    for forbidden in (
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        "seed8471",
        "aera.pt",
        "transformer.pt",
        "candidate_ordinal",
        "eligible_case",
        "resample",
        "nudge",
    ):
        assert forbidden not in source
    assert '"synthetic_only": True' in source
    assert '"scientific_seed_consumed": False' in source
    assert '"end_to_end_systems_authorized": False' in source
    assert '"architecture_freeze_authorized": False' in source
    assert '"100m_authorized": False' in source
    assert '"breakthrough_proven": False' in source


def test_issue479_workflow_is_attempt1_bound_single_invocation_and_no_retry() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-3-ficem-read-l4-repair5-successor]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'test "${marker_count}" = "0"' in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert SOURCE_MAIN in source
    assert SUCCESSOR_CPU_HEAD in source
    assert str(SUCCESSOR_CPU_RUN) in source
    assert str(SUCCESSOR_CPU_JOB) in source
    assert str(EXHAUSTED_RUN) in source
    assert str(EXHAUSTED_JOB) in source
    assert 'test "${TRIGGER_ISSUE}" != "476"' in source
    assert BACKEND_BLOB in source
    assert HISTORICAL_PROBE_BLOB in source
    assert SUCCESSOR_PROBE_BLOB in source
    assert LAUNCHER_BLOB in source
    assert source.count(
        "modal run modal_aera_v26_3_ficem_read_repair5_successor_app.py"
    ) == 1
    assert "AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE479_FICEM_READ_REPAIR5_SUCCESSOR_SUMMARY_JSON=" in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source
    assert "automatic retry" in source
    assert "redispatch" in source
    assert "alternate trigger" in source
    assert "timeout increase" in source


def test_issue479_workflow_permissions_are_narrow_and_reporting_best_effort() -> None:
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
