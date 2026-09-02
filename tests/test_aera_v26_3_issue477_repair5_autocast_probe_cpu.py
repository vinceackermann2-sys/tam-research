from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_3_ficem_read_probe as frozen
from tam_research import aera_v26_3_ficem_read_probe_repair5 as successor

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
SUCCESSOR = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe_repair5.py"
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"

SOURCE_MAIN = "2debbdb93e97ef8cc646f9730b83d61d3dcbda1a"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
FAILED_GATE = 474
FAILED_TRIGGER = 476
FAILED_RUN = 33608906596
FAILED_JOB = 100179200965


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue477_freezes_historical_probe_and_repair5_backend_exactly() -> None:
    assert _blob(HISTORICAL_PROBE) == HISTORICAL_PROBE_BLOB
    assert _blob(BACKEND) == REPAIR5_BACKEND_BLOB
    assert successor.HISTORICAL_PROBE_GIT_BLOB == HISTORICAL_PROBE_BLOB
    assert successor.REPAIR5_BACKEND_GIT_BLOB == REPAIR5_BACKEND_BLOB
    assert successor.SOURCE_MAIN == SOURCE_MAIN
    assert successor.SOURCE_FAILED_GATE == FAILED_GATE
    assert successor.SOURCE_FAILED_TRIGGER == FAILED_TRIGGER
    assert successor.SOURCE_FAILED_RUN == FAILED_RUN
    assert successor.SOURCE_FAILED_JOB == FAILED_JOB


def test_issue477_aliases_the_frozen_synthetic_contract_without_new_values() -> None:
    assert successor.DESIGN_SEED is frozen.DESIGN_SEED
    assert successor.D_MODEL is frozen.D_MODEL
    assert successor.MEMORY_DIM is frozen.MEMORY_DIM
    assert successor.CAPACITY is frozen.CAPACITY
    assert successor.TIME is frozen.TIME
    assert successor.BATCH_SIZES is frozen.BATCH_SIZES
    assert successor.DTYPE_NAMES is frozen.DTYPE_NAMES
    assert successor.VALIDITY_KINDS is frozen.VALIDITY_KINDS
    assert successor.WARMUP_CALLS is frozen.WARMUP_CALLS
    assert successor.TIMED_ROUNDS is frozen.TIMED_ROUNDS
    assert successor.CALLS_PER_ROUND is frozen.CALLS_PER_ROUND
    assert successor.FP32_ATOL is frozen.FP32_ATOL
    assert successor.FP32_RTOL is frozen.FP32_RTOL
    assert successor.BF16_ATOL is frozen.BF16_ATOL
    assert successor.BF16_RTOL is frozen.BF16_RTOL
    assert successor.MAX_GEOMEAN_LATENCY_RATIO is frozen.MAX_GEOMEAN_LATENCY_RATIO
    assert successor.MAX_ROW_LATENCY_RATIO is frozen.MAX_ROW_LATENCY_RATIO
    assert successor.MAX_FULL_EVENT_RATIO is frozen.MAX_FULL_EVENT_RATIO

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


def test_issue477_changes_only_diagnostic_reference_precision_context() -> None:
    source = SUCCESSOR.read_text()

    # The historical operations are wrapped in the real full-read precision context.
    assert source.count("with torch.no_grad(), frozen._precision_context(dtype_name):") == 2
    assert "return frozen._reference_tail(similarity, state)" in source
    assert "return frozen._reference_masked_logits(similarity, state)" in source

    # Exactly three decision-path reference diagnostic uses are routed through those helpers:
    # correctness tail, correctness masked logits, and near-tie tail.
    assert source.count("_reference_tail_in_full_read_context(") == 3
    assert source.count("_reference_masked_logits_in_full_read_context(") == 2
    assert "reference_recalled, reference_indices = _reference_tail_in_full_read_context(\n        case.dtype_name, similarity, case.state\n    )" in source
    assert "masked_logits = _reference_masked_logits_in_full_read_context(\n        case.dtype_name, similarity, case.state\n    )" in source
    assert "reference_recalled, reference_indices = _reference_tail_in_full_read_context(\n        dtype_name, similarity, state\n    )" in source

    # Production full reads remain delegated byte-for-byte to the frozen execution path.
    assert "reference_result = frozen._full_read(reference, memory, case)" in source
    assert "candidate_result = frozen._full_read(candidate, memory, case)" in source


def test_issue477_candidate_path_has_no_dtype_coercion_or_acceptance_escape_hatch() -> None:
    source = SUCCESSOR.read_text()
    assert "candidate_recalled, candidate_indices = frozen.fused_ficem_read_tail(" in source
    assert "reference_recalled, candidate_recalled, atol=atol, rtol=rtol" in source
    assert "reference_result.recalled,\n        candidate_result.recalled," in source
    assert "reference_recalled.float(), candidate_recalled.float()" not in source
    assert "candidate_recalled.float(), reference_recalled.float()" not in source
    assert "candidate_recalled = candidate_recalled.float()" not in source
    assert "reference_recalled = reference_recalled.float()" not in source
    assert "candidate_ordinal" not in source
    assert "eligible_case" not in source
    assert "resampl" not in source.lower()
    assert "nudge" not in source.lower()
    assert "monkeypatch" not in source.lower()
    assert "setattr(frozen" not in source
    assert "frozen.correctness_row =" not in source
    assert "frozen.near_tie_correctness =" not in source
    assert "frozen.run_ficem_read_probe =" not in source


def test_issue477_preserves_fixture_order_tie_semantics_and_timing_accounting() -> None:
    source = SUCCESSOR.read_text()
    assert "generator = torch.Generator().manual_seed(DESIGN_SEED)" in source
    assert "for dtype_name in DTYPE_NAMES:" in source
    assert "for batch_size in BATCH_SIZES:" in source
    assert "for validity_kind in VALIDITY_KINDS:" in source
    assert "case = frozen.make_case(" in source
    assert "known_empty[key] = frozen.known_empty_case(" in source
    assert "selection = frozen._tie_aware_top4_equivalence(" in source
    assert 'selection["selection_semantically_equivalent"]' in source
    assert 'selection["distinct_selected_set_exact"]' in source
    assert '"tied_selection_semantically_valid": selection[' in source
    assert '"pre_out_recalled_close": recalled_close' in source
    assert '"final_out_close": final_close' in source
    assert '"query_and_normalized_keys_bit_exact": reuse_exact' in source
    assert "timing = frozen._timed_summaries(calls)" in source
    assert "frozen._cuda_profile(call)" in source
    assert "frozen._peak_vram(call)" in source
    assert "latency_ratio <= MAX_ROW_LATENCY_RATIO" in source
    assert "event_ratio <= MAX_FULL_EVENT_RATIO" in source
    assert "geomean <= MAX_GEOMEAN_LATENCY_RATIO" in source
    assert 'tail_profile["cuda_device_events"] == 1' in source
    assert 'tail_profile["triton_read_tail_events"] == 1' in source
    assert source.count('profiles["candidate"]["relevant_operator_calls"][') == 3
    assert '"topk"]' in source
    assert '"softmax"' in source
    assert '"gather"' in source


def test_issue477_cpu_preflight_and_protocol_authorize_no_gpu_or_science() -> None:
    contract = successor.cpu_contract_preflight()
    protocol = contract["protocol"]
    assert contract["gpu_authorized_by_cpu_preflight"] is False
    assert contract["synthetic_only"] is True
    assert contract["scientific_seed_consumed"] is False
    assert protocol["diagnostic_reference_precision_context_corrected"] is True
    assert protocol["historical_probe_modified"] is False
    assert protocol["candidate_path_changed_by_probe_successor"] is False
    assert protocol["fixtures_changed_by_probe_successor"] is False
    assert protocol["thresholds_changed_by_probe_successor"] is False
    assert protocol["timing_changed_by_probe_successor"] is False
    for key in (
        "gpu_authorized_by_issue477",
        "scientific_seed_consumed",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue477_adds_no_launcher_workflow_model_training_or_scientific_path() -> None:
    source = SUCCESSOR.read_text()
    assert "import modal" not in source
    assert "modal." not in source
    assert "workflow_dispatch" not in source
    assert "torch.load(" not in source
    assert "load_state_dict" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "seed8471" not in source
    assert "aera.pt" not in source
    assert "transformer.pt" not in source
    assert '"model_loaded": False' in source
    assert '"checkpoint_loaded": False' in source
    assert '"corpus_accessed": False' in source
    assert '"training_performed": False' in source
    assert '"optimizer_created": False' in source
    assert '"backward_performed": False' in source
    assert '"scientific_seed_consumed": False' in source
    assert '"end_to_end_systems_authorized": False' in source
    assert '"architecture_freeze_authorized": False' in source
    assert '"s2_authorized": False' in source
    assert '"fresh_scientific_seed_authorized": False' in source
    assert '"100m_authorized": False' in source
    assert '"breakthrough_proven": False' in source
