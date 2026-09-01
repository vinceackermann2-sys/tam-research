from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_3_ficem_read_probe as probe
from tam_research import aera_v26_3_repair4_product_reduction_localize as localize

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tam_research" / "aera_v26_3_repair4_product_reduction_localize.py"
LAUNCHER = ROOT / "modal_aera_v26_3_repair4_product_reduction_localize_app.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "aera-v26-3-repair4-product-reduction-localize-l4.yml"
)
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
FROZEN_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
FROZEN_REPAIR4_BACKEND_BLOB = "a3a603c8a2d4b20ebcccd7663970978f4288a760"


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue452_contract_is_localization_only_and_frozen() -> None:
    contract = localize.cpu_contract_preflight()
    assert contract["research_issue"] == 452
    assert contract["source_main"] == "9d45b41d40c6859f4dc4ffc1b70c26e0f7768976"
    assert contract["source_failed_trigger"] == 451
    assert contract["source_failed_actions_run"] == 33537116699
    assert contract["source_failed_job"] == 99954032841
    assert contract["target_row"] == "bfloat16_batch8_mixed"
    assert contract["target_ordinal"] == 5
    assert contract["design_seed"] == 408411
    assert contract["design_seed_is_scientific_seed"] is False
    assert contract["frozen_probe_git_blob"] == FROZEN_PROBE_BLOB
    assert contract["frozen_repair4_backend_git_blob"] == FROZEN_REPAIR4_BACKEND_BLOB
    assert contract["original_global_case_order_preserved"] is True
    assert contract["resampling"] is False
    assert contract["rejection_sampling"] is False
    assert contract["fixture_nudging"] is False
    assert contract["alternate_seed"] is False
    assert contract["localization_only"] is True
    assert contract["timing_authorized"] is False
    assert contract["profiling_authorized"] is False
    assert contract["performance_decision_authorized"] is False
    assert contract["production_backend_modified"] is False
    assert contract["production_probe_modified"] is False
    assert contract["scientific_seed_consumed"] is False
    assert contract["end_to_end_systems_authorized"] is False
    assert contract["architecture_freeze_authorized"] is False
    assert contract["s2_authorized"] is False
    assert contract["fresh_scientific_seed_authorized"] is False
    assert contract["100m_authorized"] is False
    assert contract["breakthrough_proven"] is False


def test_issue452_frozen_backend_probe_and_fixture_contract_are_exact() -> None:
    assert _git_blob(BACKEND) == FROZEN_REPAIR4_BACKEND_BLOB
    assert _git_blob(PROBE) == FROZEN_PROBE_BLOB
    assert probe.DESIGN_SEED == 408411
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert (probe.D_MODEL, probe.TIME, probe.CAPACITY, probe.MEMORY_DIM) == (
        200,
        256,
        48,
        50,
    )
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert localize._ordinary_rows()[4] == "bfloat16_batch8_mixed"


def test_issue452_invokes_frozen_correctness_row_once_and_has_no_fixture_escape() -> None:
    source = MODULE.read_text()
    assert source.count("probe.correctness_row(memory, case, reference, candidate)") == 1
    assert "torch.Generator().manual_seed(probe.DESIGN_SEED)" in source
    assert "probe.make_case(" in source
    assert 'TARGET_ROW = "bfloat16_batch8_mixed"' in source
    assert "TARGET_ORDINAL = 5" in source
    assert "manual_seed(probe.DESIGN_SEED +" not in source
    assert "candidate_ordinal" not in source
    assert "eligible_case" not in source
    assert "while " not in source


def test_issue452_has_one_isolated_product_reduction_triton_microkernel() -> None:
    source = MODULE.read_text()
    assert source.count("@triton.jit") == 1
    assert source.count("def _product_reduction_checkpoint_kernel(") == 1
    for token in (
        "raw0 = w0.to(tl.float32) * v0.to(tl.float32)",
        "raw3 = w3.to(tl.float32) * v3.to(tl.float32)",
        "p0 = raw0.to(tl.bfloat16).to(tl.float32)",
        "p3 = raw3.to(tl.bfloat16).to(tl.float32)",
        "sequential = p0 + p1 + p2 + p3",
        "pairwise = (p0 + p1) + (p2 + p3)",
        "diagnostic_product_mirror_matches_production",
    ):
        assert token in source
    module_token = "aera_v26_3_repair4_product_reduction_localize"
    for path in (ROOT / "tam_research").glob("*.py"):
        if path == MODULE:
            continue
        assert module_token not in path.read_text()


def test_issue452_diagnostic_has_no_timing_training_model_or_acceptance_path() -> None:
    source = MODULE.read_text()
    for forbidden in (
        "torch.cuda.Event",
        "torch.profiler",
        "_timed_round",
        "_timed_summaries",
        "latency_ratio",
        "MAX_GEOMEAN_LATENCY_RATIO",
        "MAX_ROW_LATENCY_RATIO",
        "MAX_FULL_EVENT_RATIO",
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        ".step(",
        "SOURCE_RUN_DIR",
        "aera.pt",
        "transformer.pt",
        "seed8471",
    ):
        assert forbidden not in source
    assert '"performance_decision": None' in source
    assert '"candidate_acceptance_changed": False' in source


def test_issue452_launcher_is_duplicate_safe_one_l4_and_durable() -> None:
    source = LAUNCHER.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue452-repair4-product-reduction-localize/result.json"' in source
    assert "AERA_V26_ISSUE452_REPAIR4_PRODUCT_REDUCTION_LOCALIZE_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE452_REPAIR4_PRODUCT_REDUCTION_LOCALIZE_SUMMARY_JSON=" in source
    assert "MAX_GPU_SECONDS = 180" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("result_path.exists()") >= 2
    assert FROZEN_PROBE_BLOB in source
    assert FROZEN_REPAIR4_BACKEND_BLOB in source
    for forbidden in (
        "torch.load(",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        "modal deploy",
        "seed8471",
    ):
        assert forbidden not in source


def test_issue452_workflow_is_attempt1_exact_bound_and_source_failure_grounded() -> None:
    source = WORKFLOW.read_text()
    assert "types: [opened]" in source
    assert "[aera-v26-3-repair4-product-reduction-localize-l4]" in source
    assert '${GITHUB_RUN_ATTEMPT}' in source
    assert '!= "1"' in source
    assert "33537116699" in source
    assert "99954032841" in source
    assert "Run sole issue449 repair4 synthetic L4 probe" in source
    assert 'test "${source_l4_step}" = "failure"' in source
    assert "9d45b41d40c6859f4dc4ffc1b70c26e0f7768976" in source
    assert "Expected exactly one issue452 repair4 product/reduction localization trigger" in source
    assert FROZEN_PROBE_BLOB in source
    assert FROZEN_REPAIR4_BACKEND_BLOB in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "workflow_dispatch" not in source
    assert "gh run rerun" not in source
    assert "rerun_workflow" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source


def test_issue452_workflow_permissions_and_marker_are_narrow() -> None:
    source = WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests:" not in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert source.count("modal run modal_aera_v26_3_repair4_product_reduction_localize_app.py") == 1
    assert "AERA_V26_ISSUE452_REPAIR4_PRODUCT_REDUCTION_LOCALIZE_RESULT_JSON=" in source
    assert "No rerun, retry, redispatch, alternate trigger, or timeout increase is authorized" in source
