from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "modal_cortex_s_100m_2b_v5_triton_scan_preflight.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-s-100m-2b-preflight-v5-triton-scan.yml"
TRAIN_IMPL = ROOT / "architectures/cortex_s/experiments/scale100m_2b/train.py"
EXPECTED_TRAIN_IMPL_SHA256 = "6ded33f5981ab2bad6aad75b6fc238a4608960fd9ac839f1f3fa71924ad58844"


def _function_source(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"def {name}(")
    if next_name is None:
        return text[start:]
    return text[start:text.index(f"def {next_name}(", start + 1)]


def test_runner_is_syntax_valid_and_prereg_constants_are_frozen() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    required = (
        'PHASE = "preflight-v5-triton-scan"',
        'TRIGGER_TITLE = "[modal-cortex-s-100m-2b-preflight-v5-triton-scan]"',
        "ENGINEERING_SEED = 2_026_091_002",
        'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/preflight-v5-triton-scan"',
        "EXPECTED_CORTEX_PARAMS = 101_778_112",
        "SEQ_LEN = 512",
        "MICRO_BATCH_SIZE = 64",
        "GRAD_ACCUM_STEPS = 2",
        "GLOBAL_BATCH = 128",
        "TOTAL_OPTIMIZER_STEPS = 30_518",
        "FULL_BATCH_TOKEN_EXPOSURES = 2_000_027_648",
        "MEASURED_STEPS = 80",
        "FIRST_WINDOW_STEPS = (1, 40)",
        "STEADY_WINDOW_STEPS = (41, 80)",
        "STARTUP_ALLOWANCE_SECONDS = 1_144.16",
        "MIN_STEADY_TOKENS_PER_SECOND = 259_750.0",
        "MAX_FULL_ENVELOPE_SECONDS = 8_500.0",
        "MAX_ALLOCATED_VRAM_GIB = 80.0",
        f'TRAIN_IMPLEMENTATION_SHA256 = "{EXPECTED_TRAIN_IMPL_SHA256}"',
    )
    for needle in required:
        assert needle in text
    assert text.count('gpu="H100!"') == 1
    assert "train_full_2b(" not in text
    assert "torch.autocast" not in text


def test_dispatch_marker_is_committed_before_h100_invocation() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    reserve = _function_source(text, "reserve_h100_dispatch", "_one_optimizer_step_no_amp")
    h100 = _function_source(text, "h100_preflight", "main")
    main = _function_source(text, "main")
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in reserve
    assert "_atomic_write(marker_path, marker)" in reserve
    assert "volume.commit()" in reserve
    assert 'marker_path = root / "H100_DISPATCH_CONSUMED.json"' in h100
    assert "if not marker_path.exists():" in h100
    assert "_atomic_write(marker_path" not in h100
    assert main.index("reserve_h100_dispatch.remote") < main.index("h100_preflight.remote")


def test_controlling_full_envelope_gate_is_computed_from_steady_window() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    h100 = _function_source(text, "h100_preflight", "main")
    assert "steady_seconds = sum(step_seconds[STEADY_START_INDEX:])" in h100
    assert "steady_tps = STEADY_TOKENS / max(steady_seconds, 1e-9)" in h100
    assert "projected_training_seconds = FULL_BATCH_TOKEN_EXPOSURES / steady_tps" in h100
    assert "projected_full_seconds = projected_training_seconds + STARTUP_ALLOWANCE_SECONDS" in h100
    assert '"steady_state_tps_ge_259750": steady_tps >= MIN_STEADY_TOKENS_PER_SECOND' in h100
    assert '"projected_full_envelope_le_8500_seconds": projected_full_seconds <= MAX_FULL_ENVELOPE_SECONDS' in h100
    assert 'status = "SYSTEMS_PREFLIGHT_PASS" if all(gates.values()) else "SYSTEMS_PREFLIGHT_FAIL"' in h100


def test_no_amp_and_actual_dispatcher_fallback_is_fail_closed() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    step = _function_source(text, "_one_optimizer_step_no_amp", "_prove_triton_dispatch")
    proof = _function_source(text, "_prove_triton_dispatch", "h100_preflight")
    assert "torch.autocast" not in step
    assert "integration.affine_scan_triton_candidate = counted_candidate" in proof
    assert "integration.affine_scan = forbidden_fallback" in proof
    assert 'calls["candidate"] != EXPECTED_LAYERS' in proof
    assert 'calls["fallback"] != 0' in proof


def test_workflow_is_exact_issue_triggered_and_has_no_manual_dispatch() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "issues:" in text and "types: [opened]" in text
    assert "workflow_dispatch" not in text
    assert "github.event.issue.user.login == github.repository_owner" in text
    assert "github.event.issue.title == '[modal-cortex-s-100m-2b-preflight-v5-triton-scan]'" in text
    assert 'test "$(git -C source rev-parse origin/main)" = "$SOURCE_SHA"' in text
    assert 'git -C source merge-base --is-ancestor "$HARNESS_SHA" "$SOURCE_SHA"' in text
    assert '426f9ee9916ea293dedd89f4fecb26a31e0b78f3' in text
    assert "modal run --detach --timestamps modal_cortex_s_100m_2b_v5_triton_scan_preflight.py" in text


def test_training_implementation_sha256_matches_preregistered_identity() -> None:
    assert hashlib.sha256(TRAIN_IMPL.read_bytes()).hexdigest() == EXPECTED_TRAIN_IMPL_SHA256
