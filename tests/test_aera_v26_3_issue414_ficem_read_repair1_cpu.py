from __future__ import annotations

import inspect
from pathlib import Path

from tam_research.aera_hardware_core_v24 import (
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
)
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    fused_ficem_read_tail,
)
from tam_research.aera_v26_3_ficem_read_probe import (
    BATCH_SIZES,
    BF16_ATOL,
    BF16_RTOL,
    CALLS_PER_ROUND,
    CAPACITY,
    DESIGN_SEED,
    DTYPE_NAMES,
    D_MODEL,
    FP32_ATOL,
    FP32_RTOL,
    MAX_FULL_EVENT_RATIO,
    MAX_GEOMEAN_LATENCY_RATIO,
    MAX_ROW_LATENCY_RATIO,
    MEMORY_DIM,
    TIME,
    TIMED_ROUNDS,
    VALIDITY_KINDS,
    WARMUP_CALLS,
)

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE_PATH = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
LAUNCHER_PATH = ROOT / "modal_aera_v26_3_ficem_read_repair1_app.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4-repair1.yml"


def _kernel_source() -> str:
    source = BACKEND_PATH.read_text()
    return source.split("def _ficem_read_tail_kernel(", 1)[1].split(
        "def triton_ficem_read_available", 1
    )[0]


def test_issue414_inherits_every_issue411_decision_constant_unchanged():
    assert DESIGN_SEED == 408_411
    assert (D_MODEL, TIME, CAPACITY, MEMORY_DIM) == (200, 256, 48, 50)
    assert BATCH_SIZES == (8, 64)
    assert DTYPE_NAMES == ("float32", "bfloat16")
    assert VALIDITY_KINDS == ("mixed", "full")
    assert (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) == (10, 5, 100)
    assert (FP32_ATOL, FP32_RTOL) == (1e-5, 1e-5)
    assert (BF16_ATOL, BF16_RTOL) == (1e-2, 1e-2)
    assert MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert MAX_ROW_LATENCY_RATIO == 1.05
    assert MAX_FULL_EVENT_RATIO == 0.75
    assert READ_TOP_K == 4
    assert READ_TEMPERATURE == 0.10
    assert MIN_STRENGTH == 1e-4


def test_issue414_probe_is_exact_issue411_probe_not_a_repair_variant():
    source = PROBE_PATH.read_text()
    assert '"""Issue #411 production-shaped synthetic gate for the fused FICEM read tail."""' in source
    assert "DESIGN_SEED = 408_411" in source
    assert "WARMUP_CALLS = 10" in source
    assert "TIMED_ROUNDS = 5" in source
    assert "CALLS_PER_ROUND = 100" in source
    assert "MAX_GEOMEAN_LATENCY_RATIO = 0.90" in source
    assert "MAX_ROW_LATENCY_RATIO = 1.05" in source
    assert "MAX_FULL_EVENT_RATIO = 0.75" in source
    assert "issue414" not in source.lower()


def test_issue414_kernel_constants_are_explicit_triton_constexpr_formals():
    source = BACKEND_PATH.read_text()
    kernel = _kernel_source()
    assert source.count("@triton.jit") == 1
    assert "MIN_STRENGTH: tl.constexpr" in kernel
    assert "READ_TEMPERATURE: tl.constexpr" in kernel
    assert "READ_TOP_K: tl.constexpr" in kernel
    assert "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS" not in source


def test_issue414_kernel_uses_exact_frozen_two_sided_strength_clamp():
    kernel = _kernel_source()
    assert "tl.minimum(tl.maximum(strengths, MIN_STRENGTH), 1.0)" in kernel
    assert "strength_bias = tl.log(clamped_strengths)" in kernel
    assert "logits = (similarity + strength_bias) / READ_TEMPERATURE" in kernel


def test_issue414_launch_passes_exact_frozen_globals_as_constexpr_values():
    source = inspect.getsource(fused_ficem_read_tail)
    assert "MIN_STRENGTH=MIN_STRENGTH" in source
    assert "READ_TEMPERATURE=READ_TEMPERATURE" in source
    assert "READ_TOP_K=READ_TOP_K" in source
    assert "SLOT_BLOCK=64" in source
    assert "DIM_BLOCK=64" in source
    assert "num_warps=4" in source


def test_issue414_does_not_change_accelerated_tail_architecture():
    source = BACKEND_PATH.read_text()
    assert "torch.topk(" not in source
    assert "torch.softmax(" not in source
    assert ".gather(" not in source
    assert "torch.cat(" not in source
    assert "torch.stack(" not in source
    assert "register_buffer" not in source
    assert "nn.Parameter" not in source
    kernel = _kernel_source()
    assert kernel.count("tl.argmax(") == 4
    assert "tl.exp(" in kernel
    assert "weight0 * value0" in kernel
    assert "weight3 * value3" in kernel


def test_issue414_launcher_has_new_unique_result_and_reuses_original_probe():
    source = LAUNCHER_PATH.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue414-ficem-read-repair1/result.json"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in source
    assert "result_path.exists()" in source
    assert "run_ficem_read_probe" in source
    assert "AERA_V26_ISSUE414_FICEM_READ_REPAIR1_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE414_FICEM_READ_REPAIR1_SUMMARY_JSON=" in source
    assert "source_failed_actions_run\": 33496856233" in source
    assert "probe_reused_unchanged\": True" in source
    assert "torch.load(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "seed8471" not in source.lower()


def test_issue414_workflow_is_separate_attempt1_boundary_not_issue411_rerun():
    source = WORKFLOW_PATH.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-3-ficem-read-l4-repair1]" in source
    assert 'if [ "${GITHUB_RUN_ATTEMPT}" != "1" ]; then' in source
    assert "33496856233" in source
    assert 'test "${run411_conclusion}" = "failure"' in source
    assert 'test "${run411_attempt}" = "1"' in source
    assert "Expected exactly one issue414 FICEM read repair1 trigger" in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor 8f956aea5dd25fbbef5c718e634da4d293b96b4e HEAD" in source
    assert source.count("modal run modal_aera_v26_3_ficem_read_repair1_app.py") == 1
    assert "AERA_V26_ISSUE414_FICEM_READ_REPAIR1_RESULT_JSON=" in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source


def test_issue414_workflow_permissions_and_no_threshold_mutation():
    source = WORKFLOW_PATH.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "contents: write" not in permissions
    assert "actions: write" not in permissions
    assert "pull-requests: write" not in permissions
    assert "timeout increase" in source
    assert "Record issue414 result (best effort)" in source
    assert "continue-on-error: true" in source
