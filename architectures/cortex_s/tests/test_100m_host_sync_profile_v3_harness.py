from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import architectures.cortex_s.experiments.scale100m_2b.train as training_module
from architectures.cortex_s.host_sync_profile_v3 import (
    EXPECTED_GROUPED_MM_CALLS_PER_OPTIMIZER_STEP,
    PROFILED_STEPS,
    STEADY_MEASURED_STEPS,
    STEADY_WARMUP_STEPS,
    aggregate_sync_attribution,
    classify_source,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_text(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


@dataclass
class FakeEvent:
    key: str
    self_cpu_time_total: float = 1.0
    self_device_time_total: float = 0.0
    cpu_parent: object | None = None
    stack: list[str] | None = None


def test_repo_root_resolution_is_cwd_independent() -> None:
    assert (REPO_ROOT / "architectures/cortex_s/grouped_moe_v4.py").is_file()
    assert (REPO_ROOT / "modal_cortex_s_100m_host_sync_profile_v3.py").is_file()


def test_grouped_mm_parent_is_attributed_before_generic_compiled_parent() -> None:
    compiled = FakeEvent("Call CompiledFxGraph")
    grouped = FakeEvent("aten::_grouped_mm", cpu_parent=compiled)
    sync = FakeEvent("cudaStreamSynchronize", self_cpu_time_total=10.0, cpu_parent=grouped)
    source, parents, _ = classify_source(sync)
    assert source == "grouped_mm"
    assert parents[:2] == ["aten::_grouped_mm", "Call CompiledFxGraph"]


def test_scalar_parent_is_attributed() -> None:
    scalar = FakeEvent("aten::_local_scalar_dense")
    sync = FakeEvent("cudaStreamSynchronize", cpu_parent=scalar)
    source, _, _ = classify_source(sync)
    assert source == "training_scalar_sync"


def test_frozen_dominant_threshold_classifies_grouped_mm() -> None:
    grouped_parent = FakeEvent("aten::_grouped_mm")
    scalar_parent = FakeEvent("aten::_local_scalar_dense")
    events = [FakeEvent("cudaStreamSynchronize", cpu_parent=grouped_parent) for _ in range(160)]
    events.extend(FakeEvent("cudaStreamSynchronize", cpu_parent=scalar_parent) for _ in range(46))
    result = aggregate_sync_attribution(events, profiled_steps=2)
    assert result["stream_sync_events"] == 206
    assert result["stream_sync_events_per_optimizer_step"] == 103.0
    assert result["stream_sync_events_per_step_by_source"]["grouped_mm"] == 80.0
    assert result["classification"] == "GROUPED_MM_HOST_SYNC_DOMINANT"


def test_other_dominant_source_classification() -> None:
    routing_parent = FakeEvent("aten::sort")
    grouped_parent = FakeEvent("aten::_grouped_mm")
    events = [FakeEvent("cudaStreamSynchronize", cpu_parent=routing_parent) for _ in range(120)]
    events.extend(FakeEvent("cudaStreamSynchronize", cpu_parent=grouped_parent) for _ in range(80))
    result = aggregate_sync_attribution(events, profiled_steps=2)
    assert result["largest_source"] == "routing"
    assert result["classification"] == "OTHER_HOST_SYNC_SOURCE_DOMINANT"


def test_unknown_attribution_fails_closed() -> None:
    events = [FakeEvent("cudaStreamSynchronize") for _ in range(206)]
    result = aggregate_sync_attribution(events, profiled_steps=2)
    assert result["classification"] == "ATTRIBUTION_INCONCLUSIVE"
    assert result["non_generic_attribution_coverage"] == 0.0


def test_exact_diagnostic_constants_are_frozen() -> None:
    assert STEADY_WARMUP_STEPS == 5
    assert STEADY_MEASURED_STEPS == 4
    assert PROFILED_STEPS == 2
    assert EXPECTED_GROUPED_MM_CALLS_PER_OPTIMIZER_STEP == 96


def test_production_call_arithmetic_and_runtime_contract_are_unchanged() -> None:
    grouped_v4 = _repo_text("architectures/cortex_s/grouped_moe_v4.py")
    grouped_base = _repo_text("architectures/cortex_s/grouped_moe.py")

    assert "hidden_physical = self._grouped_mm(packed, w1_physical, plan.offsets)" in grouped_v4
    assert "expert_output = self._grouped_mm(hidden_physical, w2_physical, plan.offsets)" in grouped_v4
    assert "return F.grouped_mm(packed_bf16, matrices_bf16, offs=offsets_i32)" in grouped_base
    assert int(training_module.GRAD_ACCUM_STEPS) == 2
    assert int(training_module.MICRO_BATCH_SIZE) == 64
    assert int(training_module.SEQ_LEN) == 512
    assert training_module.COMPILE_MODE == "max-autotune-no-cudagraphs"
    assert 24 * 2 * int(training_module.GRAD_ACCUM_STEPS) == EXPECTED_GROUPED_MM_CALLS_PER_OPTIMIZER_STEP


def test_authority_and_profiler_contract_are_static() -> None:
    core = _repo_text("architectures/cortex_s/host_sync_profile_v3.py")
    runner = _repo_text("modal_cortex_s_100m_host_sync_profile_v3.py")
    workflow = _repo_text(".github/workflows/modal-cortex-s-100m-host-sync-profile-v3.yml")

    assert 'PHASE = "host-sync-profile-v3"' in runner
    assert 'TRIGGER_TITLE = "[modal-cortex-s-100m-host-sync-profile-v3]"' in runner
    assert "ENGINEERING_SEED = 2_026_091_007" in runner
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/100m-2b/host-sync-profile-v3"' in runner
    assert "2_026_091_006" in runner
    assert "FORBIDDEN_CONTROL_AND_SCIENTIFIC_SEEDS = (8_100, 48_131, 48_132, 48_133)" in runner
    assert runner.count('gpu="H100!"') == 1
    assert "run_host_sync_profile(" in runner
    assert "training_module._compile_model(model)" in runner
    assert "training_module._one_optimizer_step(" in runner
    assert "train_full_2b(" not in runner
    assert "full_training_authorized\": False" in runner
    assert "scientific_claim_authorized\": False" in runner

    assert "STEADY_WARMUP_STEPS = 5" in core
    assert "STEADY_MEASURED_STEPS = 4" in core
    assert "PROFILED_STEPS = 2" in core
    assert "with_stack=True" in core
    assert "events = list(prof.events())" in core
    assert "cpu_parent" in core
    assert "GROUPED_MM_HOST_SYNC_DOMINANT" in core
    assert "train_full_2b(" not in core

    assert "modal run --detach --timestamps modal_cortex_s_100m_host_sync_profile_v3.py" in workflow
    assert "source_sha" in workflow and "source_tree" in workflow and "harness_sha" in workflow
    assert "seed 8100" in workflow.lower()


def test_dispatch_order_and_durable_marker_precede_h100() -> None:
    runner = _repo_text("modal_cortex_s_100m_host_sync_profile_v3.py")
    main = runner[runner.index("def main("):]
    assert main.index("verify_zero_gpu.remote") < main.index("reserve_h100_dispatch.remote") < main.index("h100_profile.remote")

    reserve = runner[runner.index("def reserve_h100_dispatch("):runner.index("def _prove_triton_dispatch(")]
    h100 = runner[runner.index("def h100_profile("):runner.index("@app.local_entrypoint()")]
    assert "_atomic_write(marker_path, marker)" in reserve
    assert "volume.commit()" in reserve
    assert '"h100_allocation_started": False' in reserve
    assert 'marker["h100_allocation_started"] = True' in h100
