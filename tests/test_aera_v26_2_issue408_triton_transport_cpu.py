from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from tam_research.aera import AERAState
from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState
from tam_research.aera_hardware_core_v26_2_triton import (
    TritonFusedStateTransport,
    _allocate_state_like,
    fused_triton_transport_v26_2_protocol,
    triton_transport_available,
)
from tam_research.aera_v26_2_triton_transport_probe import (
    BATCH_SIZES,
    CALLS_PER_ROUND,
    CAPACITY,
    DESIGN_SEED,
    DTYPE_NAMES,
    D_MODEL,
    MAX_BATCH64_GEOMEAN_LATENCY_RATIO,
    MAX_BATCH64_ROW_LATENCY_RATIO,
    MAX_BATCH8_ROW_LATENCY_RATIO,
    MAX_KERNEL_RATIO,
    MEMORY_DIM,
    SELECTED_FRACTIONS,
    TIMED_ROUNDS,
    WARMUP_CALLS,
    cpu_contract_preflight,
    issue408_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_2_triton.py"
PROBE_PATH = ROOT / "tam_research" / "aera_v26_2_triton_transport_probe.py"
LAUNCHER_PATH = ROOT / "modal_aera_v26_2_triton_transport_app.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "aera-v26-2-triton-transport-l4.yml"


def _cpu_state(batch: int = 3) -> AERAState:
    g = torch.Generator().manual_seed(40801)
    return AERAState(
        stream=torch.randn(batch, D_MODEL, generator=g),
        memory=ContextualEpisodicMemoryState(
            keys=torch.randn(batch, CAPACITY, MEMORY_DIM, generator=g),
            values=torch.randn(batch, CAPACITY, MEMORY_DIM, generator=g),
            strengths=torch.rand(batch, CAPACITY, generator=g),
            valid=torch.rand(batch, CAPACITY, generator=g) > 0.5,
        ),
    )


def test_issue408_cpu_protocol_is_frozen_before_gpu_measurement():
    assert DESIGN_SEED == 406_408
    assert (D_MODEL, MEMORY_DIM, CAPACITY) == (200, 50, 48)
    assert BATCH_SIZES == (8, 64)
    assert SELECTED_FRACTIONS == (0.25, 0.50, 0.75, 1.00)
    assert DTYPE_NAMES == ("float32", "bfloat16")
    assert (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) == (10, 5, 200)
    assert MAX_KERNEL_RATIO == 0.25
    assert MAX_BATCH64_GEOMEAN_LATENCY_RATIO == 0.90
    assert MAX_BATCH64_ROW_LATENCY_RATIO == 1.05
    assert MAX_BATCH8_ROW_LATENCY_RATIO == 1.10

    check = cpu_contract_preflight()
    assert check["gpu_authorized_by_cpu_preflight"] is False
    assert check["synthetic_only"] is True
    assert check["scientific_seed_consumed"] is False

    protocol = issue408_protocol()
    assert protocol["research_issue"] == 408
    assert protocol["design_seed_is_scientific_seed"] is False
    assert protocol["dtypes"] == ["float32", "bfloat16"]
    assert protocol["max_kernel_ratio_each_row"] == 0.25
    assert protocol["max_batch64_geomean_latency_ratio_each_dtype"] == 0.90
    assert protocol["model_loaded"] is False
    assert protocol["checkpoint_loaded"] is False
    assert protocol["corpus_accessed"] is False
    assert protocol["training_performed"] is False
    assert protocol["scientific_seed_consumed"] is False
    assert protocol["end_to_end_systems_authorized"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["fresh_scientific_seed_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_v26_2_backend_protocol_freezes_two_kernel_zero_pack_boundary():
    protocol = fused_triton_transport_v26_2_protocol()
    assert protocol["research_issue"] == 408
    assert protocol["source_main"] == "27fc272e495bc3c125f7a1786c09581557670b3d"
    assert protocol["select_triton_launches_target"] == 1
    assert protocol["merge_triton_launches_target"] == 1
    assert protocol["auxiliary_row_map_kernel"] is False
    assert protocol["persistent_pack_state"] is False
    assert protocol["persistent_cache"] is False
    assert protocol["training_backend_authorized"] is False
    assert protocol["model_integration_authorized"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["end_to_end_systems_authorized"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_v26_2_module_imports_on_cpu_and_backend_fails_closed_without_cuda():
    # CI is intentionally CPU-only. Triton may be absent or importable, but a fused
    # backend must never silently fall back to CPU/componentwise execution.
    assert triton_transport_available() is False
    with pytest.raises(RuntimeError):
        TritonFusedStateTransport()


def test_v26_2_transient_output_allocation_preserves_five_semantic_tensors():
    state = _cpu_state(batch=3)
    output = _allocate_state_like(state, 2)
    assert isinstance(output.memory, ContextualEpisodicMemoryState)
    assert output.stream.shape == (2, D_MODEL)
    assert output.memory.keys.shape == (2, CAPACITY, MEMORY_DIM)
    assert output.memory.values.shape == (2, CAPACITY, MEMORY_DIM)
    assert output.memory.strengths.shape == (2, CAPACITY)
    assert output.memory.valid.shape == (2, CAPACITY)
    assert output.stream.dtype == state.stream.dtype
    assert output.memory.keys.dtype == state.memory.keys.dtype
    assert output.memory.values.dtype == state.memory.values.dtype
    assert output.memory.strengths.dtype == state.memory.strengths.dtype
    assert output.memory.valid.dtype is torch.bool
    assert output.stream.data_ptr() != state.stream.data_ptr()
    assert output.memory.keys.data_ptr() != state.memory.keys.data_ptr()


def test_v26_2_backend_source_has_exactly_two_triton_kernels_and_no_pack_path():
    source = BACKEND_PATH.read_text()
    assert source.count("@triton.jit") == 2
    assert "def _fused_select_kernel(" in source
    assert "def _fused_merge_kernel(" in source
    assert "_fused_select_kernel[(selected,)]" in source
    assert "_fused_merge_kernel[(batch,)]" in source
    assert "torch.cat(" not in source
    assert "torch.stack(" not in source
    assert "pack_ephemeral_epi_state" not in source
    assert "select_packed_epi_state" not in source
    assert "merge_packed_epi_state" not in source
    assert "register_buffer" not in source
    assert "nn.Parameter" not in source


def test_v26_2_merge_kernel_performs_bounded_index_lookup_inside_same_launch():
    source = BACKEND_PATH.read_text()
    merge_source = source.split("def _fused_merge_kernel(", 1)[1].split(
        "def _next_power_of_two", 1
    )[0]
    assert "INDEX_BLOCK: tl.constexpr" in merge_source
    assert "selected_ids = tl.load(" in merge_source
    assert "selected_ids == base_row" in merge_source
    assert "selected_position = tl.max(" in merge_source
    assert "run_idx_ptr" in merge_source
    assert "torch.empty" not in merge_source
    assert "row_map" not in merge_source.lower()


def test_issue408_probe_is_synthetic_and_uses_actual_merged_transport_backends():
    source = PROBE_PATH.read_text()
    assert "TorchComponentwiseStateTransport" in source
    assert "TritonFusedStateTransport" in source
    assert "reference.select(case.base, case.run_idx)" in source
    assert "candidate.select(case.base, case.run_idx)" in source
    assert "reference.merge(case.base, case.update, case.run_idx)" in source
    assert "candidate.merge(case.base, case.update, case.run_idx)" in source
    assert "torch.cuda.Event(enable_timing=True)" in source
    assert "torch.profiler.ProfilerActivity.CUDA" in source
    assert "torch.load(" not in source
    assert ".load_state_dict(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "seed8471" not in source.lower()


def test_issue408_launcher_is_unique_l4_duplicate_safe_and_no_training_path():
    source = LAUNCHER_PATH.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue408-triton-transport/result.json"' in source
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in source
    assert "result_path.exists()" in source
    assert "refusing duplicate issue408 Triton transport run" in source
    assert "AERA_V26_ISSUE408_TRITON_TRANSPORT_RESULT_JSON=" in source
    assert "run_triton_transport_probe" in source
    assert "torch.load(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "seed8471" not in source.lower()


def test_issue408_workflow_is_one_attempt_issue_only_and_bound_to_exact_main():
    source = WORKFLOW_PATH.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-2-triton-transport-l4]" in source
    assert 'if [ "${GITHUB_RUN_ATTEMPT}" != "1" ]; then' in source
    assert "Expected exactly one issue408 Triton transport trigger" in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor 27fc272e495bc3c125f7a1786c09581557670b3d HEAD" in source
    assert source.count("modal run modal_aera_v26_2_triton_transport_app.py") == 1
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source


def test_issue408_workflow_permissions_cover_guard_reads_without_broad_writes():
    source = WORKFLOW_PATH.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests: read" in permissions
    assert "contents: write" not in permissions
    assert "actions: write" not in permissions
    assert "pull-requests: write" not in permissions


def test_v26_2_backend_class_does_not_hold_tensor_or_session_state_by_contract():
    # On CPU we cannot instantiate the CUDA-only class, so inspect its declared
    # methods/attributes. Runtime instance state is limited to integer geometry.
    source = inspect.getsource(TritonFusedStateTransport)
    assert "self.max_batch = int(max_batch)" in source
    assert "self.index_block = _next_power_of_two(self.max_batch)" in source
    stripped = source.replace("self.max_batch", "").replace(
        "self.index_block", ""
    ).replace("self._validate_index", "")
    assert "self." not in stripped
