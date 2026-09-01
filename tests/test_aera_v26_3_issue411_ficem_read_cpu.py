from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from tam_research.aera_hardware_core_v24 import (
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
)
from tam_research.aera_hardware_core_v26_3_ficem_read_triton import (
    TritonFICEMReadBackend,
    fused_ficem_read_v26_3_protocol,
    triton_ficem_read_available,
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
    cpu_contract_preflight,
    issue411_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE_PATH = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
LAUNCHER_PATH = ROOT / "modal_aera_v26_3_ficem_read_app.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "aera-v26-3-ficem-read-l4.yml"


def test_issue411_cpu_protocol_is_frozen_before_gpu_measurement():
    assert DESIGN_SEED == 408_411
    assert (D_MODEL, MEMORY_DIM, CAPACITY, TIME) == (200, 50, 48, 256)
    assert BATCH_SIZES == (8, 64)
    assert DTYPE_NAMES == ("float32", "bfloat16")
    assert VALIDITY_KINDS == ("mixed", "full")
    assert (WARMUP_CALLS, TIMED_ROUNDS, CALLS_PER_ROUND) == (10, 5, 100)
    assert (FP32_ATOL, FP32_RTOL) == (1e-5, 1e-5)
    assert (BF16_ATOL, BF16_RTOL) == (1e-2, 1e-2)
    assert MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert MAX_ROW_LATENCY_RATIO == 1.05
    assert MAX_FULL_EVENT_RATIO == 0.75

    check = cpu_contract_preflight()
    assert check["gpu_authorized_by_cpu_preflight"] is False
    assert check["synthetic_only"] is True
    assert check["scientific_seed_consumed"] is False

    protocol = issue411_protocol()
    assert protocol["research_issue"] == 411
    assert protocol["design_seed_is_scientific_seed"] is False
    assert protocol["dtypes"] == ["float32", "bfloat16"]
    assert protocol["max_geomean_latency_ratio_each_dtype"] == 0.90
    assert protocol["max_row_latency_ratio"] == 1.05
    assert protocol["max_full_read_cuda_event_ratio"] == 0.75
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


def test_v26_3_backend_protocol_freezes_exact_read_only_boundary():
    protocol = fused_ficem_read_v26_3_protocol()
    assert protocol["research_issue"] == 411
    assert protocol["source_main"] == "8227367a9c53cbb2c3ad14be426f4e9d95f46c89"
    assert (protocol["capacity"], protocol["memory_dim"]) == (48, 50)
    assert protocol["read_top_k"] == READ_TOP_K == 4
    assert protocol["read_temperature"] == READ_TEMPERATURE == 0.10
    assert protocol["min_strength"] == MIN_STRENGTH == 1e-4
    assert protocol["read_tail_triton_launches_target"] == 1
    assert protocol["address_projection_changed"] is False
    assert protocol["key_normalization_changed"] is False
    assert protocol["similarity_einsum_changed"] is False
    assert protocol["learned_out_projection_changed"] is False
    assert protocol["write_backend_changed"] is False
    assert protocol["training_backend_changed"] is False
    assert protocol["known_empty_fastpath_preserved"] is True
    assert protocol["same_call_query_key_reuse_preserved"] is True
    assert protocol["persistent_state_changed"] is False
    assert protocol["persistent_cache"] is False
    assert protocol["persistent_packed_state"] is False
    assert protocol["gpu_authorized_by_module"] is False
    assert protocol["scientific_training_authorized"] is False
    assert protocol["end_to_end_systems_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_v26_3_module_imports_on_cpu_and_backend_fails_closed_without_cuda():
    assert triton_ficem_read_available() is False
    with pytest.raises(RuntimeError):
        TritonFICEMReadBackend()


def test_v26_3_backend_has_exactly_one_direct_triton_read_tail_kernel():
    source = BACKEND_PATH.read_text()
    assert source.count("@triton.jit") == 1
    assert "def _ficem_read_tail_kernel(" in source
    assert "_ficem_read_tail_kernel[(batch * time,)]" in source
    assert "SLOT_BLOCK=64" in source
    assert "DIM_BLOCK=64" in source
    assert "num_warps=4" in source
    assert "torch.topk(" not in source
    assert "torch.softmax(" not in source
    assert ".gather(" not in source
    assert "torch.cat(" not in source
    assert "torch.stack(" not in source
    assert "register_buffer" not in source
    assert "nn.Parameter" not in source


def test_v26_3_kernel_source_preserves_frozen_top4_strength_mask_and_fp32_softmax_math():
    source = BACKEND_PATH.read_text()
    kernel = source.split("def _ficem_read_tail_kernel(", 1)[1].split(
        "def triton_ficem_read_available", 1
    )[0]
    assert "tl.log(tl.maximum(strengths, MIN_STRENGTH))" in kernel
    assert "READ_TEMPERATURE" in kernel
    assert "-float(\"inf\")" in kernel
    assert kernel.count("tl.argmax(") == 4
    assert "-1.0e9" in kernel
    assert "tl.exp(" in kernel
    assert "tl.maximum(valid_weight_sum, 1.0e-9)" in kernel
    assert "weight0 * value0" in kernel
    assert "weight3 * value3" in kernel


def test_v26_3_backend_keeps_projection_similarity_outside_kernel_and_reuse_exact():
    source = inspect.getsource(TritonFICEMReadBackend.read)
    training_guard = source.index("if torch.is_grad_enabled() or memory.differentiable_pretraining")
    empty_guard = source.index("if _known_empty_hint(state)")
    projection = source.index("memory.address_factors")
    key_norm = source.index("F.normalize(state.keys, dim=-1)")
    similarity = source.index('torch.einsum("btd,bsd->bts", query, keys)')
    fused_tail = source.index("fused_ficem_read_tail(")
    output_projection = source.index("recalled=memory.out(recalled)")
    assert training_guard < empty_guard < projection < key_norm < similarity < fused_tail < output_projection
    assert "projected_query=query" in source
    assert "normalized_old_keys=keys" in source


def test_v26_3_training_and_write_methods_delegate_to_exact_reference_backend():
    read_source = inspect.getsource(TritonFICEMReadBackend.read)
    update_source = inspect.getsource(TritonFICEMReadBackend.update)
    projected_source = inspect.getsource(TritonFICEMReadBackend.update_from_projected)
    assert "return self._reference.read(" in read_source
    assert "return self._reference.update(" in update_source
    assert "return self._reference.update_from_projected(" in projected_source
    class_source = inspect.getsource(TritonFICEMReadBackend)
    assert "self._reference = TorchFICEMReferenceBackend()" in class_source
    assert "self." not in class_source.replace("self._reference", "")


def test_issue411_probe_is_standalone_synthetic_and_uses_real_reference_and_candidate():
    source = PROBE_PATH.read_text()
    assert "TorchFICEMReferenceBackend()" in source
    assert "TritonFICEMReadBackend()" in source
    assert "CoalescedFICEMMemory(compact)" in source
    assert "FactorizedIdentityContextEpisodicMemory" in source
    assert "run_ficem_read_probe" in source
    assert "torch.cuda.Event(enable_timing=True)" in source
    assert "torch.profiler.ProfilerActivity.CUDA" in source
    assert "validity_kind == \"full\"" in source
    assert "near_tie_correctness" in source
    assert "known_empty_case" in source
    assert "torch.load(" not in source
    assert ".load_state_dict(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "seed8471" not in source.lower()


def test_issue411_launcher_is_unique_l4_duplicate_safe_and_no_training_path():
    source = LAUNCHER_PATH.read_text()
    assert 'RESULT_PATH = "/vol/aera-v26/issue411-ficem-read/result.json"' in source
    assert 'gpu="L4"' in source
    assert source.count('gpu="L4"') == 1
    assert "MAX_GPU_SECONDS = 300" in source
    assert "result_path.exists()" in source
    assert "refusing duplicate issue411 FICEM read run" in source
    assert "AERA_V26_ISSUE411_FICEM_READ_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE411_FICEM_READ_SUMMARY_JSON=" in source
    assert "run_ficem_read_probe" in source
    assert "torch.load(" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "seed8471" not in source.lower()


def test_issue411_workflow_is_one_attempt_issue_only_and_bound_to_exact_main():
    source = WORKFLOW_PATH.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-3-ficem-read-l4]" in source
    assert 'if [ "${GITHUB_RUN_ATTEMPT}" != "1" ]; then' in source
    assert "Expected exactly one issue411 FICEM read trigger" in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert "git merge-base --is-ancestor 8227367a9c53cbb2c3ad14be426f4e9d95f46c89 HEAD" in source
    assert source.count("modal run modal_aera_v26_3_ficem_read_app.py") == 1
    assert "AERA_V26_ISSUE411_FICEM_READ_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE411_FICEM_READ_SUMMARY_JSON=" in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source


def test_issue411_workflow_permissions_are_narrow_and_reporting_is_non_authoritative():
    source = WORKFLOW_PATH.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "contents: write" not in permissions
    assert "actions: write" not in permissions
    assert "pull-requests: write" not in permissions
    assert "Record issue411 result (best effort)" in source
    assert "continue-on-error: true" in source
    assert "Durable Modal result + authoritative result marker" in source
