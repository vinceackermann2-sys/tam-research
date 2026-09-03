from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_hardware_core_v26_3_ficem_read_triton as repair5
from tam_research import aera_hardware_core_v26_6_ficem_write_materialize_cast as v26_6
from tam_research import aera_hardware_core_v26_7_ficem_read_mixed_dtype as v26_7
from tam_research import aera_hardware_core_v26_8_ficem_read_mixed_strength_precision as successor


ROOT = Path(__file__).resolve().parents[1]
SUCCESSOR_PATH = (
    ROOT / "tam_research" / "aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py"
)
V26_7_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_7_ficem_read_mixed_dtype.py"
REPAIR5_PATH = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
V26_6_WRITE_PATH = (
    ROOT / "tam_research" / "aera_hardware_core_v26_6_ficem_write_materialize_cast.py"
)
ISSUE553_PROBE_PATH = (
    ROOT / "tam_research" / "aera_v26_7_issue553_ficem_read_mixed_dtype_probe.py"
)
ISSUE553_LAUNCHER_PATH = ROOT / "modal_aera_v26_7_issue553_ficem_read_mixed_dtype_l4_app.py"
ISSUE553_WORKFLOW_PATH = (
    ROOT / ".github" / "workflows" / "aera-v26-7-issue553-ficem-read-mixed-dtype-l4.yml"
)
HISTORICAL_PROBE_PATH = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe.py"
REPAIR5_PROBE_PATH = ROOT / "tam_research" / "aera_v26_3_ficem_read_probe_repair5.py"
ISSUE530_SYSTEMS_PATH = ROOT / "tam_research" / "aera_v26_6_issue530_end_to_end_systems.py"
V26_INTERFACE_PATH = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE_PATH = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"


def _git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def _kernel_source() -> str:
    source = SUCCESSOR_PATH.read_text()
    return source.split("def _mixed_strength_precision_kernel(", 1)[1].split(
        "def _validate_v26_8_mixed_inputs(", 1
    )[0]


def test_issue556_frozen_authoritative_failure_and_lineage() -> None:
    assert successor.RESEARCH_ISSUE == 556
    assert successor.SOURCE_MAIN == "e4866dd6d4556fc090b556c09ae49fcf4c59105f"
    assert successor.SOURCE_TREE == "86af0b1dd4e6d3fd2cdb6412460b7e4f5cbbd6ff"

    assert successor.ISSUE553_TRIGGER == 555
    assert successor.ISSUE553_RUN == 33727540468
    assert successor.ISSUE553_JOB == 100559866985
    assert successor.ISSUE553_RESULT_PATH == (
        "/vol/aera-v26/issue553-ficem-read-mixed-dtype/result.json"
    )
    assert successor.ISSUE553_RESULT_SHA256 == (
        "009af31baf70e46eb93b6e7489d62f356a02b727521d3fabe4a7dab2dcf5ab47"
    )
    assert successor.ISSUE553_DECISION == "FAIL"

    assert successor.ISSUE545_TRIGGER == 550
    assert successor.ISSUE545_RUN == 33686037672
    assert successor.ISSUE545_JOB == 100433658768
    assert successor.ISSUE545_FAILURE == "FICEM read-tail floating dtypes must match"

    assert successor.ISSUE479_TRIGGER == 484
    assert successor.ISSUE479_RUN == 33618950619
    assert successor.ISSUE479_JOB == 100211244996

    assert successor.ISSUE529_TRIGGER == 529
    assert successor.ISSUE529_RUN == 33680028132
    assert successor.ISSUE529_JOB == 100414089065
    assert successor.ISSUE529_RESULT_SHA256 == (
        "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
    )


def test_issue556_frozen_blobs_are_unchanged() -> None:
    frozen = {
        V26_7_PATH: successor.V26_7_BACKEND_BLOB,
        REPAIR5_PATH: successor.REPAIR5_READ_BLOB,
        V26_6_WRITE_PATH: successor.V26_6_WRITE_BLOB,
        ISSUE553_PROBE_PATH: successor.ISSUE553_PROBE_BLOB,
        ISSUE553_LAUNCHER_PATH: successor.ISSUE553_LAUNCHER_BLOB,
        ISSUE553_WORKFLOW_PATH: successor.ISSUE553_WORKFLOW_BLOB,
        HISTORICAL_PROBE_PATH: successor.HISTORICAL_PROBE_BLOB,
        REPAIR5_PROBE_PATH: successor.REPAIR5_PROBE_BLOB,
        ISSUE530_SYSTEMS_PATH: successor.ISSUE530_SYSTEMS_BLOB,
        V26_INTERFACE_PATH: successor.V26_INTERFACE_BLOB,
        STABLE_REFERENCE_PATH: successor.STABLE_REFERENCE_BLOB,
    }
    for path, expected in frozen.items():
        assert _git_blob_sha(path) == expected, path


def test_issue556_dispatch_matrix_is_exact_and_narrow() -> None:
    dispatch = successor.read_dispatch_kind

    assert dispatch(torch.float32, torch.float32, torch.float32) == "historical-repair5"
    assert dispatch(torch.bfloat16, torch.bfloat16, torch.bfloat16) == "historical-repair5"
    assert dispatch(torch.float16, torch.float16, torch.float16) == "historical-repair5"

    assert (
        dispatch(torch.bfloat16, torch.float32, torch.float32)
        == "mixed-strength-precision-v26.8"
    )
    assert (
        dispatch(torch.float32, torch.bfloat16, torch.bfloat16)
        == "mixed-strength-precision-v26.8"
    )

    for layout in (
        (torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.float16, torch.float16),
        (torch.bfloat16, torch.float16, torch.float16),
        (torch.float32, torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (torch.float32, torch.bfloat16, torch.float32),
        (torch.float64, torch.float64, torch.float64),
    ):
        assert dispatch(*layout) == "unsupported"


def test_same_dtype_calls_delegate_to_exact_historical_repair5_helper() -> None:
    assert successor.fused_ficem_read_tail is repair5.fused_ficem_read_tail
    source = inspect.getsource(successor.fused_ficem_read_tail_v26_8)
    historical = source.split('if dispatch == "historical-repair5":', 1)[1].split(
        'if dispatch != "mixed-strength-precision-v26.8":', 1
    )[0]
    assert "return fused_ficem_read_tail(" in historical
    assert "_mixed_strength_precision_kernel" not in historical


def test_successor_defines_exactly_one_new_mixed_only_triton_kernel() -> None:
    source = SUCCESSOR_PATH.read_text()
    assert source.count("@triton.jit") == 1
    assert "def _mixed_strength_precision_kernel(" in source
    assert "_mixed_strength_precision_kernel[(batch * time,)]" in source
    assert "SLOT_BLOCK=64" in source
    assert "DIM_BLOCK=64" in source
    assert "num_warps=4" in source
    assert "CAPACITY=capacity" in source
    assert "MEMORY_DIM=memory_dim" in source
    assert "READ_TOP_K=READ_TOP_K" in source
    assert "READ_TEMPERATURE=READ_TEMPERATURE" in source
    assert "MIN_STRENGTH=MIN_STRENGTH" in source


def test_compute_and_durable_strength_precision_controls_are_separate() -> None:
    kernel = _kernel_source()
    launch = inspect.getsource(successor.fused_ficem_read_tail_v26_8)

    assert "IS_BF16_COMPUTE: tl.constexpr" in kernel
    assert "DURABLE_IS_BF16: tl.constexpr" in kernel
    assert "IS_BF16_COMPUTE=similarity.dtype is torch.bfloat16" in launch
    assert "DURABLE_IS_BF16=strengths.dtype is torch.bfloat16" in launch
    assert "IS_BF16=similarity.dtype" not in launch


def test_bf16_compute_fp32_strength_keeps_strength_fp32_through_log() -> None:
    kernel = _kernel_source()
    compute_branch = kernel.split("if IS_BF16_COMPUTE:", 1)[1].split(
        "else:\n            # #556's only non-BF16-compute mixed layout", 1
    )[0]

    assert "similarity_visible = similarity.to(tl.bfloat16)" in compute_branch
    assert "if DURABLE_IS_BF16:" in compute_branch
    assert "strength_bias = tl.log(clamped_strengths.to(tl.float32))" in compute_branch
    assert "else:\n                strength_bias = tl.log(clamped_strengths)" in compute_branch
    # The FP32-durable else branch must not insert a BF16 strength visibility cast.
    fp32_durable_branch = compute_branch.split("else:\n                strength_bias", 1)[1]
    assert "clamped_strengths.to(tl.bfloat16)" not in fp32_durable_branch


def test_fp32_compute_bf16_strength_has_explicit_bf16_bias_visibility() -> None:
    kernel = _kernel_source()
    non_bf16_compute = kernel.split(
        "# #556's only non-BF16-compute mixed layout", 1
    )[1].split("logits = tl.where", 1)[0]

    assert "similarity_visible = similarity.to(tl.float32)" in non_bf16_compute
    assert "if DURABLE_IS_BF16:" in non_bf16_compute
    assert "strength_bias_visible = tl.log(" in non_bf16_compute
    assert ").to(tl.bfloat16)" in non_bf16_compute
    assert "strength_bias = strength_bias_visible.to(tl.float32)" in non_bf16_compute


def test_softmax_weight_visibility_is_controlled_only_by_compute_dtype() -> None:
    kernel = _kernel_source()
    weight_section = kernel.split("softmax_sum =", 1)[1].split("dim_offsets =", 1)[0]

    assert "if IS_BF16_COMPUTE:" in weight_section
    assert "weight0_visible = soft0.to(tl.bfloat16)" in weight_section
    assert "weight1_visible = soft1.to(tl.bfloat16)" in weight_section
    assert "weight2_visible = soft2.to(tl.bfloat16)" in weight_section
    assert "weight3_visible = soft3.to(tl.bfloat16)" in weight_section
    assert "if DURABLE_IS_BF16" not in weight_section


def test_selected_value_source_visibility_and_fp32_reduction_are_in_kernel() -> None:
    kernel = _kernel_source()
    value_section = kernel.split("dim_offsets =", 1)[1]

    assert "if DURABLE_IS_BF16:" in value_section
    assert "value0_visible = value0.to(tl.bfloat16)" in value_section
    assert "value0_visible = value0.to(tl.float32)" in value_section
    for index in range(4):
        assert (
            f"product{index} = weight{index}.to(tl.float32) * "
            f"value{index}_visible.to(tl.float32)"
        ) in value_section
    assert "recalled = product0 + product1 + product2 + product3" in value_section


def test_mixed_wrapper_has_no_host_cast_copy_or_second_tail_kernel() -> None:
    source = inspect.getsource(successor.fused_ficem_read_tail_v26_8)
    mixed = source.split('if dispatch != "mixed-strength-precision-v26.8":', 1)[1]

    assert ".to(" not in mixed
    assert "_to_copy" not in mixed
    assert "copy_(" not in mixed
    assert "torch.cat" not in mixed
    assert "torch.stack" not in mixed
    assert mixed.count("_mixed_strength_precision_kernel[") == 1
    assert "dtype=torch.float32" in mixed


def test_backend_preserves_v26_6_write_and_reference_training_path() -> None:
    backend = successor.StrengthPrecisionTritonFICEMReadWriteBackend
    assert issubclass(backend, v26_7.MixedDtypeTritonFICEMReadWriteBackend)
    assert issubclass(backend, v26_6.MaterializeCastTritonFICEMReadWriteBackend)
    assert (
        backend._inference_update_from_projected
        is v26_6.MaterializeCastTritonFICEMReadWriteBackend._inference_update_from_projected
    )

    source = inspect.getsource(backend.read)
    assert "if torch.is_grad_enabled() or memory.differentiable_pretraining:" in source
    assert "return self._reference.read(" in source
    assert "_known_empty_hint(state)" in source
    assert "fused_ficem_read_tail_v26_8(" in source
    assert "memory.out(recalled)" in source


def test_issue556_protocol_records_failure_partition_and_keeps_higher_auth_false() -> None:
    protocol = successor.mixed_strength_precision_v26_8_protocol()

    assert protocol["version"] == "aera-v26.8-ficem-read-mixed-strength-precision"
    assert protocol["research_issue"] == 556
    assert protocol["issue553_consumed"] is True
    assert protocol["issue553_decision"] == "FAIL"
    assert protocol["issue553_historical_surface_pass"] is True
    assert protocol["issue553_mixed_regular_rows_pass"] == 0
    assert protocol["issue553_mixed_regular_rows_total"] == 8
    assert protocol["issue553_mixed_topology_pass"] is True
    assert protocol["issue553_mixed_near_tie_pass"] is True
    assert protocol["issue553_mixed_known_empty_pass"] is True

    assert protocol["same_dtype_dispatch"] == "historical-repair5"
    assert protocol["same_dtype_arithmetic_changed_by_v26_8"] is False
    assert protocol["same_dtype_kernel_changed_by_v26_8"] is False
    assert protocol["mixed_new_triton_kernels"] == 1
    assert protocol["mixed_tail_triton_launches_target"] == 1
    assert protocol["mixed_compute_precision_control_separate"] is True
    assert protocol["mixed_durable_strength_precision_control_separate"] is True
    assert protocol["mixed_strengths_values_dtype_equality_required"] is True
    assert protocol["mixed_arbitrary_strengths_values_mixing_authorized"] is False
    assert protocol["mixed_fp16_authorized"] is False
    assert protocol["bf16_compute_fp32_strength_prelog_bf16_cast"] is False
    assert protocol["fp32_compute_bf16_strength_bias_bf16_visibility"] is True
    assert protocol["softmax_weight_visibility_controlled_by_compute_dtype"] is True
    assert protocol["mixed_selected_value_source_dtype_preserved"] is True
    assert protocol["mixed_recalled_pre_out_dtype"] == "float32"
    assert protocol["mixed_host_pre_tail_cast_kernels"] == 0
    assert protocol["write_backend_changed_by_v26_8"] is False
    assert protocol["training_backend_changed_by_v26_8"] is False

    for key in (
        "gpu_authorized_by_issue556",
        "mixed_dtype_read_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
        "scientific_seed_consumed",
    ):
        assert protocol[key] is False


def test_issue556_cpu_preflight_is_zero_gpu_and_exact() -> None:
    preflight = successor.cpu_contract_preflight_issue556()
    assert preflight["research_issue"] == 556
    assert preflight["source_main"] == successor.SOURCE_MAIN
    assert preflight["source_tree"] == successor.SOURCE_TREE
    assert preflight["gpu_authorized_by_cpu_preflight"] is False
    assert preflight["synthetic_only"] is True
    assert preflight["scientific_seed_consumed"] is False


def test_issue556_two_file_scope_contains_no_gpu_or_systems_execution_surface() -> None:
    source = SUCCESSOR_PATH.read_text()
    assert "gpu=\"L4\"" not in source
    assert "modal.App" not in source
    assert "workflow_dispatch" not in source
    assert "run_end_to_end_systems" not in source
    assert "optimizer" not in source.lower()
    assert "backward(" not in source
    assert "checkpoint" not in inspect.getsource(
        successor.StrengthPrecisionTritonFICEMReadWriteBackend.read
    ).lower()
