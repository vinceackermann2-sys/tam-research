from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

from tam_research import aera_v26_5_end_to_end_systems as systems
from tam_research import aera_v25_post8471_triage as triage
from tam_research.aera_hardware_core_v26 import (
    HardwareAwareAERATextLMV26,
    TorchFICEMReferenceBackend,
)
from tam_research.aera_hardware_core_v26_4_ficem_write_triton import (
    TritonFICEMReadWriteBackend,
)

ROOT = Path(__file__).resolve().parents[1]
V26 = ROOT / "tam_research" / "aera_hardware_core_v26.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
WRITE_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
MODULE = ROOT / "tam_research" / "aera_v26_5_end_to_end_systems.py"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue501_freezes_source_and_primitive_pass_evidence() -> None:
    assert systems.RESEARCH_ISSUE == 501
    assert systems.SOURCE_MAIN == "148bde16c4995877798a874154f0f18363c406f4"
    assert _blob(V26) == "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
    assert _blob(READ_BACKEND) == "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
    assert _blob(WRITE_BACKEND) == "e54570292489bd17570038dca7518419ac00418c"
    assert _blob(STABLE_REFERENCE) == "4e336b6e1a6238dac782fa320751d68281493ee1"
    assert systems.READ_PASS_RUN == 33618950619
    assert systems.READ_PASS_JOB == 100211244996
    assert systems.READ_PASS_RESULT_SHA256 == (
        "a3b5a85b1de26a3e76d1908753860c7f6105195f63a2faa18a9bdd62db549dac"
    )
    assert systems.WRITE_PASS_RUN == 33651216734
    assert systems.WRITE_PASS_JOB == 100318422299
    assert systems.WRITE_PASS_RESULT_SHA256 == (
        "64105bb08a65f7d3d55528ed35c5b8e77edc55c2c2ec818765896a6a4d16ea8b"
    )


def test_issue501_inherits_issue381_workload_and_deployment_thresholds_exactly() -> None:
    assert systems.SOURCE_CHECKPOINT_SEED == 8471
    assert systems.CHECKPOINT_RELATIVE_DIR == "/vol/aera-real-language/v25-dev-seed8471"
    assert systems.SYSTEM_BATCH_SIZES == (8, 64)
    assert (systems.SYSTEM_WARMUP_CALLS, systems.SYSTEM_TIMED_CALLS_PER_ROUND, systems.SYSTEM_ROUNDS) == (
        3,
        20,
        5,
    )
    assert systems.BATCH8_MIN_FULL_SPEED_RATIO == 0.25
    assert systems.BATCH64_MIN_FULL_SPEED_RATIO == 1.25
    assert (systems.INTEGRATED_ATOL, systems.INTEGRATED_RTOL) == (1e-2, 1e-2)
    assert systems.EXPECTED_STATE_BYTES == 77_760
    assert (
        systems.EXPECTED_SELECTED_WRITES,
        systems.EXPECTED_CANDIDATES,
        systems.EXPECTED_VECTOR_UPDATES,
    ) == (16, 255, 1)
    assert systems.MAX_GPU_SECONDS == 600
    protocol = systems.systems_protocol()
    assert protocol["random_token_seed_rule"] == "138471 + 10000 + batch_size"
    assert protocol["timing_order"] == "rotated interleaved conditions per issue381"
    assert protocol["timing_clock"] == "CUDA events with synchronize before/after"
    assert protocol["hard"] is True
    assert protocol["route_mode"] == "hard_sparse"
    assert protocol["physically_real_sparse_required"] is True
    assert protocol["dense_masked_sparse_credit"] is False


def test_issue501_cpu_preflight_is_zero_gpu_and_preserves_frozen_protocols() -> None:
    check = systems.cpu_contract_preflight()
    assert check["gpu_authorized_by_cpu_preflight"] is False
    assert check["checkpoint_loaded"] is False
    assert check["training_performed"] is False
    assert check["scientific_seed_consumed"] is False
    assert check["v26_protocol"]["coalesced_optional_state"] is True
    assert check["v26_protocol"]["real_language_selected_writes"] == 16
    assert (
        check["v26_protocol"]["persistent_state_bytes_real_language_four_stage_memory_dim50"]
        == 77_760
    )
    assert check["read_write_protocol"]["backend_name"] == TritonFICEMReadWriteBackend.name
    assert check["read_write_protocol"]["write_threshold_input_dtype_visibility_repair1"] is True
    assert check["read_write_protocol"]["write_tail_triton_launches_target"] == 2
    preflight_source = inspect.getsource(systems.cpu_contract_preflight)
    assert "TritonFICEMReadWriteBackend()" not in preflight_source
    assert ".cuda(" not in preflight_source
    assert "torch.cuda" not in preflight_source


def test_issue501_builds_state_dict_identical_v26_twins_and_installs_backend_only_on_candidate() -> None:
    source = MODULE.read_text()
    assert "reference = _build_v26(aera_payload, device)" in source
    assert "candidate = _build_v26(aera_payload, device)" in source
    assert "_install_candidate_backend(candidate)" in source
    install_source = inspect.getsource(systems._install_candidate_backend)
    assert "memory._execution_backend = TritonFICEMReadWriteBackend()" in install_source
    assert "memory.execution_backend_name != TritonFICEMReadWriteBackend.name" in install_source
    load_source = inspect.getsource(systems.load_models)
    assert "reference.state_dict().keys() != candidate.state_dict().keys()" in load_source
    assert "_parameter_schema(reference) != _parameter_schema(candidate)" in load_source
    assert "memory.execution_backend_name != TorchFICEMReferenceBackend.name" in load_source
    assert issubclass(HardwareAwareAERATextLMV26, object)


def test_issue501_uses_actual_hard_sparse_selected_population_execution() -> None:
    triage_forward = inspect.getsource(triage._model_forward)
    assert "hard=True" in triage_forward
    assert 'route_mode="hard_sparse"' in triage_forward
    v26_source = V26.read_text()
    assert "run_idx = (gate[:, 0] >= 0.5).nonzero" in v26_source
    assert "x.index_select(0, run_idx)" in v26_source
    assert "x.index_copy(0, run_idx, selected_y)" in v26_source
    assert "select_packed_epi_state(base_packed, run_idx)" in v26_source
    assert "merge_packed_epi_state(base_packed, update_packed, run_idx)" in v26_source
    sparse_source = inspect.getsource(systems._physical_sparse_proof)
    assert "any(fraction < 1.0 for fraction in fractions)" in sparse_source
    assert "coalesced_float_state_select_calls > 0" in sparse_source
    assert "coalesced_float_state_merge_calls > 0" in sparse_source
    assert "read_calls > 0" in sparse_source
    assert "update_calls + projected_update_calls" in sparse_source
    assert '"dense_masked_sparse_credit": False' in sparse_source


def test_issue501_integrated_equivalence_keeps_discrete_routing_exact_and_bf16_budget() -> None:
    route_source = inspect.getsource(systems._route_signature)
    assert ".ge(0.5).cpu()" in route_source
    run_source = inspect.getsource(systems.run_end_to_end_systems)
    assert "torch.equal(a, b)" in run_source
    assert "triage._routing_accounting(reference_output, batch_size)" in run_source
    logit_source = inspect.getsource(systems._logit_equivalence)
    assert "torch.allclose" in logit_source
    assert "atol=INTEGRATED_ATOL" in logit_source
    assert "rtol=INTEGRATED_RTOL" in logit_source
    state_source = inspect.getsource(systems._state_equivalence)
    assert "validity_exact" in state_source
    assert "dtype_device_shape_exact" in state_source
    assert "max_continuous_abs" in state_source


def test_issue501_frozen_timing_is_rotated_interleaved_and_decision_is_full_system() -> None:
    source = inspect.getsource(systems._timed_summaries)
    assert "SYSTEM_WARMUP_CALLS" in source
    assert "SYSTEM_ROUNDS" in source
    assert "SYSTEM_TIMED_CALLS_PER_ROUND" in source
    assert "rotated = names[round_index:] + names[:round_index]" in source
    assert "triage._cuda_timed_call" in source
    run_source = inspect.getsource(systems.run_end_to_end_systems)
    assert '"transformer": transformer_call' in run_source
    assert '"v26_torch_reference_full_ficem": reference_full_call' in run_source
    assert '"v26_4_triton_full_ficem": candidate_full_call' in run_source
    assert "candidate_tps / transformer_tps" in run_source
    assert "full_speed_ratio >= required_speed_ratio" in run_source
    assert "candidate_ms <= reference_ms" in run_source


def test_issue501_checkpoint_is_read_only_hashed_before_after_and_never_retrained() -> None:
    source = MODULE.read_text()
    assert 'root / "aera.pt"' in source
    assert 'root / "transformer.pt"' in source
    assert 'aera_payload.get("seed") != SOURCE_CHECKPOINT_SEED' in source
    assert 'transformer_payload.get("seed") != SOURCE_CHECKPOINT_SEED' in source
    run_source = inspect.getsource(systems.run_end_to_end_systems)
    assert "hashes_before = checkpoint_hashes(run_dir)" in run_source
    assert "hashes_after = checkpoint_hashes(run_dir)" in run_source
    assert "checkpoint_hashes_unchanged = hashes_before == hashes_after" in run_source
    assert "_parameter_versions(reference)" in run_source
    assert "_parameter_versions(candidate)" in run_source
    assert "_parameter_versions(transformer)" in run_source
    for forbidden in (
        "torch.save(",
        ".write_text(",
        ".write_bytes(",
        "TokenBin",
        "torch.optim",
        ".backward(",
        "optimizer.step",
        "zero_grad(",
    ):
        assert forbidden not in source


def test_issue501_has_no_modal_workflow_or_scientific_scale_authorization() -> None:
    source = MODULE.read_text()
    lowered = source.lower()
    for forbidden in (
        "modal.",
        "workflow_dispatch",
        "github_run_attempt",
        "modal run",
    ):
        assert forbidden not in lowered
    tree = ast.parse(source)
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"backward", "step", "zero_grad"}:
            forbidden_calls.append(func.attr)
    assert forbidden_calls == []
    protocol = systems.systems_protocol()
    for key in (
        "training_performed",
        "optimizer_created",
        "backward_performed",
        "corpus_accessed",
        "checkpoint_write_authorized",
        "scientific_seed_consumed",
        "independent_replication_credit",
        "end_to_end_systems_authorized_by_issue501",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False
