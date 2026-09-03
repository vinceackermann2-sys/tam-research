from __future__ import annotations

"""Issue #602 one-shot AERA-v26.9 identity-weight-visibility READ gate.

The entire authoritative #558/#553 decision surface is reused without threshold
or fixture relaxation, substituting only the v26.9 backend/tail. Four additional
production-shaped rows exercise FP32 identity/context and durable state under
CUDA BF16 autocast, where projected query/similarity are BF16 while the literal
Torch-reference softmax-weight visibility remains FP32.
"""

from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import aera_v26_3_ficem_read_probe as frozen
from . import aera_v26_7_issue553_ficem_read_mixed_dtype_probe as frozen553
from . import aera_v26_8_issue558_ficem_read_mixed_strength_precision_probe as frozen558
from .aera_hardware_core_v24 import (
    MIN_STRENGTH,
    READ_TEMPERATURE,
    READ_TOP_K,
    ContextualEpisodicMemoryState,
)
from .aera_hardware_core_v25_1 import _set_known_empty_hint
from .aera_hardware_core_v26 import TorchFICEMReferenceBackend
from . import aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as v26_9

RESEARCH_ISSUE = 602
SOURCE_MAIN = "37d2352050730c75dff0ab4b547e990b7865a95d"
SOURCE_TREE = "c3fd42879162cdc5e01b1ed0fcc34f2f82aa454f"

V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"
V26_9_CPU_TEST_BLOB = "305ec5732c46ceab2de9116898c54beb859e41e8"
V26_8_BACKEND_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
ISSUE558_PROBE_BLOB = "99ab8252f2b594404aae1ca86752eaa902eb80a5"
FROZEN_ISSUE553_PROBE_BLOB = "ff9a47f510be07e8adeff018f327338147163cdb"
HISTORICAL_PROBE_BLOB = "16aa99b9f6f0a1d11bd7bf5f36b2f0b1fb97047b"
REPAIR5_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
FACTORIZED_V25_BLOB = "f8cce87fa4dcae69fd171ba95fcbdab50e743a2f"

ISSUE558_TRIGGER = 561
ISSUE558_BOUND_MAIN = "75987bfb7976c6a970d63801c6e81b5b4993f544"
ISSUE558_DECISION = "PASS"
ISSUE594_TRIGGER = 596
ISSUE594_RUN = 33772104621
ISSUE594_JOB = 100704667286
ISSUE594_RESULT_SHA256 = (
    "c950d8fa50e70a48ec64a87f860d70d854cf1a2b58e1acbdfbcb0052495e809e"
)
ISSUE597_TRIGGER = 599
ISSUE597_RUN = 33774062361
ISSUE597_JOB = 100711243436

DESIGN_SEED = frozen553.DESIGN_SEED
D_MODEL = frozen553.D_MODEL
MEMORY_DIM = frozen553.MEMORY_DIM
CAPACITY = frozen553.CAPACITY
TIME = frozen553.TIME
BATCH_SIZES = frozen553.BATCH_SIZES
VALIDITY_KINDS = frozen553.VALIDITY_KINDS
BF16_ATOL = frozen553.BF16_ATOL
BF16_RTOL = frozen553.BF16_RTOL


def issue602_protocol() -> dict[str, Any]:
    protocol = dict(v26_9.identity_weight_visibility_v26_9_protocol())
    protocol.update(
        {
            "probe_version": "aera-v26.9-issue602-identity-weight-visibility-l4",
            "research_issue": RESEARCH_ISSUE,
            "source_main_issue602": SOURCE_MAIN,
            "source_tree_issue602": SOURCE_TREE,
            "v26_9_backend_blob": V26_9_BACKEND_BLOB,
            "v26_9_cpu_test_blob": V26_9_CPU_TEST_BLOB,
            "v26_8_backend_blob": V26_8_BACKEND_BLOB,
            "issue558_probe_blob": ISSUE558_PROBE_BLOB,
            "frozen_issue553_probe_blob": FROZEN_ISSUE553_PROBE_BLOB,
            "historical_probe_blob": HISTORICAL_PROBE_BLOB,
            "repair5_backend_blob": REPAIR5_BACKEND_BLOB,
            "v26_6_write_blob": V26_6_WRITE_BLOB,
            "v26_interface_blob": V26_INTERFACE_BLOB,
            "stable_reference_blob": STABLE_REFERENCE_BLOB,
            "factorized_v25_blob": FACTORIZED_V25_BLOB,
            "issue558_trigger": ISSUE558_TRIGGER,
            "issue558_bound_main": ISSUE558_BOUND_MAIN,
            "issue558_decision": ISSUE558_DECISION,
            "issue558_surface_preserved_wholesale": True,
            "issue558_historical_tail_identity_dtype_equals_similarity_dtype": True,
            "issue558_thresholds_relaxed": False,
            "issue594_trigger": ISSUE594_TRIGGER,
            "issue594_run": ISSUE594_RUN,
            "issue594_job": ISSUE594_JOB,
            "issue594_result_sha256": ISSUE594_RESULT_SHA256,
            "issue597_trigger": ISSUE597_TRIGGER,
            "issue597_run": ISSUE597_RUN,
            "issue597_job": ISSUE597_JOB,
            "integrated_rows": 4,
            "integrated_identity_dtype": "float32",
            "integrated_context_dtype": "float32",
            "integrated_durable_dtype": "float32",
            "integrated_projected_query_dtype": "bfloat16",
            "integrated_similarity_dtype": "bfloat16",
            "integrated_normalized_keys_dtype": "float32",
            "integrated_atol": BF16_ATOL,
            "integrated_rtol": BF16_RTOL,
            "integrated_timing_decision_bearing": False,
            "integrated_full_backend_required": True,
            "gpu_authorized_by_probe_module": False,
            "end_to_end_systems_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
            "scientific_seed_consumed": False,
        }
    )
    return protocol


def cpu_contract_preflight_issue602() -> dict[str, Any]:
    frozen_contract = frozen558.cpu_contract_preflight_issue558()
    candidate_contract = v26_9.cpu_contract_preflight_issue600()
    if DESIGN_SEED != 408_411:
        raise RuntimeError("issue602 design seed drifted")
    if (D_MODEL, MEMORY_DIM, CAPACITY, TIME) != (200, 50, 48, 256):
        raise RuntimeError("issue602 geometry drifted")
    if BATCH_SIZES != (8, 64) or VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("issue602 row grid drifted")
    if (BF16_ATOL, BF16_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue602 integrated tolerance drifted")
    if frozen_contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue602 frozen #558 CPU contract drifted")
    if candidate_contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue602 v26.9 CPU contract drifted")
    protocol = issue602_protocol()
    forbidden = (
        "gpu_authorized_by_probe_module",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
        "scientific_seed_consumed",
    )
    if any(protocol[key] for key in forbidden):
        raise RuntimeError("issue602 CPU contract unexpectedly authorizes higher work")
    return {
        "frozen_issue558_contract": frozen_contract,
        "v26_9_candidate_contract": candidate_contract,
        "protocol": protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "scientific_seed_consumed": False,
    }


def _historical_tail_adapter(
    similarity: torch.Tensor,
    strengths: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
    *,
    return_top_indices: bool = False,
):
    # #558 fixtures deliberately made source dtype == compute/similarity dtype.
    return v26_9.fused_ficem_read_tail_v26_9(
        similarity,
        strengths,
        valid,
        values,
        identity_dtype=similarity.dtype,
        return_top_indices=return_top_indices,
    )


def _tail_profile_v26_9(call: Callable[[], Any]) -> dict[str, Any]:
    call()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        output = call()
    torch.cuda.synchronize()
    del output

    cuda_events = 0
    triton_events = 0
    for event in profile.events():
        device_type = getattr(event, "device_type", None)
        if device_type == torch.autograd.DeviceType.CUDA or str(device_type).endswith("CUDA"):
            cuda_events += 1
            if "mixed_identity_weight_visibility_kernel" in str(
                getattr(event, "name", "")
            ):
                triton_events += 1

    operators = {"topk": 0, "softmax": 0, "gather": 0, "_to_copy": 0, "copy_": 0}
    for item in profile.key_averages():
        key = str(item.key).lower()
        for token in operators:
            if token in key:
                operators[token] += int(item.count)
    return {
        "cuda_device_events": int(cuda_events),
        "triton_read_tail_events": int(triton_events),
        "relevant_operator_calls": operators,
    }


def _run_preserved_issue558_surface() -> dict[str, Any]:
    original_backend = frozen553.MixedDtypeTritonFICEMReadWriteBackend
    original_tail = frozen553.fused_ficem_read_tail_mixed_dtype
    original_profiler = frozen553._tail_profile_with_cast_accounting
    try:
        frozen553.MixedDtypeTritonFICEMReadWriteBackend = (
            v26_9.IdentityWeightVisibilityTritonFICEMReadWriteBackend
        )
        frozen553.fused_ficem_read_tail_mixed_dtype = _historical_tail_adapter
        frozen553._tail_profile_with_cast_accounting = _tail_profile_v26_9
        result = frozen553.run_ficem_read_mixed_dtype_probe_v26_7()
    finally:
        frozen553.MixedDtypeTritonFICEMReadWriteBackend = original_backend
        frozen553.fused_ficem_read_tail_mixed_dtype = original_tail
        frozen553._tail_profile_with_cast_accounting = original_profiler

    result["frozen_issue553_protocol"] = result["protocol"]
    result["protocol"] = frozen558.issue558_protocol()
    result["issue602_preservation_metadata"] = {
        "candidate_backend": v26_9.IdentityWeightVisibilityTritonFICEMReadWriteBackend.name,
        "tail_adapter_identity_dtype_equals_similarity_dtype": True,
        "thresholds_relaxed": False,
        "profiler_acceptance_changed": False,
        "profiler_event_name_updated_only": True,
    }
    return result


def _advance_generator_through_issue558_regular_rows(
    generator: torch.Generator, device: torch.device
) -> None:
    # Replays only fixture creation, in the exact #553 order, to recover the
    # deterministic generator state immediately after all #558 regular rows.
    for dtype_name in frozen553.DTYPE_NAMES:
        for batch_size in BATCH_SIZES:
            for validity_kind in VALIDITY_KINDS:
                case = frozen.make_case(
                    dtype_name=dtype_name,
                    batch_size=batch_size,
                    validity_kind=validity_kind,
                    generator=generator,
                    device=device,
                )
                del case
    for compute_dtype_name, durable_dtype_name in frozen553.MIXED_LAYOUTS:
        for batch_size in BATCH_SIZES:
            for validity_kind in VALIDITY_KINDS:
                case = frozen553._make_mixed_case(
                    compute_dtype_name=compute_dtype_name,
                    durable_dtype_name=durable_dtype_name,
                    batch_size=batch_size,
                    validity_kind=validity_kind,
                    generator=generator,
                    device=device,
                )
                del case


def _make_integrated_case(
    *,
    batch_size: int,
    validity_kind: str,
    generator: torch.Generator,
    device: torch.device,
) -> frozen.ReadCase:
    identity = frozen._cpu_randn(
        (batch_size, TIME, D_MODEL), generator=generator, dtype=torch.float32
    ).to(device)
    context = frozen._cpu_randn(
        (batch_size, TIME, D_MODEL), generator=generator, dtype=torch.float32
    ).to(device)
    keys = frozen._cpu_randn(
        (batch_size, CAPACITY, MEMORY_DIM), generator=generator, dtype=torch.float32
    ).to(device)
    values = frozen._cpu_randn(
        (batch_size, CAPACITY, MEMORY_DIM), generator=generator, dtype=torch.float32
    ).to(device)
    strengths = (
        torch.rand(batch_size, CAPACITY, generator=generator).mul_(0.95).add_(0.05)
    ).to(dtype=torch.float32, device=device)
    if validity_kind == "full":
        valid = torch.ones(batch_size, CAPACITY, dtype=torch.bool, device=device)
    else:
        valid_cpu = torch.rand(batch_size, CAPACITY, generator=generator) > 0.35
        valid_cpu[:, :8] = True
        valid = valid_cpu.to(device)
    state = ContextualEpisodicMemoryState(keys, values, strengths, valid)
    _set_known_empty_hint(state, False)
    # "bfloat16" selects the frozen CUDA BF16 autocast context while source/state
    # tensors deliberately remain FP32.
    return frozen.ReadCase(
        "bfloat16", batch_size, validity_kind, identity, context, state
    )


def _literal_reference_tail(
    identity_dtype: torch.dtype,
    similarity: torch.Tensor,
    state: ContextualEpisodicMemoryState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    strength_bias = torch.log(state.strengths.clamp(MIN_STRENGTH, 1.0))[:, None, :]
    logits = (similarity + strength_bias) / READ_TEMPERATURE
    masked = logits.masked_fill(~state.valid[:, None, :], -torch.inf)
    top_logits, top_indices = torch.topk(masked, k=READ_TOP_K, dim=-1)
    top_valid = state.valid[:, None, :].expand(-1, similarity.size(1), -1).gather(
        -1, top_indices
    )
    safe_logits = top_logits.masked_fill(~top_valid, -1e9)
    weights = torch.softmax(safe_logits.float(), dim=-1).to(identity_dtype)
    weights = weights * top_valid.to(weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    expanded_values = state.values[:, None, :, :].expand(
        -1, similarity.size(1), -1, -1
    )
    gathered_values = expanded_values.gather(
        2,
        top_indices.unsqueeze(-1).expand(-1, -1, -1, MEMORY_DIM),
    )
    recalled = (weights.unsqueeze(-1) * gathered_values).sum(dim=2)
    return recalled, top_indices, masked


def _integrated_row(
    memory,
    reference: TorchFICEMReferenceBackend,
    candidate: v26_9.IdentityWeightVisibilityTritonFICEMReadWriteBackend,
    case: frozen.ReadCase,
) -> dict[str, Any]:
    identity_before = case.identity.clone()
    context_before = case.context.clone()
    state_before = frozen._clone_state(case.state)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, _, query = memory.address_factors(case.identity, case.context)
        keys = F.normalize(case.state.keys, dim=-1)
        similarity = torch.einsum("btd,bsd->bts", query, keys).contiguous()

    dtype_split_exact = bool(
        case.identity.dtype is torch.float32
        and case.context.dtype is torch.float32
        and case.state.keys.dtype is torch.float32
        and case.state.values.dtype is torch.float32
        and case.state.strengths.dtype is torch.float32
        and case.state.valid.dtype is torch.bool
        and query.dtype is torch.bfloat16
        and similarity.dtype is torch.bfloat16
        and keys.dtype is torch.float32
    )

    reference_recalled, reference_indices, masked_logits = _literal_reference_tail(
        torch.float32, similarity, case.state
    )
    candidate_recalled, candidate_indices = v26_9.fused_ficem_read_tail_v26_9(
        similarity,
        case.state.strengths,
        case.state.valid,
        case.state.values,
        identity_dtype=torch.float32,
        return_top_indices=True,
    )
    torch.cuda.synchronize()
    if candidate_indices is None:
        raise RuntimeError("issue602 integrated row missing candidate indices")

    selection = frozen._tie_aware_top4_equivalence(
        masked_logits,
        case.state.valid,
        reference_indices,
        candidate_indices,
    )
    pre_out_close = torch.allclose(
        reference_recalled,
        candidate_recalled,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    reference_result = frozen._full_read(reference, memory, case)
    candidate_result = frozen._full_read(candidate, memory, case)
    torch.cuda.synchronize()
    final_close = torch.allclose(
        reference_result.recalled,
        candidate_result.recalled,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )

    reuse_exact = bool(
        reference_result.projected_query is not None
        and candidate_result.projected_query is not None
        and reference_result.normalized_old_keys is not None
        and candidate_result.normalized_old_keys is not None
        and torch.equal(reference_result.projected_query, candidate_result.projected_query)
        and torch.equal(
            reference_result.normalized_old_keys,
            candidate_result.normalized_old_keys,
        )
        and torch.equal(reference_result.projected_query, query)
        and torch.equal(reference_result.normalized_old_keys, keys)
    )
    source_unchanged = bool(
        torch.equal(case.identity, identity_before)
        and torch.equal(case.context, context_before)
        and frozen._state_equal(case.state, state_before)
    )
    finite = all(
        bool(torch.isfinite(t).all())
        for t in (
            query,
            keys,
            similarity,
            reference_recalled,
            candidate_recalled,
            reference_result.recalled,
            candidate_result.recalled,
        )
    )
    meta_exact = bool(
        reference_recalled.shape == candidate_recalled.shape
        == (case.batch_size, TIME, MEMORY_DIM)
        and reference_recalled.dtype == candidate_recalled.dtype == torch.float32
        and reference_recalled.device == candidate_recalled.device
        and candidate_recalled.device.type == "cuda"
        and reference_result.recalled.shape == candidate_result.recalled.shape
        and reference_result.recalled.dtype == candidate_result.recalled.dtype
        and reference_result.recalled.device == candidate_result.recalled.device
        and candidate_result.recalled.device.type == "cuda"
    )

    direct_profile = _tail_profile_v26_9(
        lambda: v26_9.fused_ficem_read_tail_v26_9(
            similarity,
            case.state.strengths,
            case.state.valid,
            case.state.values,
            identity_dtype=torch.float32,
            return_top_indices=False,
        )
    )
    direct_topology_pass = bool(
        direct_profile["cuda_device_events"] == 1
        and direct_profile["triton_read_tail_events"] == 1
        and all(
            direct_profile["relevant_operator_calls"][token] == 0
            for token in ("topk", "softmax", "gather", "_to_copy", "copy_")
        )
    )
    full_profile = frozen._cuda_profile(
        lambda: frozen._full_read(candidate, memory, case)
    )
    full_no_reference_tail_ops = bool(
        all(
            full_profile["relevant_operator_calls"][token] == 0
            for token in ("topk", "softmax", "gather")
        )
    )

    passed = bool(
        dtype_split_exact
        and selection["selection_semantically_equivalent"]
        and pre_out_close
        and final_close
        and reuse_exact
        and source_unchanged
        and finite
        and meta_exact
        and direct_topology_pass
        and full_no_reference_tail_ops
    )
    return {
        "pass": passed,
        "batch_size": case.batch_size,
        "validity_kind": case.validity_kind,
        "identity_dtype": str(case.identity.dtype),
        "context_dtype": str(case.context.dtype),
        "keys_dtype": str(case.state.keys.dtype),
        "values_dtype": str(case.state.values.dtype),
        "strengths_dtype": str(case.state.strengths.dtype),
        "valid_dtype": str(case.state.valid.dtype),
        "projected_query_dtype": str(query.dtype),
        "similarity_dtype": str(similarity.dtype),
        "normalized_keys_dtype": str(keys.dtype),
        "dtype_split_exact": dtype_split_exact,
        "selection_semantically_equivalent": bool(
            selection["selection_semantically_equivalent"]
        ),
        "distinct_selected_set_exact": bool(selection["distinct_selected_set_exact"]),
        "tied_selection_semantically_valid": bool(
            selection["tied_selection_semantically_valid"]
        ),
        "pre_out_recalled_close": bool(pre_out_close),
        "final_out_close": bool(final_close),
        "query_and_normalized_keys_bit_exact": reuse_exact,
        "source_unchanged": source_unchanged,
        "finite": finite,
        "dtype_device_shape_exact": meta_exact,
        "direct_tail_topology_pass": direct_topology_pass,
        "direct_tail_profile": direct_profile,
        "full_backend_no_reference_tail_ops": full_no_reference_tail_ops,
        "full_backend_profile": full_profile,
        "atol": BF16_ATOL,
        "rtol": BF16_RTOL,
        "pre_out_max_abs_diff": float(
            (reference_recalled.float() - candidate_recalled.float()).abs().max()
        ),
        "final_out_max_abs_diff": float(
            (
                reference_result.recalled.float()
                - candidate_result.recalled.float()
            ).abs().max()
        ),
        "timing_decision_bearing": False,
    }


def run_identity_weight_visibility_gate_v26_9_issue602() -> dict[str, Any]:
    cpu_contract_preflight_issue602()
    if not torch.cuda.is_available():
        raise RuntimeError("issue602 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"issue602 requires NVIDIA L4, found {device_name}")

    preserved = _run_preserved_issue558_surface()

    memory = frozen.build_memory(device)
    reference = TorchFICEMReferenceBackend()
    candidate = v26_9.IdentityWeightVisibilityTritonFICEMReadWriteBackend()
    generator = torch.Generator().manual_seed(DESIGN_SEED)
    _advance_generator_through_issue558_regular_rows(generator, device)

    integrated_rows: dict[str, dict[str, Any]] = {}
    for batch_size in BATCH_SIZES:
        for validity_kind in VALIDITY_KINDS:
            case = _make_integrated_case(
                batch_size=batch_size,
                validity_kind=validity_kind,
                generator=generator,
                device=device,
            )
            key = f"fp32_source_bf16_projection_fp32_durable_batch{batch_size}_{validity_kind}"
            integrated_rows[key] = _integrated_row(
                memory, reference, candidate, case
            )
            del case

    integrated_pass = all(row["pass"] for row in integrated_rows.values())
    overall_pass = bool(preserved["overall_pass"] and integrated_pass)
    return {
        "protocol": issue602_protocol(),
        "device": device_name,
        "preserved_issue558": preserved,
        "integrated": {
            "rows": integrated_rows,
            "rows_pass": integrated_pass,
            "overall_pass": integrated_pass,
            "timing_decision_bearing": False,
        },
        "overall_pass": overall_pass,
        "decision": "PASS" if overall_pass else "FAIL",
        "synthetic_only": True,
        "memory_module_random_weights_only": True,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
