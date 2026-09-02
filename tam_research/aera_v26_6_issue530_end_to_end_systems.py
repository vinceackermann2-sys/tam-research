from __future__ import annotations

"""Issue #530 CPU-first v26.6 end-to-end systems adapter.

This module preserves the frozen #501 systems workload/decision surface and the
#503 version-tracked orchestration repair.  Its only candidate semantic change is
execution-backend selection: the unmeasured v26.4 backend installed by the frozen
#501 loader is immediately replaced with the exact #527-passed v26.6
MaterializeCast backend before parameter snapshots or any model call/measurement.

Issue #530 authorizes no GPU run.  A later separately preregistered one-shot
systems gate is required after this module is CPU-green and merged.
"""

from typing import Any, Callable

import torch

from . import aera_v25_post8471_triage as triage
from . import aera_v26_5_end_to_end_systems as base
from . import aera_v26_5_end_to_end_systems_repair1 as repair1
from .aera_hardware_core import HardwareAERAState
from .aera_hardware_core_v26 import CoalescedFICEMMemory, TorchFICEMReferenceBackend
from .aera_hardware_core_v26_6_ficem_write_materialize_cast import (
    MaterializeCastTritonFICEMReadWriteBackend,
    materialize_cast_ficem_read_write_v26_6_protocol,
)

RESEARCH_ISSUE = 530
SOURCE_MAIN = "2c0c28005bff8d9b4f36a96de86144dd74107e39"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
HISTORICAL_V26_4_WRITE_BACKEND_BLOB = "e54570292489bd17570038dca7518419ac00418c"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
V26_6_CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"

ISSUE527_TRIGGER = 529
ISSUE527_RUN = 33680028132
ISSUE527_JOB = 100414089065
ISSUE527_ATTEMPT = 1
ISSUE527_BOUND_MAIN = SOURCE_MAIN
ISSUE527_RESULT_PATH = "/vol/aera-v26/issue527-ficem-write-repaired-oracle/result.json"
ISSUE527_RESULT_SHA256 = "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"
ISSUE527_ORACLE_BLOB = "8f472451af4024bb3faacb56d814f7d6bdb25cc9"
ISSUE527_PROBE_BLOB = "bcfeb6a93ed062b7d00359603dc9fbc7aca5767f"

ISSUE508_TRIGGER = 510
ISSUE508_RUN = 33661498305
ISSUE508_JOB = 100352870198
ISSUE508_ATTEMPT = 1
ISSUE508_BOUND_MAIN = "1d475a199cfd2b14d5e94e5cffa29e05ac868ab1"
ISSUE508_FAILURE = "FICEM write state/value floating dtypes must match"


def issue530_systems_protocol() -> dict[str, Any]:
    """Return #503/#501 protocol with only v26.6 candidate identity substituted."""

    protocol = dict(repair1.repair1_protocol())
    candidate_protocol = materialize_cast_ficem_read_write_v26_6_protocol()
    protocol.update(
        {
            "version": "aera-v26.6-issue530-end-to-end-systems-adapter",
            "research_issue": RESEARCH_ISSUE,
            "source_main": SOURCE_MAIN,
            "base_systems_blob": BASE_SYSTEMS_BLOB,
            "repair1_systems_blob": REPAIR1_SYSTEMS_BLOB,
            "v26_6_candidate_blob": V26_6_CANDIDATE_BLOB,
            "candidate_backend": MaterializeCastTritonFICEMReadWriteBackend.name,
            "reference_backend": TorchFICEMReferenceBackend.name,
            "historical_v26_4_candidate_backend_decision_bearing": False,
            "frozen_loader_v26_4_backend_replaced_before_parameter_snapshot": True,
            "frozen_loader_v26_4_backend_replaced_before_any_model_call": True,
            "only_candidate_semantic_change": "execution_backend_v26_4_to_v26_6_materialize_cast",
            "issue527_trigger": ISSUE527_TRIGGER,
            "issue527_run": ISSUE527_RUN,
            "issue527_job": ISSUE527_JOB,
            "issue527_attempt": ISSUE527_ATTEMPT,
            "issue527_bound_main": ISSUE527_BOUND_MAIN,
            "issue527_result_path": ISSUE527_RESULT_PATH,
            "issue527_result_sha256": ISSUE527_RESULT_SHA256,
            "issue527_decision": "PASS",
            "issue527_direct_pass": [256, 256],
            "issue527_edge_pass": [32, 32],
            "issue527_public_pass": [6, 6],
            "issue527_topology_pass": [4, 4],
            "issue527_oracle_blob": ISSUE527_ORACLE_BLOB,
            "issue527_probe_blob": ISSUE527_PROBE_BLOB,
            "issue508_trigger": ISSUE508_TRIGGER,
            "issue508_run": ISSUE508_RUN,
            "issue508_job": ISSUE508_JOB,
            "issue508_attempt": ISSUE508_ATTEMPT,
            "issue508_bound_main": ISSUE508_BOUND_MAIN,
            "issue508_authoritative_result_emitted": False,
            "issue508_integrated_failure": ISSUE508_FAILURE,
            "candidate_fieldwise_mixed_dtype_supported": candidate_protocol[
                "write_fieldwise_mixed_dtype_supported"
            ],
            "candidate_global_cross_field_dtype_equality_required": candidate_protocol[
                "write_global_cross_field_dtype_equality_required"
            ],
            "candidate_write_supported_float_dtypes": candidate_protocol[
                "write_supported_float_dtypes"
            ],
            "candidate_write_tail_triton_launches_target": candidate_protocol[
                "write_tail_triton_launches_target"
            ],
            "candidate_read_backend_changed_by_v26_6": candidate_protocol[
                "read_backend_changed_by_v26_6"
            ],
            "systems_gpu_authorized_by_issue530": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol


def cpu_contract_preflight_issue530() -> dict[str, Any]:
    """CPU-only proof that #530 does not alter the frozen systems decision surface."""

    predecessor = repair1.cpu_contract_preflight_repair1()
    protocol = issue530_systems_protocol()
    candidate_protocol = materialize_cast_ficem_read_write_v26_6_protocol()

    if base.SYSTEM_BATCH_SIZES != (8, 64):
        raise RuntimeError("issue530 batch sizes drifted")
    if (base.SYSTEM_WARMUP_CALLS, base.SYSTEM_TIMED_CALLS_PER_ROUND, base.SYSTEM_ROUNDS) != (
        3,
        20,
        5,
    ):
        raise RuntimeError("issue530 timing design drifted")
    if (base.BATCH8_MIN_FULL_SPEED_RATIO, base.BATCH64_MIN_FULL_SPEED_RATIO) != (0.25, 1.25):
        raise RuntimeError("issue530 throughput thresholds drifted")
    if (base.INTEGRATED_ATOL, base.INTEGRATED_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue530 integrated tolerances drifted")
    if base.EXPECTED_STATE_BYTES != 77_760:
        raise RuntimeError("issue530 persistent-state bytes drifted")
    if (base.EXPECTED_SELECTED_WRITES, base.EXPECTED_CANDIDATES, base.EXPECTED_VECTOR_UPDATES) != (
        16,
        255,
        1,
    ):
        raise RuntimeError("issue530 write geometry drifted")

    required_candidate = {
        "backend_name": MaterializeCastTritonFICEMReadWriteBackend.name,
        "repair5_read_backend_blob": READ_BACKEND_BLOB,
        "predecessor_write_backend_blob": HISTORICAL_V26_4_WRITE_BACKEND_BLOB,
        "write_global_cross_field_dtype_equality_required": False,
        "write_supported_float_dtypes": ["float32", "bfloat16"],
        "write_fieldwise_mixed_dtype_supported": True,
        "write_materialization_output_follows_durable_state_field_dtype": True,
        "write_duplicate_decisions_before_materialization": True,
        "write_materialize_both_branches_cast_to_output_element_type": True,
        "write_materialize_cast_numeric_not_bitcast": True,
        "write_explicit_pre_tail_cast_kernels": 0,
        "write_adjudicate_kernel_changed_by_v26_6": False,
        "write_tail_triton_launches_target": 2,
        "read_backend_changed_by_v26_6": False,
        "write_similarity_einsums_changed_by_v26_6": False,
        "write_value_projection_changed_by_v26_6": False,
        "write_strength_semantics_changed_by_v26_6": False,
        "write_duplicate_semantics_changed_by_v26_6": False,
        "write_state_schema_changed_by_v26_6": False,
        "write_persistent_state_changed_by_v26_6": False,
        "write_training_backend_changed_by_v26_6": False,
    }
    for key, expected in required_candidate.items():
        if candidate_protocol.get(key) != expected:
            raise RuntimeError(
                f"issue530 candidate protocol drift: {key}={candidate_protocol.get(key)!r} expected={expected!r}"
            )

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "protocol": protocol,
        "predecessor": predecessor,
        "candidate_protocol": candidate_protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _install_v26_6_candidate_backend(candidate: torch.nn.Module) -> tuple[str, ...]:
    names: list[str] = []
    stages = getattr(candidate, "stages", None)
    if stages is None:
        raise RuntimeError("issue530 candidate lacks stages")
    for stage in stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue530 candidate stage lacks CoalescedFICEMMemory")
        memory._execution_backend = MaterializeCastTritonFICEMReadWriteBackend()
        if memory.execution_backend_name != MaterializeCastTritonFICEMReadWriteBackend.name:
            raise RuntimeError("issue530 v26.6 candidate backend installation failed")
        names.append(memory.execution_backend_name)
    if not names:
        raise RuntimeError("issue530 candidate has no memory stages")
    return tuple(names)


def load_models_v26_6(*, run_dir: str, device: torch.device):
    """Reuse frozen #501 loading/schema checks, then replace candidate before use."""

    reference, candidate, transformer = base.load_models(run_dir=run_dir, device=device)
    candidate_backend_names = _install_v26_6_candidate_backend(candidate)

    for stage in reference.stages:
        memory = stage.memory
        if not isinstance(memory, CoalescedFICEMMemory):
            raise RuntimeError("issue530 reference stage lacks CoalescedFICEMMemory")
        if memory.execution_backend_name != TorchFICEMReferenceBackend.name:
            raise RuntimeError("issue530 reference backend drifted from exact Torch reference")
    if any(
        name != MaterializeCastTritonFICEMReadWriteBackend.name
        for name in candidate_backend_names
    ):
        raise RuntimeError("issue530 candidate backend identity mismatch")
    return reference, candidate, transformer, candidate_backend_names


def run_end_to_end_systems_v26_6(
    *, run_dir: str = base.CHECKPOINT_RELATIVE_DIR
) -> dict[str, Any]:
    """Run the frozen #501/#503 systems decision surface with v26.6 candidate."""

    if not torch.cuda.is_available():
        raise RuntimeError("issue530 integrated systems comparison requires one NVIDIA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    hashes_before = base.checkpoint_hashes(run_dir)
    reference, candidate, transformer, candidate_backend_names = load_models_v26_6(
        run_dir=run_dir, device=device
    )
    reference_versions_before = base._parameter_versions(reference)
    candidate_versions_before = base._parameter_versions(candidate)
    transformer_versions_before = base._parameter_versions(transformer)

    rows: dict[str, Any] = {}
    with torch.inference_mode():
        for batch_size in base.SYSTEM_BATCH_SIZES:
            generator = torch.Generator(device="cpu").manual_seed(
                triage.DIAGNOSTIC_SEED + 10_000 + batch_size
            )
            tokens = torch.randint(
                0,
                triage.VOCAB_SIZE,
                (batch_size, triage.SEQ_LEN),
                generator=generator,
            ).to(device)

            transformer_call: Callable[[], object] = lambda: base._transformer_call(
                transformer, tokens
            )
            reference_full_call: Callable[[], object] = lambda: base._model_call(
                reference, tokens, update_memory=True
            )
            candidate_full_call: Callable[[], object] = lambda: base._model_call(
                candidate, tokens, update_memory=True
            )
            calls: dict[str, Callable[[], object]] = {
                "transformer": transformer_call,
                "v26_torch_reference_full_ficem": reference_full_call,
                "v26_6_triton_full_ficem": candidate_full_call,
            }
            summaries = base._timed_summaries(calls, batch_size=batch_size)

            base._reset_execution_counters(reference)
            reference_output = reference_full_call()
            if not isinstance(reference_output, dict):
                raise RuntimeError("issue530 reference full call did not return mapping")
            base._reset_execution_counters(candidate)
            candidate_output = candidate_full_call()
            if not isinstance(candidate_output, dict):
                raise RuntimeError("issue530 candidate full call did not return mapping")

            reference_logits = reference_output.get("logits")
            candidate_logits = candidate_output.get("logits")
            reference_state = reference_output.get("state")
            candidate_state = candidate_output.get("state")
            if not isinstance(reference_logits, torch.Tensor) or not isinstance(
                candidate_logits, torch.Tensor
            ):
                raise RuntimeError("issue530 full call missing logits")
            if not isinstance(reference_state, HardwareAERAState) or not isinstance(
                candidate_state, HardwareAERAState
            ):
                raise RuntimeError("issue530 full call missing HardwareAERAState")

            reference_signature = base._route_signature(reference_output)
            candidate_signature = base._route_signature(candidate_output)
            routing_exact = bool(
                len(reference_signature) == len(candidate_signature)
                and all(
                    torch.equal(reference_gate, candidate_gate)
                    for reference_gate, candidate_gate in zip(
                        reference_signature, candidate_signature
                    )
                )
                and triage._routing_accounting(reference_output, batch_size)
                == triage._routing_accounting(candidate_output, batch_size)
            )
            logit_equivalence = base._logit_equivalence(reference_logits, candidate_logits)
            state_equivalence = base._state_equivalence(reference_state, candidate_state)
            physical_sparse = base._physical_sparse_proof(candidate, candidate_output)
            write_geometry = base._write_geometry(candidate)
            finite = base._finite_output(reference_output) and base._finite_output(
                candidate_output
            )
            actual_state_bytes = base._episodic_state_bytes_per_session(
                candidate_state, batch_size
            )

            transformer_tps = summaries["transformer"]["tokens_per_second_from_median"]
            candidate_tps = summaries["v26_6_triton_full_ficem"][
                "tokens_per_second_from_median"
            ]
            reference_ms = summaries["v26_torch_reference_full_ficem"]["median_ms"]
            candidate_ms = summaries["v26_6_triton_full_ficem"]["median_ms"]
            full_speed_ratio = candidate_tps / transformer_tps
            required_speed_ratio = base._threshold_for_batch(batch_size)
            no_reference_latency_regression = candidate_ms <= reference_ms

            rows[str(batch_size)] = {
                "timings": summaries,
                "routing_reference": triage._routing_accounting(
                    reference_output, batch_size
                ),
                "routing_candidate": triage._routing_accounting(
                    candidate_output, batch_size
                ),
                "routing_exact": routing_exact,
                "logit_equivalence": logit_equivalence,
                "state_equivalence": state_equivalence,
                "physical_sparse": physical_sparse,
                "write_geometry": write_geometry,
                "finite": finite,
                "persistent_state_bytes_per_session_actual": actual_state_bytes,
                "persistent_state_bytes_pass": actual_state_bytes == base.EXPECTED_STATE_BYTES,
                "candidate_full_vs_transformer_speed_ratio": full_speed_ratio,
                "required_full_speed_ratio": required_speed_ratio,
                "throughput_pass": full_speed_ratio >= required_speed_ratio,
                "reference_full_latency_ms": reference_ms,
                "candidate_full_latency_ms": candidate_ms,
                "candidate_vs_reference_latency_ratio": candidate_ms / reference_ms,
                "no_reference_full_latency_regression": no_reference_latency_regression,
                "peak_vram": {
                    "transformer": base._peak_vram_mb(transformer_call),
                    "v26_torch_reference_full": base._peak_vram_mb(reference_full_call),
                    "v26_6_triton_full": base._peak_vram_mb(candidate_full_call),
                },
                "profiler_candidate_full": base._profile_candidate(candidate_full_call),
            }
            del reference_output, candidate_output, reference_state, candidate_state, tokens
            torch.cuda.empty_cache()

    reference_versions_after = base._parameter_versions(reference)
    candidate_versions_after = base._parameter_versions(candidate)
    transformer_versions_after = base._parameter_versions(transformer)
    versions_unchanged = bool(
        reference_versions_before == reference_versions_after
        and candidate_versions_before == candidate_versions_after
        and transformer_versions_before == transformer_versions_after
    )

    hashes_after = base.checkpoint_hashes(run_dir)
    checkpoint_hashes_unchanged = hashes_before == hashes_after

    per_batch_pass = {
        batch: bool(
            row["routing_exact"]
            and row["logit_equivalence"]["pass"]
            and row["state_equivalence"]["pass"]
            and row["physical_sparse"]["pass"]
            and row["write_geometry"]["pass"]
            and row["finite"]
            and row["persistent_state_bytes_pass"]
            and row["throughput_pass"]
            and row["no_reference_full_latency_regression"]
        )
        for batch, row in rows.items()
    }
    overall_pass = bool(
        all(per_batch_pass.values())
        and versions_unchanged
        and checkpoint_hashes_unchanged
    )
    return {
        "scope": "aera_v26_6_issue530_physically_real_sparse_end_to_end_systems_adapter",
        "protocol": issue530_systems_protocol(),
        "device": torch.cuda.get_device_name(device),
        "candidate_backend_names": list(candidate_backend_names),
        "rows": rows,
        "per_batch_pass": per_batch_pass,
        "parameter_versions_before": {
            "reference": reference_versions_before,
            "candidate": candidate_versions_before,
            "transformer": transformer_versions_before,
        },
        "parameter_versions_after": {
            "reference": reference_versions_after,
            "candidate": candidate_versions_after,
            "transformer": transformer_versions_after,
        },
        "parameter_versions_unchanged": versions_unchanged,
        "checkpoint_hashes_before": hashes_before,
        "checkpoint_hashes_after": hashes_after,
        "checkpoint_hashes_unchanged": checkpoint_hashes_unchanged,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "corpus_accessed": False,
        "checkpoint_written": False,
        "scientific_seed_consumed": False,
        "independent_replication_credit": False,
        "overall_pass": overall_pass,
        "decision": "PASS_E2E_SYSTEMS" if overall_pass else "FAIL_FROZEN_E2E_SYSTEMS_GATE",
        "claims": {
            "systems_prerequisite_satisfied": overall_pass,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        },
    }
