from __future__ import annotations

"""AERA-v26.5 issue #503 orchestration-only systems harness repair1.

The merged #501 systems evaluator is preserved byte-for-byte as historical CPU
proof.  This successor reuses its frozen models, helpers, workload, thresholds,
and decision semantics.  The only semantic correction is orchestration context:
models are constructed as ordinary version-tracked parameters, while every
actual systems measurement executes under an explicit inference-mode context.

Issue #503 authorizes no GPU run.  A separately preregistered one-shot systems
gate is still required after this CPU-only compatibility repair is green/merged.
"""

from typing import Any, Callable

import torch

from . import aera_v25_post8471_triage as triage
from . import aera_v26_5_end_to_end_systems as base
from .aera_hardware_core import HardwareAERAState

REPAIR_ISSUE = 503
SOURCE_MAIN = "bd7fe8aab50af30006b7cb8a5f790699736379e0"
PREDECESSOR_MODULE_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"


def repair1_protocol() -> dict[str, Any]:
    protocol = dict(base.systems_protocol())
    predecessor_version = protocol["version"]
    protocol.update(
        {
            "version": "aera-v26.5-issue503-version-tracked-orchestration-repair1",
            "repair_issue": REPAIR_ISSUE,
            "repair_source_main": SOURCE_MAIN,
            "predecessor_module_blob": PREDECESSOR_MODULE_BLOB,
            "predecessor_protocol_version": predecessor_version,
            "top_level_inference_decorated": False,
            "model_construction_outside_inference_mode": True,
            "parameter_version_snapshots_outside_inference_mode": True,
            "measurements_inside_explicit_inference_mode": True,
            "historical_issue501_module_mutated": False,
            "gpu_authorized_by_issue503": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "fresh_scientific_seed_authorized": False,
            "independent_replication_credit": False,
            "100m_authorized": False,
            "breakthrough_proven": False,
        }
    )
    return protocol


def cpu_contract_preflight_repair1() -> dict[str, Any]:
    predecessor = base.cpu_contract_preflight()
    return {
        "repair_issue": REPAIR_ISSUE,
        "source_main": SOURCE_MAIN,
        "predecessor_module_blob": PREDECESSOR_MODULE_BLOB,
        "predecessor": predecessor,
        "gpu_authorized_by_issue503": False,
        "model_construction_performed": False,
        "checkpoint_loaded": False,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def run_end_to_end_systems_repair1(
    *, run_dir: str = base.CHECKPOINT_RELATIVE_DIR
) -> dict[str, Any]:
    """Run the frozen #501 systems gate with version-tracked model construction.

    This function is deliberately *not* decorated with ``torch.inference_mode``.
    Model construction/loading and before/after parameter-version snapshots occur
    in normal mode.  The complete measurement region is explicitly inference-only.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("issue503 integrated systems comparison requires one NVIDIA L4")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    hashes_before = base.checkpoint_hashes(run_dir)
    reference, candidate, transformer = base.load_models(run_dir=run_dir, device=device)
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
                "v26_4_triton_full_ficem": candidate_full_call,
            }
            summaries = base._timed_summaries(calls, batch_size=batch_size)

            base._reset_execution_counters(reference)
            reference_output = reference_full_call()
            if not isinstance(reference_output, dict):
                raise RuntimeError("issue503 reference full call did not return mapping")
            base._reset_execution_counters(candidate)
            candidate_output = candidate_full_call()
            if not isinstance(candidate_output, dict):
                raise RuntimeError("issue503 candidate full call did not return mapping")

            reference_logits = reference_output.get("logits")
            candidate_logits = candidate_output.get("logits")
            reference_state = reference_output.get("state")
            candidate_state = candidate_output.get("state")
            if not isinstance(reference_logits, torch.Tensor) or not isinstance(
                candidate_logits, torch.Tensor
            ):
                raise RuntimeError("issue503 full call missing logits")
            if not isinstance(reference_state, HardwareAERAState) or not isinstance(
                candidate_state, HardwareAERAState
            ):
                raise RuntimeError("issue503 full call missing HardwareAERAState")

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
            logit_equivalence = base._logit_equivalence(
                reference_logits, candidate_logits
            )
            state_equivalence = base._state_equivalence(
                reference_state, candidate_state
            )
            physical_sparse = base._physical_sparse_proof(candidate, candidate_output)
            write_geometry = base._write_geometry(candidate)
            finite = base._finite_output(reference_output) and base._finite_output(
                candidate_output
            )
            actual_state_bytes = base._episodic_state_bytes_per_session(
                candidate_state, batch_size
            )

            transformer_tps = summaries["transformer"][
                "tokens_per_second_from_median"
            ]
            candidate_tps = summaries["v26_4_triton_full_ficem"][
                "tokens_per_second_from_median"
            ]
            reference_ms = summaries["v26_torch_reference_full_ficem"]["median_ms"]
            candidate_ms = summaries["v26_4_triton_full_ficem"]["median_ms"]
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
                "persistent_state_bytes_pass": actual_state_bytes
                == base.EXPECTED_STATE_BYTES,
                "candidate_full_vs_transformer_speed_ratio": full_speed_ratio,
                "required_full_speed_ratio": required_speed_ratio,
                "throughput_pass": full_speed_ratio >= required_speed_ratio,
                "reference_full_latency_ms": reference_ms,
                "candidate_full_latency_ms": candidate_ms,
                "candidate_vs_reference_latency_ratio": candidate_ms / reference_ms,
                "no_reference_full_latency_regression": no_reference_latency_regression,
                "peak_vram": {
                    "transformer": base._peak_vram_mb(transformer_call),
                    "v26_torch_reference_full": base._peak_vram_mb(
                        reference_full_call
                    ),
                    "v26_4_triton_full": base._peak_vram_mb(candidate_full_call),
                },
                "profiler_candidate_full": base._profile_candidate(
                    candidate_full_call
                ),
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
        "scope": "aera_v26_5_issue503_physically_real_sparse_end_to_end_systems_repair1",
        "protocol": repair1_protocol(),
        "device": torch.cuda.get_device_name(device),
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
