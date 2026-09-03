from __future__ import annotations

"""CPU-only read-only extractor for immutable #602 evidence with durable-key ordering."""

from typing import Any

RESEARCH_ISSUE = 619
SOURCE_MAIN = "793e488a6afd71a1dbe7142a771f4f936a9f4dda"
SOURCE_TREE = "0c18bc00b6fda5ec52a76968fa46745dcef31fd2"
ISSUE602_TRIGGER = 604
ISSUE602_RUN = 33783777048
ISSUE602_JOB = 100743509172
ISSUE602_RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"
ISSUE602_RESULT_SHA256 = "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"
ISSUE602_DECISION = "FAIL"
ISSUE617_RUN = 33793519431
ISSUE617_JOB = 100775569626

LOGICAL_ROWS = (
    "fp32_source_bf16_projection_fp32_durable_batch8_mixed",
    "fp32_source_bf16_projection_fp32_durable_batch8_full",
    "fp32_source_bf16_projection_fp32_durable_batch64_mixed",
    "fp32_source_bf16_projection_fp32_durable_batch64_full",
)
DURABLE_ROW_ORDER = (
    "fp32_source_bf16_projection_fp32_durable_batch64_full",
    "fp32_source_bf16_projection_fp32_durable_batch64_mixed",
    "fp32_source_bf16_projection_fp32_durable_batch8_full",
    "fp32_source_bf16_projection_fp32_durable_batch8_mixed",
)
DTYPE_FIELDS = (
    "identity_dtype",
    "context_dtype",
    "keys_dtype",
    "values_dtype",
    "strengths_dtype",
    "valid_dtype",
    "projected_query_dtype",
    "similarity_dtype",
    "normalized_keys_dtype",
)
TRUE_GATE_FIELDS = (
    "selection_semantically_equivalent",
    "pre_out_recalled_close",
    "final_out_close",
    "query_and_normalized_keys_bit_exact",
    "source_unchanged",
    "finite",
    "dtype_device_shape_exact",
    "direct_tail_topology_pass",
    "full_backend_no_reference_tail_ops",
)


def issue619_protocol() -> dict[str, Any]:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "source_issue": 602,
        "source_trigger": ISSUE602_TRIGGER,
        "source_run": ISSUE602_RUN,
        "source_job": ISSUE602_JOB,
        "source_result_path": ISSUE602_RESULT_PATH,
        "source_result_sha256": ISSUE602_RESULT_SHA256,
        "source_decision": ISSUE602_DECISION,
        "consumed_predecessor_run": ISSUE617_RUN,
        "consumed_predecessor_job": ISSUE617_JOB,
        "durable_json_sort_keys": True,
        "logical_row_count": 4,
        "durable_row_order": list(DURABLE_ROW_ORDER),
        "read_only_existing_json": True,
        "stored_values_only": True,
        "dtype_inference_or_recomputation": False,
        "gpu_authorized": False,
        "model_execution_authorized": False,
        "checkpoint_execution_authorized": False,
        "corpus_access_authorized": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "backward_authorized": False,
        "result_mutation_authorized": False,
        "volume_commit_authorized": False,
        "repair_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
        "scientific_seed_consumed": False,
    }


def cpu_contract_preflight_issue619() -> dict[str, Any]:
    protocol = issue619_protocol()
    if len(LOGICAL_ROWS) != 4 or len(set(LOGICAL_ROWS)) != 4:
        raise RuntimeError("issue619 logical row set drifted")
    if tuple(sorted(LOGICAL_ROWS)) != DURABLE_ROW_ORDER:
        raise RuntimeError("issue619 durable sorted row order drifted")
    if len(DTYPE_FIELDS) != 9:
        raise RuntimeError("issue619 dtype field set drifted")
    forbidden = (
        "gpu_authorized",
        "model_execution_authorized",
        "checkpoint_execution_authorized",
        "corpus_access_authorized",
        "training_authorized",
        "optimizer_authorized",
        "backward_authorized",
        "result_mutation_authorized",
        "volume_commit_authorized",
        "repair_authorized",
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
        raise RuntimeError("issue619 unexpectedly authorizes higher work")
    return {
        "protocol": protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "volume_mutated": False,
        "scientific_seed_consumed": False,
    }


def inspect_issue602_result_sorted(result: dict[str, Any]) -> dict[str, Any]:
    """Validate #602's stored classification and return existing evidence only."""

    if result.get("decision") != "FAIL" or result.get("overall_pass") is not False:
        raise RuntimeError("issue619 source result is not the frozen #602 FAIL")
    preserved = result.get("preserved_issue558")
    if not isinstance(preserved, dict) or preserved.get("overall_pass") is not True:
        raise RuntimeError("issue619 source no longer preserves authoritative #558 PASS")
    integrated = result.get("integrated")
    if not isinstance(integrated, dict):
        raise RuntimeError("issue619 source missing integrated section")
    if integrated.get("overall_pass") is not False or integrated.get("rows_pass") is not False:
        raise RuntimeError("issue619 source integrated classification drifted")
    rows = integrated.get("rows")
    if not isinstance(rows, dict):
        raise RuntimeError("issue619 source integrated rows are not an object")
    observed_order = tuple(rows.keys())
    if observed_order != DURABLE_ROW_ORDER:
        raise RuntimeError(
            f"issue619 durable row order drifted: got={observed_order!r} "
            f"expected={DURABLE_ROW_ORDER!r}"
        )
    if set(rows.keys()) != set(LOGICAL_ROWS) or len(rows) != 4:
        raise RuntimeError("issue619 source integrated row set drifted")

    extracted: dict[str, dict[str, Any]] = {}
    for key in LOGICAL_ROWS:
        row = rows[key]
        if not isinstance(row, dict):
            raise RuntimeError(f"issue619 row {key} is not an object")
        if row.get("pass") is not False or row.get("dtype_split_exact") is not False:
            raise RuntimeError(f"issue619 row {key} classification drifted")
        failed_non_dtype = [
            field for field in TRUE_GATE_FIELDS if row.get(field) is not True
        ]
        if failed_non_dtype:
            raise RuntimeError(
                f"issue619 row {key} has unexpected non-dtype failures: {failed_non_dtype}"
            )
        missing = [field for field in DTYPE_FIELDS if field not in row]
        if missing:
            raise RuntimeError(f"issue619 row {key} missing stored dtype fields: {missing}")
        extracted[key] = {
            "batch_size": row["batch_size"],
            "validity_kind": row["validity_kind"],
            **{field: row[field] for field in DTYPE_FIELDS},
            "dtype_split_exact": row["dtype_split_exact"],
            **{field: row[field] for field in TRUE_GATE_FIELDS},
            "pre_out_max_abs_diff": row["pre_out_max_abs_diff"],
            "final_out_max_abs_diff": row["final_out_max_abs_diff"],
            "atol": row["atol"],
            "rtol": row["rtol"],
            "timing_decision_bearing": row["timing_decision_bearing"],
        }

    return {
        "protocol": issue619_protocol(),
        "source": {
            "issue": 602,
            "trigger": ISSUE602_TRIGGER,
            "run": ISSUE602_RUN,
            "job": ISSUE602_JOB,
            "result_path": ISSUE602_RESULT_PATH,
            "result_sha256": ISSUE602_RESULT_SHA256,
            "decision": result["decision"],
            "overall_pass": result["overall_pass"],
            "preserved_issue558_overall_pass": preserved["overall_pass"],
            "integrated_overall_pass": integrated["overall_pass"],
        },
        "observed_durable_row_order": list(observed_order),
        "expected_durable_row_order": list(DURABLE_ROW_ORDER),
        "row_set_exact": True,
        "rows": extracted,
        "row_count": len(extracted),
        "stored_values_only": True,
        "dtype_inference_or_recomputation": False,
        "gpu_executed": False,
        "model_executed": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "volume_mutated": False,
        "scientific_seed_consumed": False,
        "repair_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
