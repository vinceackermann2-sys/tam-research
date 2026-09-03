from __future__ import annotations

"""CPU-only read-only extractor for the immutable #602 primitive READ result."""

from typing import Any

RESEARCH_ISSUE = 605
SOURCE_MAIN = "b20b88d35f2afa044eb09a70687852bbc6284f06"
SOURCE_TREE = "104900cab12f085e77ae883fb2311f2f2bea4ebe"
ISSUE602_TRIGGER = 604
ISSUE602_RUN = 33783777048
ISSUE602_JOB = 100743509172
ISSUE602_RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"
ISSUE602_RESULT_SHA256 = "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"
ISSUE602_DECISION = "FAIL"

EXPECTED_ROWS = (
    "fp32_source_bf16_projection_fp32_durable_batch8_mixed",
    "fp32_source_bf16_projection_fp32_durable_batch8_full",
    "fp32_source_bf16_projection_fp32_durable_batch64_mixed",
    "fp32_source_bf16_projection_fp32_durable_batch64_full",
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


def issue605_protocol() -> dict[str, Any]:
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


def cpu_contract_preflight_issue605() -> dict[str, Any]:
    protocol = issue605_protocol()
    if len(EXPECTED_ROWS) != 4:
        raise RuntimeError("issue605 row set drifted")
    if len(DTYPE_FIELDS) != 9:
        raise RuntimeError("issue605 dtype field set drifted")
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
        raise RuntimeError("issue605 unexpectedly authorizes higher work")
    return {
        "protocol": protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "volume_mutated": False,
        "scientific_seed_consumed": False,
    }


def inspect_issue602_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate #602's stored classification and return only existing evidence."""

    if result.get("decision") != "FAIL" or result.get("overall_pass") is not False:
        raise RuntimeError("issue605 source result is not the frozen #602 FAIL")
    preserved = result.get("preserved_issue558")
    if not isinstance(preserved, dict) or preserved.get("overall_pass") is not True:
        raise RuntimeError("issue605 source no longer preserves authoritative #558 PASS")
    integrated = result.get("integrated")
    if not isinstance(integrated, dict):
        raise RuntimeError("issue605 source missing integrated section")
    if integrated.get("overall_pass") is not False or integrated.get("rows_pass") is not False:
        raise RuntimeError("issue605 source integrated classification drifted")
    rows = integrated.get("rows")
    if not isinstance(rows, dict) or tuple(rows.keys()) != EXPECTED_ROWS:
        raise RuntimeError("issue605 source integrated row order/set drifted")

    extracted: dict[str, dict[str, Any]] = {}
    for key in EXPECTED_ROWS:
        row = rows[key]
        if not isinstance(row, dict):
            raise RuntimeError(f"issue605 row {key} is not an object")
        if row.get("pass") is not False or row.get("dtype_split_exact") is not False:
            raise RuntimeError(f"issue605 row {key} classification drifted")
        failed_non_dtype = [
            field for field in TRUE_GATE_FIELDS if row.get(field) is not True
        ]
        if failed_non_dtype:
            raise RuntimeError(
                f"issue605 row {key} has unexpected non-dtype failures: {failed_non_dtype}"
            )
        missing = [field for field in DTYPE_FIELDS if field not in row]
        if missing:
            raise RuntimeError(f"issue605 row {key} missing stored dtype fields: {missing}")
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
        "protocol": issue605_protocol(),
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
