from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

RESEARCH_ISSUE = 522
SOURCE_MAIN = "b4c117b01ae851327e9feca25ea3c12078831904"
APP_NAME = "tam-research-aera-v26-6-issue522-readonly-issue519-result-inspector"
VOLUME_NAME = "tam-research-data"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue519-ficem-write-materialize-cast/result.json"
SOURCE_RESULT_SHA256 = "b9fba0fca96644ef8db9bc46faf2c73d0c0cc1f1aaac6a321abe2411d3703cd5"
SOURCE_RUN = 33672232063
SOURCE_JOB = 100388368044
SOURCE_ATTEMPT = 1
CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
PROBE_BLOB = "ec22807434192f58e292bffc3de9828be2b44272"
RESULT_MARKER = "AERA_V26_6_ISSUE522_READONLY_ISSUE519_RESULT_INSPECTOR_RESULT_JSON="

FROZEN_PASS_MASKS = (
    0, 1, 2, 3, 36, 37, 38, 39, 72, 73, 74, 75, 108, 109, 110, 111,
    144, 145, 146, 147, 180, 181, 182, 183, 216, 217, 218, 219,
    252, 253, 254, 255,
)
REPRESENTATIVE_MASKS = (0, 4, 8, 16, 32, 36, 64, 72, 128, 144, 252, 255)
FIELD_DEST_DTYPE_INDEX = {"keys": 5, "values": 6, "strengths": 7}
FIELDS = ("keys", "values", "strengths")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


def _mask_from_key(key: str) -> int:
    if not key.startswith("mask_"):
        raise ValueError(f"unexpected direct-matrix key: {key!r}")
    return int(key.split("_", 1)[1])


def _destination_dtype(row: dict[str, Any], field: str) -> str | None:
    dtypes = row.get("dtypes")
    index = FIELD_DEST_DTYPE_INDEX[field]
    if not isinstance(dtypes, list) or len(dtypes) <= index:
        return None
    value = dtypes[index]
    return str(value) if value is not None else None


def _new_dtype(row: dict[str, Any], field: str) -> str | None:
    index = {"keys": 2, "values": 3, "strengths": 4}[field]
    dtypes = row.get("dtypes")
    if not isinstance(dtypes, list) or len(dtypes) <= index:
        return None
    value = dtypes[index]
    return str(value) if value is not None else None


def _classify(rows: dict[str, Any], *, direct: bool) -> dict[str, Any]:
    categories: dict[str, list[Any]] = {
        "exception_or_error": [],
        "source_mutation": [],
        "validity_mismatch": [],
        "dtype_mismatch": [],
        "device_or_shape_mismatch": [],
        "nonfinite": [],
    }
    per_field: dict[str, dict[str, Any]] = {
        field: {
            "float_close_failures": [],
            "dtype_exact_failures": [],
            "pair_dtype_mismatch_rows": [],
            "max_abs_by_destination_dtype": {},
        }
        for field in FIELDS
    }

    passed: list[Any] = []
    failed: list[Any] = []
    for key, raw_row in rows.items():
        row = raw_row if isinstance(raw_row, dict) else {"pass": False, "error": "row is not an object"}
        identity: Any = _mask_from_key(key) if direct else key
        if bool(row.get("pass", False)):
            passed.append(identity)
        else:
            failed.append(identity)

        if "error" in row or "error_type" in row:
            categories["exception_or_error"].append(identity)
        if row.get("source_unchanged") is False:
            categories["source_mutation"].append(identity)
        if row.get("valid_exact") is False:
            categories["validity_mismatch"].append(identity)
        dtype_exact = row.get("dtype_exact")
        if isinstance(dtype_exact, dict) and any(dtype_exact.get(field) is False for field in FIELDS):
            categories["dtype_mismatch"].append(identity)
        if row.get("device_shape_exact") is False:
            categories["device_or_shape_mismatch"].append(identity)
        if row.get("finite") is False:
            categories["nonfinite"].append(identity)

        float_close = row.get("float_close")
        max_abs = row.get("max_abs")
        for field in FIELDS:
            if isinstance(float_close, dict) and float_close.get(field) is False:
                per_field[field]["float_close_failures"].append(identity)
            if isinstance(dtype_exact, dict) and dtype_exact.get(field) is False:
                per_field[field]["dtype_exact_failures"].append(identity)
            new_dtype = _new_dtype(row, field)
            destination_dtype = _destination_dtype(row, field)
            if new_dtype is not None and destination_dtype is not None and new_dtype != destination_dtype:
                per_field[field]["pair_dtype_mismatch_rows"].append(identity)
            if isinstance(max_abs, dict) and field in max_abs and destination_dtype is not None:
                try:
                    value = float(max_abs[field])
                except (TypeError, ValueError):
                    pass
                else:
                    previous = per_field[field]["max_abs_by_destination_dtype"].get(destination_dtype)
                    if previous is None or value > previous:
                        per_field[field]["max_abs_by_destination_dtype"][destination_dtype] = value

    for values in categories.values():
        values.sort()
    for field in FIELDS:
        for key in ("float_close_failures", "dtype_exact_failures", "pair_dtype_mismatch_rows"):
            per_field[field][key].sort()
        per_field[field]["float_close_failure_count"] = len(per_field[field]["float_close_failures"])
        per_field[field]["dtype_exact_failure_count"] = len(per_field[field]["dtype_exact_failures"])
        per_field[field]["pair_dtype_mismatch_count"] = len(per_field[field]["pair_dtype_mismatch_rows"])

    passed.sort()
    failed.sort()
    return {
        "row_count": len(rows),
        "pass_count": len(passed),
        "fail_count": len(failed),
        "passed": passed,
        "failed": failed,
        "categories": categories,
        "category_counts": {name: len(values) for name, values in categories.items()},
        "per_field": per_field,
    }


def _inspect_payload(result: dict[str, Any]) -> dict[str, Any]:
    direct = result.get("direct_matrix")
    edges = result.get("edge_cases")
    public = result.get("public_rows")
    topology = result.get("topology_rows")
    if not all(isinstance(value, dict) for value in (direct, edges, public, topology)):
        raise RuntimeError("issue519 durable result is missing expected row dictionaries")

    direct_classification = _classify(direct, direct=True)
    edge_classification = _classify(edges, direct=False)
    actual_pass_masks = tuple(direct_classification["passed"])
    frozen_pass_masks_match = actual_pass_masks == FROZEN_PASS_MASKS

    representative_rows = {
        f"mask_{mask:03d}": direct[f"mask_{mask:03d}"] for mask in REPRESENTATIVE_MASKS
    }

    return {
        "research_issue": RESEARCH_ISSUE,
        "inspection_kind": "read_only_existing_durable_result",
        "source_run": SOURCE_RUN,
        "source_job": SOURCE_JOB,
        "source_attempt": SOURCE_ATTEMPT,
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_decision": result.get("decision"),
        "source_overall_pass": result.get("overall_pass"),
        "source_device": result.get("device"),
        "candidate_blob": CANDIDATE_BLOB,
        "probe_blob": PROBE_BLOB,
        "frozen_pass_masks": list(FROZEN_PASS_MASKS),
        "actual_pass_masks": list(actual_pass_masks),
        "frozen_pass_masks_match": frozen_pass_masks_match,
        "direct_classification": direct_classification,
        "representative_direct_rows": representative_rows,
        "edge_classification": edge_classification,
        "edge_rows": edges,
        "public_rows": public,
        "topology_rows": topology,
        "public_all_pass": all(bool(row.get("pass", False)) for row in public.values()),
        "topology_all_pass": all(bool(row.get("pass", False)) for row in topology.values()),
        "volume_mutated": False,
        "gpu_used": False,
        "experiment_rerun": False,
        "repair_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(image=image, cpu=1, memory=512, timeout=120, volumes={"/vol": volume})
def inspect_existing_issue519_result() -> dict[str, Any]:
    volume.reload()
    path = Path(SOURCE_RESULT_PATH)
    if not path.is_file():
        raise RuntimeError(f"frozen issue519 durable result is missing: {SOURCE_RESULT_PATH}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_RESULT_SHA256:
        raise RuntimeError(
            f"issue519 durable SHA mismatch: got={digest} expected={SOURCE_RESULT_SHA256}"
        )
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("issue519 durable result root must be an object")
    inspection = _inspect_payload(parsed)
    inspection["verified_source_bytes"] = len(raw)
    return inspection


@app.local_entrypoint()
def main() -> None:
    result = inspect_existing_issue519_result.remote()
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
