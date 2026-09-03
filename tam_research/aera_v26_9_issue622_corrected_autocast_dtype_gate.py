from __future__ import annotations

"""Issue #622 corrected production-autocast dtype gate for AERA-v26.9 READ.

This successor does not rerun the frozen #558/#602 fixtures. It reads the
immutable #602 result only for preserved #558 authority and runs four fresh
synthetic rows from the preregistered design-only seed. Numerical, selection,
tie, topology, source-immutability and tolerance logic is delegated to the
frozen #602 row evaluator; only the preregistered dtype predicate is corrected.
"""

from pathlib import Path
from typing import Any
import hashlib
import json

import torch

from . import aera_v26_9_issue602_identity_weight_visibility_probe as frozen602
from . import aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as v26_9
from .aera_hardware_core_v26 import TorchFICEMReferenceBackend

RESEARCH_ISSUE = 622
SOURCE_MAIN = "caa7b019e9232d607d69b0e422e6d9550d675ff4"
SOURCE_TREE = "fd76f479a16036bcc81d3e48ba70956fc79c409e"
DESIGN_SEED = 891_475_817

ISSUE602_RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"
ISSUE602_RESULT_SHA256 = "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"
ISSUE602_PROBE_BLOB = "456203f515d67d1c92b0a9c3e0e59ce4137ac10a"
V26_9_BACKEND_BLOB = "b81cc209f5d95abbe1fb8bd620c78e87c067bc19"

D_MODEL = 200
MEMORY_DIM = 50
CAPACITY = 48
TIME = 256
BATCH_SIZES = (8, 64)
VALIDITY_KINDS = ("mixed", "full")
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2

_NON_DTYPE_PASS_KEYS = (
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


def issue622_protocol() -> dict[str, Any]:
    return {
        "probe_version": "aera-v26.9-issue622-corrected-autocast-dtype-gate",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "design_seed": DESIGN_SEED,
        "design_seed_only": True,
        "issue602_result_path": ISSUE602_RESULT_PATH,
        "issue602_result_sha256": ISSUE602_RESULT_SHA256,
        "issue602_probe_blob": ISSUE602_PROBE_BLOB,
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
        "issue602_remains_authoritative_fail": True,
        "issue558_fixtures_rerun": False,
        "issue602_fixtures_rerun": False,
        "fresh_rows": 4,
        "batch_sizes": BATCH_SIZES,
        "validity_kinds": VALIDITY_KINDS,
        "d_model": D_MODEL,
        "memory_dim": MEMORY_DIM,
        "capacity": CAPACITY,
        "time": TIME,
        "identity_dtype": "float32",
        "context_dtype": "float32",
        "projected_query_dtype": "float32",
        "similarity_dtype": "bfloat16",
        "durable_keys_dtype": "float32",
        "durable_values_dtype": "float32",
        "durable_strengths_dtype": "float32",
        "valid_dtype": "bool",
        "normalized_keys_dtype": "float32",
        "atol": BF16_ATOL,
        "rtol": BF16_RTOL,
        "timing_decision_bearing": False,
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


def cpu_contract_preflight_issue622() -> dict[str, Any]:
    protocol = issue622_protocol()
    if DESIGN_SEED != 891_475_817:
        raise RuntimeError("issue622 design-only seed drifted")
    if (D_MODEL, MEMORY_DIM, CAPACITY, TIME) != (200, 50, 48, 256):
        raise RuntimeError("issue622 geometry drifted")
    if BATCH_SIZES != (8, 64) or VALIDITY_KINDS != ("mixed", "full"):
        raise RuntimeError("issue622 row grid drifted")
    if (BF16_ATOL, BF16_RTOL) != (1e-2, 1e-2):
        raise RuntimeError("issue622 tolerance drifted")
    if frozen602.DESIGN_SEED == DESIGN_SEED:
        raise RuntimeError("issue622 fresh design-only seed unexpectedly reuses issue602 seed")
    if protocol["issue558_fixtures_rerun"] or protocol["issue602_fixtures_rerun"]:
        raise RuntimeError("issue622 must not rerun frozen fixtures")
    false_keys = (
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
    if any(protocol[key] for key in false_keys):
        raise RuntimeError("issue622 CPU contract unexpectedly authorizes higher work")
    return {
        "protocol": protocol,
        "gpu_authorized_by_cpu_preflight": False,
        "synthetic_only": True,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "corpus_accessed": False,
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "scientific_seed_consumed": False,
    }


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_issue602_preserved_authority(
    path: str | Path = ISSUE602_RESULT_PATH,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"issue622 missing immutable issue602 result: {source}")
    digest = _sha256_path(source)
    if digest != ISSUE602_RESULT_SHA256:
        raise RuntimeError(
            f"issue622 issue602 result SHA drift: got={digest} expected={ISSUE602_RESULT_SHA256}"
        )
    payload = json.loads(source.read_text())
    if payload.get("research_issue") != 602:
        raise RuntimeError("issue622 source result is not issue602")
    if payload.get("decision") != "FAIL" or payload.get("overall_pass") is not False:
        raise RuntimeError("issue622 must preserve issue602 authoritative FAIL")
    gate_meta = payload.get("issue602_gate_metadata", {})
    if gate_meta.get("v26_9_backend_blob") != V26_9_BACKEND_BLOB:
        raise RuntimeError("issue622 issue602 backend authority drifted")

    preserved = payload.get("preserved_issue558")
    if not isinstance(preserved, dict):
        raise RuntimeError("issue622 missing preserved issue558 authority")
    historical = preserved.get("historical", {})
    mixed = preserved.get("mixed", {})
    required_historical = (
        "overall_pass",
        "correctness_pass",
        "known_empty_pass",
        "near_tie_pass",
        "row_latency_pass",
        "full_event_ratio_pass",
        "single_tail_kernel_pass",
        "candidate_no_reference_tail_ops_pass",
    )
    if preserved.get("decision") != "PASS" or preserved.get("overall_pass") is not True:
        raise RuntimeError("issue622 preserved issue558 overall authority is not PASS")
    if historical.get("decision") != "PASS":
        raise RuntimeError("issue622 preserved issue558 historical decision drifted")
    if any(historical.get(key) is not True for key in required_historical):
        raise RuntimeError("issue622 preserved issue558 historical gate drifted")
    if any(
        mixed.get(key) is not True
        for key in ("overall_pass", "rows_pass", "near_tie_pass", "known_empty_pass")
    ):
        raise RuntimeError("issue622 preserved issue558 mixed gate drifted")
    if mixed.get("timing_decision_bearing") is not False:
        raise RuntimeError("issue622 preserved mixed timing contract drifted")

    return {
        "source_result_sha256": digest,
        "issue602_decision": "FAIL",
        "issue602_overall_pass": False,
        "preserved_issue558": {
            "decision": preserved["decision"],
            "overall_pass": preserved["overall_pass"],
            "historical": {
                key: historical[key]
                for key in (
                    "decision",
                    *required_historical,
                    "geomean_latency_ratio_by_dtype",
                    "geomean_latency_pass_by_dtype",
                )
            },
            "mixed": {
                key: mixed[key]
                for key in (
                    "overall_pass",
                    "rows_pass",
                    "near_tie_pass",
                    "known_empty_pass",
                    "timing_decision_bearing",
                )
            },
        },
    }


def _corrected_row(
    memory,
    reference: TorchFICEMReferenceBackend,
    candidate: v26_9.IdentityWeightVisibilityTritonFICEMReadWriteBackend,
    case,
) -> dict[str, Any]:
    # Delegate all numerical, selection/tie, reuse, immutability and topology
    # evidence to the frozen #602 evaluator. Only its preregistered dtype
    # predicate was wrong; no threshold or numerical rule is changed here.
    row = dict(frozen602._integrated_row(memory, reference, candidate, case))
    issue602_dtype_split_exact = bool(row["dtype_split_exact"])
    corrected_dtype_split_exact = bool(
        row["identity_dtype"] == "torch.float32"
        and row["context_dtype"] == "torch.float32"
        and row["keys_dtype"] == "torch.float32"
        and row["values_dtype"] == "torch.float32"
        and row["strengths_dtype"] == "torch.float32"
        and row["valid_dtype"] == "torch.bool"
        and row["projected_query_dtype"] == "torch.float32"
        and row["similarity_dtype"] == "torch.bfloat16"
        and row["normalized_keys_dtype"] == "torch.float32"
    )
    non_dtype_pass = all(bool(row[key]) for key in _NON_DTYPE_PASS_KEYS)
    row["issue602_dtype_split_exact"] = issue602_dtype_split_exact
    row["dtype_split_exact"] = corrected_dtype_split_exact
    row["non_dtype_pass"] = non_dtype_pass
    row["pass"] = bool(corrected_dtype_split_exact and non_dtype_pass)
    row["corrected_dtype_contract"] = (
        "fp32-identity-context-query-durable-normalized-keys/"
        "bf16-similarity/bool-valid"
    )
    return row


def run_corrected_autocast_dtype_gate_v26_9_issue622(
    *,
    issue602_result_path: str | Path = ISSUE602_RESULT_PATH,
) -> dict[str, Any]:
    cpu_contract_preflight_issue622()
    if not torch.cuda.is_available():
        raise RuntimeError("issue622 requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        raise RuntimeError(f"issue622 requires NVIDIA L4, found {device_name}")

    authority = load_issue602_preserved_authority(issue602_result_path)

    memory = frozen602.frozen.build_memory(device)
    reference = TorchFICEMReferenceBackend()
    candidate = v26_9.IdentityWeightVisibilityTritonFICEMReadWriteBackend()
    generator = torch.Generator().manual_seed(DESIGN_SEED)

    rows: dict[str, dict[str, Any]] = {}
    for batch_size in BATCH_SIZES:
        for validity_kind in VALIDITY_KINDS:
            case = frozen602._make_integrated_case(
                batch_size=batch_size,
                validity_kind=validity_kind,
                generator=generator,
                device=device,
            )
            key = (
                "fresh_fp32_identity_fp32_query_bf16_similarity_fp32_durable_"
                f"batch{batch_size}_{validity_kind}"
            )
            rows[key] = _corrected_row(memory, reference, candidate, case)
            del case

    rows_pass = all(row["pass"] for row in rows.values())
    overall_pass = bool(authority["preserved_issue558"]["overall_pass"] and rows_pass)
    return {
        "protocol": issue622_protocol(),
        "device": device_name,
        "issue602_authority": authority,
        "fresh_integrated": {
            "rows": rows,
            "rows_pass": rows_pass,
            "overall_pass": rows_pass,
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
