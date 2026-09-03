from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-9-issue602-identity-weight-visibility"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"
MAX_GPU_SECONDS = 300
RESEARCH_ISSUE = 602
SOURCE_MAIN = "37d2352050730c75dff0ab4b547e990b7865a95d"
SOURCE_TREE = "c3fd42879162cdc5e01b1ed0fcc34f2f82aa454f"
PROBE_BLOB = "456203f515d67d1c92b0a9c3e0e59ce4137ac10a"
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

PRECHECK_MARKER = "AERA_V26_9_ISSUE602_IDENTITY_WEIGHT_VISIBILITY_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_9_ISSUE602_IDENTITY_WEIGHT_VISIBILITY_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_9_ISSUE602_IDENTITY_WEIGHT_VISIBILITY_L4_START_JSON="
RESULT_MARKER = "AERA_V26_9_ISSUE602_IDENTITY_WEIGHT_VISIBILITY_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_9_ISSUE602_IDENTITY_WEIGHT_VISIBILITY_SUMMARY_JSON="

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def precheck() -> dict:
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "probe_blob": PROBE_BLOB,
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
        "result_path": RESULT_PATH,
        "max_gpu_seconds": MAX_GPU_SECONDS,
        "gpu": "L4",
        "gpu_remote_functions": 1,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def _summary(result: dict) -> dict:
    preserved = result["preserved_issue558"]
    historical = preserved["historical"]
    mixed = preserved["mixed"]
    integrated_rows = {
        key: {
            "pass": row["pass"],
            "batch_size": row["batch_size"],
            "validity_kind": row["validity_kind"],
            "dtype_split_exact": row["dtype_split_exact"],
            "selection_semantically_equivalent": row[
                "selection_semantically_equivalent"
            ],
            "pre_out_recalled_close": row["pre_out_recalled_close"],
            "final_out_close": row["final_out_close"],
            "query_and_normalized_keys_bit_exact": row[
                "query_and_normalized_keys_bit_exact"
            ],
            "source_unchanged": row["source_unchanged"],
            "finite": row["finite"],
            "dtype_device_shape_exact": row["dtype_device_shape_exact"],
            "direct_tail_topology_pass": row["direct_tail_topology_pass"],
            "full_backend_no_reference_tail_ops": row[
                "full_backend_no_reference_tail_ops"
            ],
            "pre_out_max_abs_diff": row["pre_out_max_abs_diff"],
            "final_out_max_abs_diff": row["final_out_max_abs_diff"],
            "timing_decision_bearing": row["timing_decision_bearing"],
        }
        for key, row in result["integrated"]["rows"].items()
    }
    return {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "preserved_issue558": {
            "decision": preserved["decision"],
            "overall_pass": preserved["overall_pass"],
            "historical": {
                "decision": historical["decision"],
                "overall_pass": historical["overall_pass"],
                "correctness_pass": historical["correctness_pass"],
                "known_empty_pass": historical["known_empty_pass"],
                "near_tie_pass": historical["near_tie_pass"],
                "row_latency_pass": historical["row_latency_pass"],
                "full_event_ratio_pass": historical["full_event_ratio_pass"],
                "single_tail_kernel_pass": historical["single_tail_kernel_pass"],
                "candidate_no_reference_tail_ops_pass": historical[
                    "candidate_no_reference_tail_ops_pass"
                ],
                "geomean_latency_ratio_by_dtype": historical[
                    "geomean_latency_ratio_by_dtype"
                ],
                "geomean_latency_pass_by_dtype": historical[
                    "geomean_latency_pass_by_dtype"
                ],
            },
            "mixed": {
                "overall_pass": mixed["overall_pass"],
                "rows_pass": mixed["rows_pass"],
                "near_tie_pass": mixed["near_tie_pass"],
                "known_empty_pass": mixed["known_empty_pass"],
                "timing_decision_bearing": mixed["timing_decision_bearing"],
            },
        },
        "integrated": {
            "overall_pass": result["integrated"]["overall_pass"],
            "rows_pass": result["integrated"]["rows_pass"],
            "timing_decision_bearing": result["integrated"][
                "timing_decision_bearing"
            ],
            "rows": integrated_rows,
        },
        "scientific_seed_consumed": result["scientific_seed_consumed"],
        "end_to_end_systems_authorized": result["end_to_end_systems_authorized"],
        "architecture_freeze_authorized": result["architecture_freeze_authorized"],
        "s2_authorized": result["s2_authorized"],
        "fresh_scientific_seed_authorized": result[
            "fresh_scientific_seed_authorized"
        ],
        "independent_replication_credit": result["independent_replication_credit"],
        "100m_authorized": result["100m_authorized"],
        "breakthrough_proven": result["breakthrough_proven"],
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v25 as factorized_v25
    import tam_research.aera_hardware_core_v25_1_compact as stable
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as repair5
    import tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast as write_backend
    import tam_research.aera_hardware_core_v26_8_ficem_read_mixed_strength_precision as v26_8
    import tam_research.aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as candidate
    import tam_research.aera_v26_3_ficem_read_probe as historical_probe
    import tam_research.aera_v26_7_issue553_ficem_read_mixed_dtype_probe as issue553
    import tam_research.aera_v26_8_issue558_ficem_read_mixed_strength_precision_probe as issue558
    import tam_research.aera_v26_9_issue602_identity_weight_visibility_probe as probe

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue602 run because durable result exists: {RESULT_PATH}"
        )

    blobs = {
        "probe": _git_blob_sha(Path(probe.__file__)),
        "v26_9_backend": _git_blob_sha(Path(candidate.__file__)),
        "v26_8_backend": _git_blob_sha(Path(v26_8.__file__)),
        "issue558_probe": _git_blob_sha(Path(issue558.__file__)),
        "frozen_issue553_probe": _git_blob_sha(Path(issue553.__file__)),
        "historical_probe": _git_blob_sha(Path(historical_probe.__file__)),
        "repair5_backend": _git_blob_sha(Path(repair5.__file__)),
        "v26_6_write": _git_blob_sha(Path(write_backend.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
        "factorized_v25": _git_blob_sha(Path(factorized_v25.__file__)),
    }
    expected = {
        "probe": PROBE_BLOB,
        "v26_9_backend": V26_9_BACKEND_BLOB,
        "v26_8_backend": V26_8_BACKEND_BLOB,
        "issue558_probe": ISSUE558_PROBE_BLOB,
        "frozen_issue553_probe": FROZEN_ISSUE553_PROBE_BLOB,
        "historical_probe": HISTORICAL_PROBE_BLOB,
        "repair5_backend": REPAIR5_BACKEND_BLOB,
        "v26_6_write": V26_6_WRITE_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
        "factorized_v25": FACTORIZED_V25_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue602 frozen blob drift: got={blobs} expected={expected}")

    contract = probe.cpu_contract_preflight_issue602()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue602 CPU preflight unexpectedly authorizes GPU")
    protocol = probe.issue602_protocol()
    required = {
        "research_issue": RESEARCH_ISSUE,
        "source_main_issue602": SOURCE_MAIN,
        "source_tree_issue602": SOURCE_TREE,
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
        "issue558_surface_preserved_wholesale": True,
        "issue558_thresholds_relaxed": False,
        "integrated_rows": 4,
        "integrated_identity_dtype": "float32",
        "integrated_context_dtype": "float32",
        "integrated_durable_dtype": "float32",
        "integrated_projected_query_dtype": "bfloat16",
        "integrated_similarity_dtype": "bfloat16",
        "integrated_normalized_keys_dtype": "float32",
        "integrated_atol": 1e-2,
        "integrated_rtol": 1e-2,
        "integrated_timing_decision_bearing": False,
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
    for key, expected_value in required.items():
        if protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue602 probe protocol drift: {key}={protocol.get(key)!r} "
                f"expected={expected_value!r}"
            )

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "blobs": blobs,
        "result_path_absent": True,
        "scientific_seed_consumed": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_gate() -> dict:
    import torch
    from tam_research.aera_v26_9_issue602_identity_weight_visibility_probe import (
        run_identity_weight_visibility_gate_v26_9_issue602,
    )

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(
            f"refusing duplicate issue602 run because durable result exists: {RESULT_PATH}"
        )

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "device": torch.cuda.get_device_name(0),
                "source_main": SOURCE_MAIN,
                "probe_blob": PROBE_BLOB,
                "v26_9_backend_blob": V26_9_BACKEND_BLOB,
                "scientific_seed_consumed": False,
                "end_to_end_systems_authorized": False,
                "architecture_freeze_authorized": False,
                "s2_authorized": False,
                "fresh_scientific_seed_authorized": False,
                "independent_replication_credit": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_identity_weight_visibility_gate_v26_9_issue602()
    result["issue602_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "probe_blob": PROBE_BLOB,
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
        "scientific_seed_consumed": False,
    }

    durable_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(durable_json)
    volume.commit()
    digest = hashlib.sha256(durable_json.encode()).hexdigest()

    marker = {
        "research_issue": RESEARCH_ISSUE,
        "result_path": RESULT_PATH,
        "result_sha256": digest,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "source_main": SOURCE_MAIN,
        "probe_blob": PROBE_BLOB,
        "v26_9_backend_blob": V26_9_BACKEND_BLOB,
    }
    print(RESULT_MARKER + json.dumps(marker, separators=(",", ":")), flush=True)
    print(
        SUMMARY_MARKER + json.dumps(_summary(result), separators=(",", ":")),
        flush=True,
    )
    return result


@app.local_entrypoint()
def main() -> None:
    print(PRECHECK_MARKER + json.dumps(precheck(), separators=(",", ":")), flush=True)
    check = preflight.remote()
    print(PREFLIGHT_MARKER + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
