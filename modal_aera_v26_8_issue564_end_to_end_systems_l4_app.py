from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import modal

APP_NAME = "tam-research-aera-v26-8-issue564-end-to-end-systems"
VOLUME_NAME = "tam-research-data"
RESULT_PATH = "/vol/aera-v26/issue564-end-to-end-systems/result.json"
MAX_GPU_SECONDS = 600
RESEARCH_ISSUE = 564
SOURCE_MAIN = "4277ae6e1f8267be3256c1c49c41835f78fe3147"
SOURCE_TREE = "56b61b639fd2ce3616b672d74b4fdf0f7736e278"

ADAPTER_BLOB = "3534103eea21f7c4d9d31798ad34601fd47090d6"
ADAPTER_CPU_TEST_BLOB = "2893a86cbdf767cdfa92601503d107d5ca3912fb"
BASE_SYSTEMS_BLOB = "c9731cae7e386f09b2a190b045532591c4fa00be"
REPAIR1_SYSTEMS_BLOB = "b3f7082b188644007b873db3733492f424d4941a"
ISSUE530_SYSTEMS_BLOB = "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b"
V26_8_CANDIDATE_BLOB = "3575c58d1cd730be77649f087908c51dbf3e6088"
V26_7_PREDECESSOR_BLOB = "d8133c6b204b1ee5f23955255fb2fb09d09bd723"
REPAIR5_READ_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_6_WRITE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
HISTORICAL_V26_4_WRITE_BLOB = "e54570292489bd17570038dca7518419ac00418c"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"

ISSUE562_HEAD = "5bb71886ab513cc73b0bc991b318a43d045b6210"
ISSUE562_CPU_RUN = 33734411109
ISSUE562_CPU_JOB = 100581497823
ISSUE562_MERGE = SOURCE_MAIN
ISSUE558_TRIGGER = 561
ISSUE558_RUN = 33733085825
ISSUE558_JOB = 100577290103
ISSUE558_RESULT_SHA256 = "e1fdc7e6b69a33084ca4b419b5489e755d7a98b12c367775ef19d1127700aa7e"
ISSUE545_TRIGGER = 550
ISSUE545_RUN = 33686037672
ISSUE545_JOB = 100433658768
ISSUE545_FAILURE = "FICEM read-tail floating dtypes must match"
ISSUE553_TRIGGER = 555
ISSUE553_RUN = 33727540468
ISSUE553_JOB = 100559866985
ISSUE553_RESULT_SHA256 = "009af31baf70e46eb93b6e7489d62f356a02b727521d3fabe4a7dab2dcf5ab47"
ISSUE529_RUN = 33680028132
ISSUE529_JOB = 100414089065
ISSUE529_RESULT_SHA256 = "a07790c2d55d7a696baf0e903a05dbdf925e3ef2a08b84c784c77d1fbdd31874"

CHECKPOINT_HASH_KEYS = frozenset({"aera", "transformer"})
PRECHECK_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_PRECHECK_JSON="
PREFLIGHT_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_PREFLIGHT_JSON="
L4_START_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_L4_START_JSON="
RESULT_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_8_ISSUE564_END_TO_END_SYSTEMS_SUMMARY_JSON="

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


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _summary(result: dict) -> dict:
    rows: dict[str, dict] = {}
    for batch, row in result["rows"].items():
        rows[batch] = {
            "pass": result["per_batch_pass"][batch],
            "routing_exact": row["routing_exact"],
            "logit_equivalence_pass": row["logit_equivalence"]["pass"],
            "state_equivalence_pass": row["state_equivalence"]["pass"],
            "physical_sparse_pass": row["physical_sparse"]["pass"],
            "write_geometry_pass": row["write_geometry"]["pass"],
            "finite": row["finite"],
            "persistent_state_bytes_pass": row["persistent_state_bytes_pass"],
            "candidate_full_vs_transformer_speed_ratio": row[
                "candidate_full_vs_transformer_speed_ratio"
            ],
            "required_full_speed_ratio": row["required_full_speed_ratio"],
            "throughput_pass": row["throughput_pass"],
            "reference_full_latency_ms": row["reference_full_latency_ms"],
            "candidate_full_latency_ms": row["candidate_full_latency_ms"],
            "candidate_vs_reference_latency_ratio": row[
                "candidate_vs_reference_latency_ratio"
            ],
            "no_reference_full_latency_regression": row[
                "no_reference_full_latency_regression"
            ],
        }
    return {
        "research_issue": RESEARCH_ISSUE,
        "decision": result["decision"],
        "overall_pass": result["overall_pass"],
        "device": result["device"],
        "candidate_backend_names": result.get("candidate_backend_names", []),
        "parameter_versions_unchanged": result["parameter_versions_unchanged"],
        "checkpoint_hashes_unchanged": result["checkpoint_hashes_unchanged"],
        "training_performed": result["training_performed"],
        "optimizer_created": result["optimizer_created"],
        "backward_performed": result["backward_performed"],
        "corpus_accessed": result["corpus_accessed"],
        "checkpoint_written": result["checkpoint_written"],
        "scientific_seed_consumed": result["scientific_seed_consumed"],
        "rows": rows,
        "claims": result["claims"],
    }


@app.function(image=image, cpu=4, memory=8192, timeout=300, volumes={"/vol": volume})
def preflight() -> dict:
    import tam_research.aera_hardware_core_v25_1_compact as stable
    import tam_research.aera_hardware_core_v26 as v26
    import tam_research.aera_hardware_core_v26_3_ficem_read_triton as repair5
    import tam_research.aera_hardware_core_v26_4_ficem_write_triton as historical_write
    import tam_research.aera_hardware_core_v26_6_ficem_write_materialize_cast as write_v26_6
    import tam_research.aera_hardware_core_v26_7_ficem_read_mixed_dtype as v26_7
    import tam_research.aera_hardware_core_v26_8_ficem_read_mixed_strength_precision as v26_8
    import tam_research.aera_v26_5_end_to_end_systems as base
    import tam_research.aera_v26_5_end_to_end_systems_repair1 as repair1
    import tam_research.aera_v26_6_issue530_end_to_end_systems as issue530
    import tam_research.aera_v26_8_issue562_end_to_end_systems as adapter

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate issue564 systems run because result exists: {RESULT_PATH}")

    blobs = {
        "adapter": _git_blob_sha(Path(adapter.__file__)),
        "base_systems": _git_blob_sha(Path(base.__file__)),
        "repair1_systems": _git_blob_sha(Path(repair1.__file__)),
        "issue530_systems": _git_blob_sha(Path(issue530.__file__)),
        "v26_8_candidate": _git_blob_sha(Path(v26_8.__file__)),
        "v26_7_predecessor": _git_blob_sha(Path(v26_7.__file__)),
        "repair5_read": _git_blob_sha(Path(repair5.__file__)),
        "v26_6_write": _git_blob_sha(Path(write_v26_6.__file__)),
        "historical_v26_4_write": _git_blob_sha(Path(historical_write.__file__)),
        "v26_interface": _git_blob_sha(Path(v26.__file__)),
        "stable_reference": _git_blob_sha(Path(stable.__file__)),
    }
    expected = {
        "adapter": ADAPTER_BLOB,
        "base_systems": BASE_SYSTEMS_BLOB,
        "repair1_systems": REPAIR1_SYSTEMS_BLOB,
        "issue530_systems": ISSUE530_SYSTEMS_BLOB,
        "v26_8_candidate": V26_8_CANDIDATE_BLOB,
        "v26_7_predecessor": V26_7_PREDECESSOR_BLOB,
        "repair5_read": REPAIR5_READ_BLOB,
        "v26_6_write": V26_6_WRITE_BLOB,
        "historical_v26_4_write": HISTORICAL_V26_4_WRITE_BLOB,
        "v26_interface": V26_INTERFACE_BLOB,
        "stable_reference": STABLE_REFERENCE_BLOB,
    }
    if blobs != expected:
        raise RuntimeError(f"issue564 frozen blob drift: got={blobs} expected={expected}")

    contract = adapter.cpu_contract_preflight_issue562()
    if contract["gpu_authorized_by_cpu_preflight"] is not False:
        raise RuntimeError("issue564 inherited CPU contract unexpectedly authorizes GPU")
    if contract["systems_measurement_performed"] is not False:
        raise RuntimeError("issue564 CPU preflight unexpectedly performed systems measurement")

    protocol = adapter.issue562_systems_protocol()
    required = {
        "research_issue": 562,
        "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
        "v26_8_candidate_blob": V26_8_CANDIDATE_BLOB,
        "candidate_v26_8_read_mixed_strength_precision": True,
        "candidate_v26_6_write_inherited": True,
        "candidate_write_backend_changed_by_v26_8": False,
        "candidate_training_backend_changed_by_v26_8": False,
        "frozen_issue530_run_function_reused": True,
        "frozen_issue530_loader_replaced_before_parameter_snapshot": True,
        "frozen_issue530_loader_replaced_before_any_model_call": True,
        "issue558_trigger": ISSUE558_TRIGGER,
        "issue558_run": ISSUE558_RUN,
        "issue558_job": ISSUE558_JOB,
        "issue558_result_sha256": ISSUE558_RESULT_SHA256,
        "issue558_decision": "PASS",
        "issue545_trigger": ISSUE545_TRIGGER,
        "issue545_run": ISSUE545_RUN,
        "issue545_job": ISSUE545_JOB,
        "issue545_authoritative_result_emitted": False,
        "issue545_failure": ISSUE545_FAILURE,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }
    for key, expected_value in required.items():
        if protocol.get(key) != expected_value:
            raise RuntimeError(
                f"issue564 adapter protocol drift: {key}={protocol.get(key)!r} expected={expected_value!r}"
            )

    if base.CHECKPOINT_RELATIVE_DIR != "/vol/aera-real-language/v25-dev-seed8471":
        raise RuntimeError("issue564 checkpoint path drift")
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if set(hashes) != CHECKPOINT_HASH_KEYS:
        raise RuntimeError(
            f"issue564 checkpoint hash inventory drift: got={sorted(hashes)} expected={sorted(CHECKPOINT_HASH_KEYS)}"
        )
    if not all(_valid_sha256(value) for value in hashes.values()):
        raise RuntimeError("issue564 checkpoint hash value drift")

    return {
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "blobs": blobs,
        "checkpoint_hashes": hashes,
        "checkpoint_hash_keys": sorted(hashes),
        "issue562_head": ISSUE562_HEAD,
        "issue562_cpu_run": ISSUE562_CPU_RUN,
        "issue562_cpu_job": ISSUE562_CPU_JOB,
        "issue562_merge": ISSUE562_MERGE,
        "issue558_result_sha256": ISSUE558_RESULT_SHA256,
        "issue545_authoritative_result_emitted": False,
        "result_path_absent": True,
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
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
    from tam_research.aera_v26_8_issue562_end_to_end_systems import run_end_to_end_systems_v26_8

    volume.reload()
    result_path = Path(RESULT_PATH)
    if result_path.exists():
        raise RuntimeError(f"refusing duplicate issue564 systems run because result exists: {RESULT_PATH}")

    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "device": torch.cuda.get_device_name(0),
                "source_main": SOURCE_MAIN,
                "adapter_blob": ADAPTER_BLOB,
                "v26_8_candidate_blob": V26_8_CANDIDATE_BLOB,
                "checkpoint_seed": 8471,
                "scientific_seed_consumed": False,
                "architecture_freeze_authorized": False,
                "fresh_scientific_seed_authorized": False,
                "independent_replication_credit": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    result = run_end_to_end_systems_v26_8()
    result["issue564_gate_metadata"] = {
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "adapter_blob": ADAPTER_BLOB,
        "adapter_cpu_test_blob": ADAPTER_CPU_TEST_BLOB,
        "issue530_systems_blob": ISSUE530_SYSTEMS_BLOB,
        "v26_8_candidate_blob": V26_8_CANDIDATE_BLOB,
        "v26_6_write_blob": V26_6_WRITE_BLOB,
        "issue562_head": ISSUE562_HEAD,
        "issue562_cpu_run": ISSUE562_CPU_RUN,
        "issue562_cpu_job": ISSUE562_CPU_JOB,
        "issue562_merge": ISSUE562_MERGE,
        "issue558_trigger": ISSUE558_TRIGGER,
        "issue558_run": ISSUE558_RUN,
        "issue558_job": ISSUE558_JOB,
        "issue558_result_sha256": ISSUE558_RESULT_SHA256,
        "issue545_trigger": ISSUE545_TRIGGER,
        "issue545_run": ISSUE545_RUN,
        "issue545_job": ISSUE545_JOB,
        "issue545_failure": ISSUE545_FAILURE,
        "issue553_trigger": ISSUE553_TRIGGER,
        "issue553_run": ISSUE553_RUN,
        "issue553_job": ISSUE553_JOB,
        "issue553_result_sha256": ISSUE553_RESULT_SHA256,
        "issue529_run": ISSUE529_RUN,
        "issue529_job": ISSUE529_JOB,
        "issue529_result_sha256": ISSUE529_RESULT_SHA256,
        "checkpoint_hash_keys": sorted(CHECKPOINT_HASH_KEYS),
        "scientific_seed_consumed": False,
        "architecture_freeze_authorized": False,
        "fresh_scientific_seed_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
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
        "adapter_blob": ADAPTER_BLOB,
        "v26_8_candidate_blob": V26_8_CANDIDATE_BLOB,
    }
    print(RESULT_MARKER + json.dumps(marker, separators=(",", ":")), flush=True)
    print(SUMMARY_MARKER + json.dumps(_summary(result), separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    print(
        PRECHECK_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "source_tree": SOURCE_TREE,
                "adapter_blob": ADAPTER_BLOB,
                "result_path": RESULT_PATH,
                "max_gpu_seconds": MAX_GPU_SECONDS,
                "gpu": "L4",
                "gpu_remote_functions": 1,
                "scientific_seed_consumed": False,
                "architecture_freeze_authorized": False,
                "fresh_scientific_seed_authorized": False,
                "independent_replication_credit": False,
                "100m_authorized": False,
                "breakthrough_proven": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    check = preflight.remote()
    print(PREFLIGHT_MARKER + json.dumps(check, separators=(",", ":")), flush=True)
    result = run_gate.remote()
    print(json.dumps(_summary(result), indent=2), flush=True)
