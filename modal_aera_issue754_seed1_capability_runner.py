from __future__ import annotations

"""Issue #754 infrastructure runner for the frozen #748/#750 seed1 capability gate.

Importing this module allocates no GPU and consumes no scientific seed. The only
GPU function is separately gated by the issue-trigger workflow after a successful
zero-GPU preauthorization and a distinct exact-main seed1 authorization.
"""

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import modal

RUNNER_ISSUE = 754
RESEARCH_ISSUE = 748
SOURCE_MAIN = "bb4c01c71deed63c94b89d91c9a713fa2a0f3b34"
SOURCE_TREE = "3caa58461f9d0513c8564210b2e8179897abbfc4"

HARNESS_WRAPPER_BLOB = "afc939a69633f68ded05eb585c95a599a8f5c981"
HARNESS_BASE_BLOB = "40003b68987d026265b1d53fcb637f18277f9cac"
PROTOCOL_BLOB = "8bd8d646a72c0d5a2b1cbe7bedddc25bff6b43bf"
CPU_TEST_BLOB = "5ad56721532353a1b9d71fac4a470b4b1e1fa704"

PREAUTH_PREFIX = "[aera-issue748-memory-capability-seed1-preauth]"
L4_PREFIX = "[aera-issue748-memory-capability-seed1-l4]"
RESULT_PATH = "/vol/aera-capability/issue748-memory-capability-seed1/result.json"
CHECKPOINT_DIR = "/vol/aera-capability/issue748-memory-capability-seed1/checkpoints"
MODEL_SEED = 17641

APP_NAME = "aera-issue754-seed1-capability-runner"
VOLUME_NAME = "tam-research-data"
MAX_L4_WALL_SECONDS = 20_400

PREAUTH_MARKER = "AERA_ISSUE754_SEED1_PREAUTH_JSON="
L4_START_MARKER = "AERA_ISSUE754_SEED1_L4_START_JSON="
RESULT_MARKER = "AERA_ISSUE754_SEED1_RESULT_JSON="
SUMMARY_MARKER = "AERA_ISSUE754_SEED1_SUMMARY_JSON="

PROTOCOL_LOCAL_PATH = "docs/aera_issue748_seed1_harness_protocol.json"
PROTOCOL_REMOTE_PATH = "/root/docs/aera_issue748_seed1_harness_protocol.json"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.10,<2.11", "numpy>=2.0,<3")
    .add_local_python_source("tam_research")
    .add_local_file(PROTOCOL_LOCAL_PATH, PROTOCOL_REMOTE_PATH)
)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _source_evidence() -> dict[str, str]:
    from tam_research import aera_issue748_memory_capability_seed1_harness as harness
    from tam_research import aera_issue748_memory_capability_seed1_harness_base as base

    got = {
        "harness_wrapper": _git_blob_sha(Path(harness.__file__)),
        "harness_base": _git_blob_sha(Path(base.__file__)),
        "protocol": _git_blob_sha(Path(PROTOCOL_REMOTE_PATH)),
    }
    expected = {
        "harness_wrapper": HARNESS_WRAPPER_BLOB,
        "harness_base": HARNESS_BASE_BLOB,
        "protocol": PROTOCOL_BLOB,
    }
    if got != expected:
        raise RuntimeError(f"issue754 frozen source blob drift: got={got} expected={expected}")

    protocol = json.loads(Path(PROTOCOL_REMOTE_PATH).read_text())
    snapshot = harness.harness_protocol_snapshot()
    if snapshot != protocol:
        raise RuntimeError("issue754 harness/protocol snapshot drift")
    future = snapshot.get("future_execution", {})
    if future.get("preauth_prefix") != PREAUTH_PREFIX:
        raise RuntimeError("issue754 preauth prefix drift")
    if future.get("gpu_prefix") != L4_PREFIX:
        raise RuntimeError("issue754 L4 prefix drift")
    if future.get("result_path") != RESULT_PATH:
        raise RuntimeError("issue754 result path drift")
    if future.get("checkpoint_dir") != CHECKPOINT_DIR:
        raise RuntimeError("issue754 checkpoint directory drift")
    if int(future.get("seed1_model_seed", -1)) != MODEL_SEED:
        raise RuntimeError("issue754 seed1 model seed drift")
    if any(value is not False for value in snapshot.get("authority", {}).values()):
        raise RuntimeError("issue754 inherited harness authority drift")
    return got


def _checkpoint_absent_or_empty() -> bool:
    root = Path(CHECKPOINT_DIR)
    return (not root.exists()) or (root.is_dir() and not any(root.iterdir()))


@app.function(image=image, cpu=4, memory=8192, timeout=180, volumes={"/vol": volume})
def preflight() -> dict[str, Any]:
    """Zero-GPU, no-model, no-write audit of exact source and fresh result namespace."""
    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue754 seed1 result already exists: {RESULT_PATH}")
    if not _checkpoint_absent_or_empty():
        raise RuntimeError(f"issue754 seed1 checkpoint namespace is not empty: {CHECKPOINT_DIR}")
    blobs = _source_evidence()
    return {
        "runner_issue": RUNNER_ISSUE,
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "frozen_blobs": blobs,
        "preauth_prefix": PREAUTH_PREFIX,
        "l4_prefix": L4_PREFIX,
        "result_path": RESULT_PATH,
        "checkpoint_dir": CHECKPOINT_DIR,
        "model_seed": MODEL_SEED,
        "result_absent": True,
        "checkpoint_absent_or_empty": True,
        "gpu_used": False,
        "model_constructed": False,
        "scientific_seed_consumed": False,
        "training_performed": False,
        "evaluation_performed": False,
        "checkpoint_written": False,
        "result_written": False,
        "seeds_2_3_authorized": False,
        "systems_optimization_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_authorized": False,
        "replication_authorized": False,
        "scaling_authorized": False,
        "breakthrough_proven": False,
    }


def _write_result(payload: dict[str, Any]) -> None:
    path = Path(RESULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    volume.commit()


def _incomplete_result(
    *,
    device_name: str,
    training: dict[str, dict[str, Any]],
    equal_token: dict[str, dict[str, Any]],
    inference_latency_ms: dict[str, float],
    failed_variant: str,
) -> dict[str, Any]:
    return {
        "scope": "aera_issue754_seed1_capability",
        "status": "INCOMPLETE_FROZEN_GPU_TIME_BUDGET",
        "runner_issue": RUNNER_ISSUE,
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "model_seed": MODEL_SEED,
        "device": device_name,
        "failed_variant": failed_variant,
        "training": training,
        "equal_token_partial": equal_token,
        "inference_latency_ms_partial": inference_latency_ms,
        "result_path": RESULT_PATH,
        "checkpoint_dir": CHECKPOINT_DIR,
        "decision": {
            "recommendation": "STOP_INCOMPLETE_SEED1_NO_RETRY",
            "capability_screen_pass": False,
        },
        "authority": {
            "seed1_attempt_consumed": True,
            "seeds_2_3_authorized": False,
            "systems_optimization_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_authorized": False,
            "replication_authorized": False,
            "scaling_authorized": False,
            "breakthrough_proven": False,
        },
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_L4_WALL_SECONDS,
    volumes={"/vol": volume},
)
def run_seed1() -> dict[str, Any]:
    """Execute exactly the frozen seed1 A/B/C/D capability gate on one L4 allocation."""
    import torch
    from tam_research import aera_issue748_memory_capability_seed1_harness as harness

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue754 seed1 result already exists: {RESULT_PATH}")
    if not _checkpoint_absent_or_empty():
        raise RuntimeError(f"issue754 seed1 checkpoints already exist: {CHECKPOINT_DIR}")
    _source_evidence()

    if not torch.cuda.is_available():
        raise RuntimeError("issue754 seed1 requires the separately authorized NVIDIA L4")
    if MODEL_SEED != harness.SEED1_MODEL_SEED:
        raise RuntimeError("issue754 seed1 model seed drift")
    if tuple(harness.TRAINED_VARIANTS) != (
        "A_backbone",
        "B_backbone_plus_memory",
        "C_backbone_plus_routing",
        "D_combined",
    ):
        raise RuntimeError("issue754 trained variant order drift")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    device_name = torch.cuda.get_device_name(device)
    cases = harness.evaluation_cases()
    if len(cases) != 432 or len({case.sample_index for case in cases}) != 432:
        raise RuntimeError("issue754 held-out evaluator drift")

    training: dict[str, dict[str, Any]] = {}
    equal_token: dict[str, dict[str, Any]] = {}
    inference_latency_ms: dict[str, float] = {}

    checkpoint_root = Path(CHECKPOINT_DIR)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for variant in harness.TRAINED_VARIANTS:
        model, info = harness.train_variant(
            variant,
            MODEL_SEED,
            device=device,
            scientific_seed_authorized=True,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        training[variant] = info
        equal_token[variant] = harness.evaluate_model(model, cases, device=device)
        inference_latency_ms[variant] = harness.measure_model_inference_latency_ms(
            model, device=device
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        volume.commit()

        if not bool(info.get("complete_frozen_token_budget")):
            result = _incomplete_result(
                device_name=device_name,
                training=training,
                equal_token=equal_token,
                inference_latency_ms=inference_latency_ms,
                failed_variant=variant,
            )
            _write_result(result)
            summary = {
                "status": result["status"],
                "research_issue": RESEARCH_ISSUE,
                "runner_issue": RUNNER_ISSUE,
                "model_seed": MODEL_SEED,
                "device": device_name,
                "failed_variant": variant,
                "recommendation": result["decision"]["recommendation"],
                "seeds_2_3_authorized": False,
                "systems_optimization_authorized": False,
                "architecture_freeze_authorized": False,
                "scaling_authorized": False,
                "breakthrough_proven": False,
            }
            print(RESULT_MARKER + json.dumps(summary, sort_keys=True))
            return summary

    checkpoint_selection = harness.select_equal_gpu_time_checkpoints(
        {
            variant: training[variant]["checkpoint_records"]
            for variant in harness.TRAINED_VARIANTS
        }
    )
    equal_gpu_time: dict[str, dict[str, Any]] = {}
    for variant in harness.TRAINED_VARIANTS:
        selected = checkpoint_selection[variant]
        path = selected.get("path")
        if not isinstance(path, str) or not path:
            raise RuntimeError(f"issue754 missing durable equal-GPU checkpoint for {variant}")
        model = harness.load_model_checkpoint(
            variant,
            MODEL_SEED,
            path,
            device=device,
            scientific_seed_authorized=True,
        )
        equal_gpu_time[variant] = harness.evaluate_model(model, cases, device=device)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    simple_retrieval = harness.evaluate_simple_retrieval(cases)
    inference_latency_ms["E_simple_retrieval"] = harness.measure_simple_retrieval_latency_ms()
    decision = harness.seed1_decision(
        equal_token=equal_token,
        equal_gpu_time=equal_gpu_time,
        training=training,
        inference_latency_ms=inference_latency_ms,
        simple_retrieval=simple_retrieval,
    )

    result = {
        "scope": "aera_issue754_seed1_capability",
        "status": "COMPLETE_FROZEN_SEED1",
        "runner_issue": RUNNER_ISSUE,
        "research_issue": RESEARCH_ISSUE,
        "source_main": harness.SOURCE_MAIN,
        "source_tree": harness.SOURCE_TREE,
        "runner_source_main": SOURCE_MAIN,
        "runner_source_tree": SOURCE_TREE,
        "model_seed": MODEL_SEED,
        "device": device_name,
        "equal_token": equal_token,
        "equal_gpu_time": equal_gpu_time,
        "training": training,
        "inference_latency_ms": inference_latency_ms,
        "simple_retrieval": simple_retrieval,
        "equal_gpu_time_checkpoint_selection": checkpoint_selection,
        "decision": decision,
        "authority": {
            "seed1_attempt_consumed": True,
            "gpu_authorized_for_seed1_run": True,
            "scientific_training_authorized_for_seed1_run": True,
            "fresh_scientific_seed_authorized_for_seed1_run": True,
            "seeds_2_3_authorized": False,
            "systems_optimization_authorized": False,
            "architecture_freeze_authorized": False,
            "scaling_authorized": False,
            "breakthrough_proven": False,
        },
    }
    harness.validate_seed1_result_schema(result)
    _write_result(result)

    summary = {
        "status": result["status"],
        "runner_issue": RUNNER_ISSUE,
        "research_issue": RESEARCH_ISSUE,
        "model_seed": MODEL_SEED,
        "device": device_name,
        "decision": decision,
        "equal_token_B_minus_A_long_accuracy": decision["checks"]["equal_token"][
            "B_minus_A_long_accuracy"
        ],
        "equal_gpu_time_B_minus_A_long_accuracy": decision["checks"]["equal_gpu_time"][
            "B_minus_A_long_accuracy"
        ],
        "seeds_2_3_authorized": False,
        "systems_optimization_authorized": False,
        "architecture_freeze_authorized": False,
        "scaling_authorized": False,
        "breakthrough_proven": False,
    }
    print(RESULT_MARKER + json.dumps(summary, sort_keys=True))
    return summary


@app.local_entrypoint()
def preauth_main() -> None:
    evidence = preflight.remote()
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True))


@app.local_entrypoint()
def l4_main() -> None:
    pre = preflight.remote()
    print(PREAUTH_MARKER + json.dumps(pre, sort_keys=True))
    print(
        L4_START_MARKER
        + json.dumps(
            {
                "runner_issue": RUNNER_ISSUE,
                "research_issue": RESEARCH_ISSUE,
                "gpu": "L4",
                "model_seed": MODEL_SEED,
                "result_path": RESULT_PATH,
                "checkpoint_dir": CHECKPOINT_DIR,
                "single_seed1_attempt": True,
            },
            sort_keys=True,
        )
    )
    summary = run_seed1.remote()
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
