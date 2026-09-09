from __future__ import annotations

"""#776 guarded Modal runner for the repaired event-memory capability retest.

Importing this module allocates no GPU and consumes no scientific seed. The
preflight is CPU-only. The sole L4 function is unreachable from governance
without a separately merged workflow, successful zero-GPU preauthorization,
and later explicit seed-27641 authorization.
"""

from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator

import modal

RUNNER_ISSUE = 776
RESEARCH_ISSUE = 776

SOURCE_MAIN = "0a838b5e20a867ddf886589445fb7e856644d109"
SOURCE_TREE = "88b06324ae046510f88e037667b6304c6e74c33e"
IMPLEMENTATION_BASE_MAIN = "5c53d405af9d76e20a1a3e10979fe3c88380157d"
IMPLEMENTATION_BASE_TREE = "81264d140d51cdba30a9c9f51abd9dc9dacabcb6"

ADAPTER_BLOB = "1489df753f2eee0f3acc5aba049c0eeb40fce10d"
PROTOCOL_BLOB = "0adac3d8b755e7a26e21cfb2440e9ead74eb13fc"
INTEGRATED_MODEL_BLOB = "b695b1b7b7476be3a16433c96dff32870f2e1f49"
FROZEN_BASE_BLOB = "40003b68987d026265b1d53fcb637f18277f9cac"
FROZEN_GATE_BLOB = "981602432b684989f7a5011ff953e6965368c5e5"
V18_BLOB = "97861a2407876f62665b13140c2135b4a11d4597"
DELTA_BLOB = "ec0b5d29b3d4ac27bd60fd9c152480b9f177c3e9"

PREAUTH_PREFIX = "[aera-event-memory-repair-seed27641-preauth-v1]"
L4_PREFIX = "[aera-event-memory-repair-seed27641-l4-v1]"
RESULT_PATH = "/vol/aera-capability/event-memory-repair-v1-seed27641/result.json"
CHECKPOINT_DIR = "/vol/aera-capability/event-memory-repair-v1-seed27641/checkpoints"
MODEL_SEED = 27_641

APP_NAME = "aera-issue776-event-memory-repair-runner"
VOLUME_NAME = "tam-research-data"
MAX_L4_WALL_SECONDS = 20_400
HEARTBEAT_SECONDS = 60

PREAUTH_MARKER = "AERA_ISSUE776_REPAIR_PREAUTH_JSON="
L4_START_MARKER = "AERA_ISSUE776_REPAIR_L4_START_JSON="
PROGRESS_MARKER = "AERA_ISSUE776_REPAIR_PROGRESS_JSON="
RESULT_MARKER = "AERA_ISSUE776_REPAIR_RESULT_JSON="
SUMMARY_MARKER = "AERA_ISSUE776_REPAIR_SUMMARY_JSON="

PROTOCOL_LOCAL_PATH = "docs/aera_issue776_event_memory_repair_protocol.json"
PROTOCOL_REMOTE_PATH = "/root/docs/aera_issue776_event_memory_repair_protocol.json"

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
    from tam_research import aera_issue776_event_memory_scientific_adapter as adapter
    from tam_research import aera_issue770_integrated_event_memory_cpu as integrated
    from tam_research import aera_issue748_memory_capability_seed1_harness_base as base
    from tam_research import aera_memory_capability_gate_v1 as gate
    from tam_research import aera_hardware_core_v18 as v18
    from tam_research import aera_delta_memory as delta

    got = {
        "adapter": _git_blob_sha(Path(adapter.__file__)),
        "integrated_model": _git_blob_sha(Path(integrated.__file__)),
        "frozen_base": _git_blob_sha(Path(base.__file__)),
        "frozen_gate": _git_blob_sha(Path(gate.__file__)),
        "v18": _git_blob_sha(Path(v18.__file__)),
        "delta": _git_blob_sha(Path(delta.__file__)),
        "protocol": _git_blob_sha(Path(PROTOCOL_REMOTE_PATH)),
    }
    expected = {
        "adapter": ADAPTER_BLOB,
        "integrated_model": INTEGRATED_MODEL_BLOB,
        "frozen_base": FROZEN_BASE_BLOB,
        "frozen_gate": FROZEN_GATE_BLOB,
        "v18": V18_BLOB,
        "delta": DELTA_BLOB,
        "protocol": PROTOCOL_BLOB,
    }
    if got != expected:
        raise RuntimeError(f"#776 source blob drift: got={got} expected={expected}")

    protocol = json.loads(Path(PROTOCOL_REMOTE_PATH).read_text())
    snapshot = adapter.protocol_snapshot()
    if snapshot != protocol:
        raise RuntimeError("#776 adapter/protocol snapshot drift")
    if snapshot["preauth_prefix"] != PREAUTH_PREFIX:
        raise RuntimeError("#776 preauth prefix drift")
    if snapshot["l4_prefix"] != L4_PREFIX:
        raise RuntimeError("#776 L4 prefix drift")
    if snapshot["result_path"] != RESULT_PATH:
        raise RuntimeError("#776 result namespace drift")
    if snapshot["checkpoint_dir"] != CHECKPOINT_DIR:
        raise RuntimeError("#776 checkpoint namespace drift")
    if int(snapshot["candidate_model_seed"]) != MODEL_SEED:
        raise RuntimeError("#776 seed drift")
    if any(value is not False for value in snapshot["authority"].values()):
        raise RuntimeError("#776 inherited authority drift")
    return got


def _checkpoint_absent_or_empty() -> bool:
    root = Path(CHECKPOINT_DIR)
    return (not root.exists()) or (root.is_dir() and not any(root.iterdir()))


@app.function(image=image, cpu=4, memory=8192, timeout=180, volumes={"/vol": volume})
def preflight() -> dict[str, Any]:
    """Zero-GPU/no-model/no-write audit of source identity and fresh namespace."""
    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"#776 result already exists: {RESULT_PATH}")
    if not _checkpoint_absent_or_empty():
        raise RuntimeError(f"#776 checkpoint namespace not empty: {CHECKPOINT_DIR}")
    blobs = _source_evidence()
    return {
        "runner_issue": RUNNER_ISSUE,
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "implementation_base_main": IMPLEMENTATION_BASE_MAIN,
        "implementation_base_tree": IMPLEMENTATION_BASE_TREE,
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
        "seed_37641_authorized": False,
        "systems_optimization_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_replication_authorized": False,
        "scaling_authorized": False,
        "breakthrough_proven": False,
    }


def _write_result(payload: dict[str, Any]) -> None:
    path = Path(RESULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    volume.commit()


def _progress(label: str, phase: str, started: float, **extra: Any) -> None:
    payload = {
        "label": label,
        "phase": phase,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        **extra,
    }
    print(PROGRESS_MARKER + json.dumps(payload, sort_keys=True), flush=True)


@contextmanager
def _heartbeat(label: str, **extra: Any) -> Iterator[None]:
    """Emit start/heartbeat/complete progress without altering scientific work."""
    started = time.monotonic()
    stopped = threading.Event()
    _progress(label, "start", started, **extra)

    def beat() -> None:
        while not stopped.wait(HEARTBEAT_SECONDS):
            _progress(label, "heartbeat", started, **extra)

    thread = threading.Thread(target=beat, name=f"#776-heartbeat-{label}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1.0)
        _progress(label, "complete", started, **extra)


def _incomplete_result(
    *,
    device_name: str,
    training: dict[str, dict[str, Any]],
    equal_token: dict[str, dict[str, Any]],
    inference_latency_ms: dict[str, float],
    failed_variant: str,
) -> dict[str, Any]:
    return {
        "scope": "aera_event_memory_repair_seed27641",
        "status": "INCOMPLETE_REPAIRED_GPU_TIME_BUDGET",
        "runner_issue": RUNNER_ISSUE,
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "implementation_base_main": IMPLEMENTATION_BASE_MAIN,
        "implementation_base_tree": IMPLEMENTATION_BASE_TREE,
        "model_seed": MODEL_SEED,
        "device": device_name,
        "failed_variant": failed_variant,
        "training": training,
        "equal_token_partial": equal_token,
        "inference_latency_ms_partial": inference_latency_ms,
        "result_path": RESULT_PATH,
        "checkpoint_dir": CHECKPOINT_DIR,
        "decision": {
            "recommendation": "STOP_INCOMPLETE_REPAIRED_SEED27641_NO_RETRY",
            "capability_screen_pass": False,
        },
        "authority": {
            "seed_27641_attempt_consumed": True,
            "seed_37641_authorized": False,
            "systems_optimization_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_replication_authorized": False,
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
def run_repaired_seed() -> dict[str, Any]:
    """Run exactly one future-authorized repaired seed-27641 A/B/C/D capability gate."""
    import torch
    from tam_research import aera_issue776_event_memory_scientific_adapter as adapter

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"#776 result already exists: {RESULT_PATH}")
    if not _checkpoint_absent_or_empty():
        raise RuntimeError(f"#776 checkpoints already exist: {CHECKPOINT_DIR}")
    _source_evidence()

    if not torch.cuda.is_available():
        raise RuntimeError("#776 repaired seed requires separately authorized NVIDIA L4")
    if MODEL_SEED != adapter.CANDIDATE_MODEL_SEED:
        raise RuntimeError("#776 candidate seed drift")
    if tuple(adapter.TRAINED_VARIANTS) != (
        "A_backbone",
        "B_backbone_plus_memory",
        "C_backbone_plus_routing",
        "D_combined",
    ):
        raise RuntimeError("#776 trained variant order drift")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    device_name = torch.cuda.get_device_name(device)
    cases = adapter.evaluation_cases()
    if len(cases) != 432 or len({case.sample_index for case in cases}) != 432:
        raise RuntimeError("#776 held-out evaluator drift")

    training: dict[str, dict[str, Any]] = {}
    equal_token: dict[str, dict[str, Any]] = {}
    inference_latency_ms: dict[str, float] = {}
    checkpoint_root = Path(CHECKPOINT_DIR)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for variant in adapter.TRAINED_VARIANTS:
        with _heartbeat(f"train:{variant}", variant=variant):
            model, info = adapter.train_variant(
                variant,
                MODEL_SEED,
                device=device,
                scientific_seed_authorized=True,
                checkpoint_dir=CHECKPOINT_DIR,
            )
        training[variant] = info
        print(
            PROGRESS_MARKER
            + json.dumps(
                {
                    "phase": "variant_training_summary",
                    "variant": variant,
                    "steps": info.get("steps"),
                    "tokens": info.get("tokens"),
                    "gpu_seconds": info.get("cumulative_gpu_seconds"),
                    "complete_frozen_token_budget": info.get("complete_frozen_token_budget"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        with _heartbeat(f"eval_equal_token:{variant}", variant=variant):
            equal_token[variant] = adapter.evaluate_model(model, cases, device=device)
        with _heartbeat(f"latency:{variant}", variant=variant):
            inference_latency_ms[variant] = adapter.measure_model_inference_latency_ms(
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
                "seed_37641_authorized": False,
                "systems_optimization_authorized": False,
                "architecture_freeze_authorized": False,
                "s2_replication_authorized": False,
                "scaling_authorized": False,
                "breakthrough_proven": False,
            }
            print(RESULT_MARKER + json.dumps(summary, sort_keys=True), flush=True)
            return summary

    checkpoint_selection = adapter.select_equal_gpu_time_checkpoints(
        {
            variant: training[variant]["checkpoint_records"]
            for variant in adapter.TRAINED_VARIANTS
        }
    )
    equal_gpu_time: dict[str, dict[str, Any]] = {}
    for variant in adapter.TRAINED_VARIANTS:
        selected = checkpoint_selection[variant]
        path = selected.get("path")
        if not isinstance(path, str) or not path:
            raise RuntimeError(f"#776 missing durable equal-GPU checkpoint for {variant}")
        with _heartbeat(f"eval_equal_gpu:{variant}", variant=variant):
            model = adapter.load_model_checkpoint(
                variant,
                MODEL_SEED,
                path,
                device=device,
                scientific_seed_authorized=True,
            )
            equal_gpu_time[variant] = adapter.evaluate_model(model, cases, device=device)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with _heartbeat("simple_retrieval"):
        simple_retrieval = adapter.evaluate_simple_retrieval(cases)
        inference_latency_ms["E_simple_retrieval"] = (
            adapter.measure_simple_retrieval_latency_ms()
        )

    frozen_decision = adapter.seed_decision(
        equal_token=equal_token,
        equal_gpu_time=equal_gpu_time,
        training=training,
        inference_latency_ms=inference_latency_ms,
        simple_retrieval=simple_retrieval,
    )
    decision = adapter.normalize_repaired_decision(frozen_decision)

    result = {
        "scope": "aera_event_memory_repair_seed27641",
        "status": "COMPLETE_EVENT_MEMORY_REPAIR_SEED27641",
        "runner_issue": RUNNER_ISSUE,
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_tree": SOURCE_TREE,
        "implementation_base_main": IMPLEMENTATION_BASE_MAIN,
        "implementation_base_tree": IMPLEMENTATION_BASE_TREE,
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
            "seed_27641_attempt_consumed": True,
            "gpu_authorized_for_this_run": True,
            "scientific_training_authorized_for_this_run": True,
            "scientific_evaluation_authorized_for_this_run": True,
            "scientific_seed_27641_authorized_for_this_run": True,
            "seed_37641_authorized": False,
            "systems_optimization_authorized": False,
            "architecture_freeze_authorized": False,
            "s2_replication_authorized": False,
            "scaling_authorized": False,
            "breakthrough_proven": False,
        },
    }
    adapter.validate_repaired_result_schema(result)
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
        "seed_37641_authorized": False,
        "systems_optimization_authorized": False,
        "architecture_freeze_authorized": False,
        "s2_replication_authorized": False,
        "scaling_authorized": False,
        "breakthrough_proven": False,
    }
    print(RESULT_MARKER + json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def preauth_main() -> None:
    evidence = preflight.remote()
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True), flush=True)


@app.local_entrypoint()
def l4_main() -> None:
    pre = preflight.remote()
    print(PREAUTH_MARKER + json.dumps(pre, sort_keys=True), flush=True)
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
                "single_seed27641_attempt": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    summary = run_repaired_seed.remote()
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True), flush=True)
