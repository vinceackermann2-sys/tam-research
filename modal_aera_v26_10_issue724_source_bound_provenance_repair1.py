from __future__ import annotations

"""Issue #724 orchestration-only source-vs-bound provenance repair for frozen #710."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import modal
import modal_aera_v26_10_issue710_memory_safe_harness as frozen710

APP_NAME = "aera-v26-10-issue724-source-bound-provenance-repair1"
VOLUME_NAME = frozen710.VOLUME_NAME
RESULT_PATH = "/vol/aera-v26/issue724-v26-10-source-bound-provenance-repair1/result.json"
SOURCE_MAIN = "8371d9ab3988df33c7b92b529016618ebf60f9ee"
SOURCE_TREE = "fd171423b5878c9082e689d7d83da9f8c7c81569"
ORCHESTRATION_BASE_MAIN = "67344b9535dd305f1478ea6ea692fe45a64c86d8"
ORCHESTRATION_BASE_TREE = "a208565bdad203e2cfb2354f53c07d022b977d85"
RESEARCH_ISSUE = 724
PARENT_HARNESS_ISSUE = 710
PARENT_ORCHESTRATION_ISSUE = 716
CONSUMED_TRIGGER = 723
CONSUMED_FAILURE_COMMENT = 5574302025
CONSUMED_RUN = 34150467379
CONSUMED_JOB = 101831501185
CONSUMED_ATTEMPT = 1
AUTHORIZATION_HEADING = "## #724 sole L4 source-bound provenance repair1 authorization"
ISSUE710_LAUNCHER = "modal_aera_v26_10_issue710_memory_safe_harness.py"
ISSUE710_LAUNCHER_BLOB = "b885250753ea169cd89dd7a978bb3647fd261fe8"
V26_10_IMPL_BLOB = "d8f691c198eed1fa96bcbb78a4e76cad82d18779"
CHECKPOINT_HASHES = dict(frozen710.CHECKPOINT_HASHES)
LOGIT_COMPARE_BATCH_CHUNK = frozen710.LOGIT_COMPARE_BATCH_CHUNK
BATCHES = tuple(frozen710.BATCHES)
WARMUP_CALLS = frozen710.WARMUP_CALLS
TIMED_CALLS_PER_CONDITION = frozen710.TIMED_CALLS_PER_CONDITION
PROFILE_CALLS_PER_CONDITION = frozen710.PROFILE_CALLS_PER_CONDITION
MIN_LATENCY_IMPROVEMENT = frozen710.MIN_LATENCY_IMPROVEMENT
MIN_STREAM_SYNCHRONIZE_REDUCTION = frozen710.MIN_STREAM_SYNCHRONIZE_REDUCTION
INTEGRATED_ATOL = frozen710.INTEGRATED_ATOL
INTEGRATED_RTOL = frozen710.INTEGRATED_RTOL
MAX_GPU_SECONDS = frozen710.MAX_GPU_SECONDS
PREAUTH_MARKER = "AERA_V26_10_ISSUE724_PREAUTH_JSON="
L4_START_MARKER = "AERA_V26_10_ISSUE724_L4_START_JSON="
RESULT_MARKER = "AERA_V26_10_ISSUE724_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_10_ISSUE724_SUMMARY_JSON="
BOUND_MAIN_ENV = "AERA_ISSUE724_BOUND_MAIN"

image = frozen710.image.add_local_file(
    ISSUE710_LAUNCHER, f"/root/{ISSUE710_LAUNCHER}"
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _validate_bound_main(value: str) -> str:
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError(f"issue724 invalid bound main: {value!r}")
    return value


def _frozen_blobs() -> dict[str, str]:
    got = dict(frozen710._frozen_blobs())
    got["issue710_launcher"] = _blob(Path(f"/root/{ISSUE710_LAUNCHER}"))
    if got.get("issue710_launcher") != ISSUE710_LAUNCHER_BLOB:
        raise RuntimeError(f"issue724 frozen #710 launcher drift: {got.get('issue710_launcher')}")
    if got.get("v26_10_impl") != V26_10_IMPL_BLOB:
        raise RuntimeError(f"issue724 frozen v26.10 implementation drift: {got.get('v26_10_impl')}")
    if frozen710.LOGIT_COMPARE_BATCH_CHUNK != 1:
        raise RuntimeError("issue724 inherited memory-safe comparator drift")
    if tuple(frozen710.BATCHES) != (8, 64):
        raise RuntimeError("issue724 inherited batch drift")
    if frozen710.WARMUP_CALLS != 3 or frozen710.TIMED_CALLS_PER_CONDITION != 20:
        raise RuntimeError("issue724 inherited timing fixture drift")
    if frozen710.PROFILE_CALLS_PER_CONDITION != 1:
        raise RuntimeError("issue724 inherited profiler fixture drift")
    if frozen710.MIN_LATENCY_IMPROVEMENT != 0.05:
        raise RuntimeError("issue724 inherited latency threshold drift")
    if frozen710.MIN_STREAM_SYNCHRONIZE_REDUCTION != 20:
        raise RuntimeError("issue724 inherited synchronization threshold drift")
    if frozen710.INTEGRATED_ATOL != 1e-2 or frozen710.INTEGRATED_RTOL != 1e-2:
        raise RuntimeError("issue724 inherited integrated tolerance drift")
    return got


_PATCH_FIELDS = {
    "RESULT_PATH": RESULT_PATH,
    "SOURCE_MAIN": SOURCE_MAIN,
    "SOURCE_TREE": SOURCE_TREE,
    "RESEARCH_ISSUE": RESEARCH_ISSUE,
    "CONSUMED_TRIGGER": CONSUMED_TRIGGER,
    "CONSUMED_FAILURE_COMMENT": CONSUMED_FAILURE_COMMENT,
    "CONSUMED_RUN": CONSUMED_RUN,
    "CONSUMED_JOB": CONSUMED_JOB,
    "CONSUMED_ATTEMPT": CONSUMED_ATTEMPT,
    "PREAUTH_MARKER": PREAUTH_MARKER,
    "L4_START_MARKER": L4_START_MARKER,
    "RESULT_MARKER": RESULT_MARKER,
    "SUMMARY_MARKER": SUMMARY_MARKER,
}


@contextmanager
def _successor_identity() -> Iterator[None]:
    previous = {name: getattr(frozen710, name) for name in _PATCH_FIELDS}
    try:
        for name, value in _PATCH_FIELDS.items():
            setattr(frozen710, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(frozen710, name, value)


def _successor_metadata(payload: dict[str, Any], bound_main: str) -> dict[str, Any]:
    bound_main = _validate_bound_main(bound_main)
    out = dict(payload)
    out["research_issue"] = RESEARCH_ISSUE
    out["source_main"] = SOURCE_MAIN
    out["source_tree"] = SOURCE_TREE
    out["bound_main"] = bound_main
    out["orchestration_base_main"] = ORCHESTRATION_BASE_MAIN
    out["orchestration_base_tree"] = ORCHESTRATION_BASE_TREE
    out["result_path"] = RESULT_PATH
    out["parent_harness_issue"] = PARENT_HARNESS_ISSUE
    out["parent_orchestration_issue"] = PARENT_ORCHESTRATION_ISSUE
    out["consumed_trigger"] = CONSUMED_TRIGGER
    out["consumed_failure_comment"] = CONSUMED_FAILURE_COMMENT
    out["consumed_run"] = CONSUMED_RUN
    out["consumed_job"] = CONSUMED_JOB
    out["consumed_attempt"] = CONSUMED_ATTEMPT
    out["authorization_heading"] = AUTHORIZATION_HEADING
    out["provenance_reporting_repair_only"] = True
    out["inherited_memory_safe_launcher_blob"] = ISSUE710_LAUNCHER_BLOB
    out["v26_10_impl_blob"] = V26_10_IMPL_BLOB
    return out


@app.function(image=image, cpu=4, memory=8192, timeout=180, volumes={"/vol": volume})
def preflight(bound_main: str) -> dict[str, Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base

    bound_main = _validate_bound_main(bound_main)
    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue724 result already exists: {RESULT_PATH}")
    _frozen_blobs()
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError("issue724 checkpoint drift")
    with _successor_identity():
        inherited = frozen710.preflight.local()
    payload = _successor_metadata(inherited, bound_main)
    if payload.get("source_main") != SOURCE_MAIN or payload.get("source_tree") != SOURCE_TREE:
        raise RuntimeError("issue724 frozen source lineage drift")
    if payload.get("bound_main") != bound_main:
        raise RuntimeError("issue724 bound execution provenance drift")
    if payload.get("result_absent") is not True:
        raise RuntimeError("issue724 inherited preflight did not prove fresh result absence")
    for key in (
        "gpu_used",
        "model_constructed",
        "new_measurement_performed",
        "optimization_microbenchmark_authorized",
        "full_e2e_systems_gate_authorized",
        "systems_pass_earned",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        if payload.get(key) is not False:
            raise RuntimeError(f"issue724 forbidden preflight authority drift: {key}")
    return payload


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_microbenchmark(bound_main: str) -> dict[str, Any]:
    bound_main = _validate_bound_main(bound_main)
    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue724 result already exists: {RESULT_PATH}")
    _frozen_blobs()
    with _successor_identity():
        summary = frozen710.run_microbenchmark.local()
    if not Path(RESULT_PATH).exists():
        raise RuntimeError("issue724 inherited benchmark returned without fresh durable result")
    result = json.loads(Path(RESULT_PATH).read_text())
    if result.get("research_issue") != RESEARCH_ISSUE:
        raise RuntimeError(f"issue724 durable result identity drift: {result.get('research_issue')}")
    if result.get("source_main") != SOURCE_MAIN or result.get("source_tree") != SOURCE_TREE:
        raise RuntimeError("issue724 durable source identity drift")
    if result.get("logit_compare_batch_chunk") != LOGIT_COMPARE_BATCH_CHUNK:
        raise RuntimeError("issue724 durable comparator drift")
    if set(result.get("rows", {})) != {"8", "64"}:
        raise RuntimeError("issue724 durable batch rows drift")
    enriched = _successor_metadata(result, bound_main)
    Path(RESULT_PATH).write_text(json.dumps(enriched, sort_keys=True, indent=2) + "\n")
    volume.commit()
    return _successor_metadata(summary, bound_main)


def _bound_main_from_env() -> str:
    value = os.environ.get(BOUND_MAIN_ENV, "")
    return _validate_bound_main(value)


@app.local_entrypoint()
def preauth_main() -> None:
    bound_main = _bound_main_from_env()
    evidence = preflight.remote(bound_main)
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True))


@app.local_entrypoint()
def l4_main() -> None:
    bound_main = _bound_main_from_env()
    evidence = preflight.remote(bound_main)
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True))
    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "source_main": SOURCE_MAIN,
                "bound_main": bound_main,
                "gpu": "L4",
                "max_gpu_seconds": MAX_GPU_SECONDS,
                "result_path": RESULT_PATH,
                "authorization_heading": AUTHORIZATION_HEADING,
                "provenance_reporting_repair_only": True,
            },
            sort_keys=True,
        )
    )
    summary = run_microbenchmark.remote(bound_main)
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
