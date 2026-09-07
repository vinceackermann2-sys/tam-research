from __future__ import annotations

"""Issue #716 orchestration-only authorization-heading guard repair for frozen #710."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import modal
import modal_aera_v26_10_issue710_memory_safe_harness as frozen710

APP_NAME = "aera-v26-10-issue716-l4-auth-guard-repair1"
VOLUME_NAME = frozen710.VOLUME_NAME
RESULT_PATH = "/vol/aera-v26/issue716-v26-10-l4-auth-guard-repair1/result.json"
SOURCE_MAIN = "8371d9ab3988df33c7b92b529016618ebf60f9ee"
SOURCE_TREE = "fd171423b5878c9082e689d7d83da9f8c7c81569"
RESEARCH_ISSUE = 716
PARENT_HARNESS_ISSUE = 710
CONSUMED_TRIGGER = 715
CONSUMED_FAILURE_COMMENT = 5573729021
CONSUMED_RUN = 34145700849
CONSUMED_JOB = 101817171862
CONSUMED_ATTEMPT = 1
PREVIOUS_AUTH_COMMENT = 5573681590
AUTHORIZATION_HEADING = "## #716 sole L4 authorization-heading guard repair1 authorization"
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
PREAUTH_MARKER = "AERA_V26_10_ISSUE716_PREAUTH_JSON="
L4_START_MARKER = "AERA_V26_10_ISSUE716_L4_START_JSON="
RESULT_MARKER = "AERA_V26_10_ISSUE716_RESULT_JSON="
SUMMARY_MARKER = "AERA_V26_10_ISSUE716_SUMMARY_JSON="

# frozen710.image already contains all transitive scientific/runtime dependencies.
# Add the exact frozen root-level #710 launcher so remote imports cannot regress.
image = frozen710.image.add_local_file(
    ISSUE710_LAUNCHER, f"/root/{ISSUE710_LAUNCHER}"
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _frozen_blobs() -> dict[str, str]:
    got = dict(frozen710._frozen_blobs())
    got["issue710_launcher"] = _blob(Path(f"/root/{ISSUE710_LAUNCHER}"))
    if got.get("issue710_launcher") != ISSUE710_LAUNCHER_BLOB:
        raise RuntimeError(f"issue716 frozen #710 launcher drift: {got.get('issue710_launcher')}")
    if got.get("v26_10_impl") != V26_10_IMPL_BLOB:
        raise RuntimeError(f"issue716 frozen v26.10 implementation drift: {got.get('v26_10_impl')}")
    if frozen710.LOGIT_COMPARE_BATCH_CHUNK != 1:
        raise RuntimeError("issue716 inherited memory-safe comparator drift")
    if tuple(frozen710.BATCHES) != (8, 64):
        raise RuntimeError("issue716 inherited batch drift")
    if frozen710.WARMUP_CALLS != 3 or frozen710.TIMED_CALLS_PER_CONDITION != 20:
        raise RuntimeError("issue716 inherited timing fixture drift")
    if frozen710.PROFILE_CALLS_PER_CONDITION != 1:
        raise RuntimeError("issue716 inherited profiler fixture drift")
    if frozen710.MIN_LATENCY_IMPROVEMENT != 0.05:
        raise RuntimeError("issue716 inherited latency threshold drift")
    if frozen710.MIN_STREAM_SYNCHRONIZE_REDUCTION != 20:
        raise RuntimeError("issue716 inherited synchronization threshold drift")
    if frozen710.INTEGRATED_ATOL != 1e-2 or frozen710.INTEGRATED_RTOL != 1e-2:
        raise RuntimeError("issue716 inherited integrated tolerance drift")
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
    """Patch only successor identity/result globals; restore them unconditionally."""
    previous = {name: getattr(frozen710, name) for name in _PATCH_FIELDS}
    try:
        for name, value in _PATCH_FIELDS.items():
            setattr(frozen710, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(frozen710, name, value)


def _successor_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["research_issue"] = RESEARCH_ISSUE
    out["source_main"] = SOURCE_MAIN
    out["source_tree"] = SOURCE_TREE
    out["result_path"] = RESULT_PATH
    out["parent_harness_issue"] = PARENT_HARNESS_ISSUE
    out["consumed_trigger"] = CONSUMED_TRIGGER
    out["consumed_failure_comment"] = CONSUMED_FAILURE_COMMENT
    out["consumed_run"] = CONSUMED_RUN
    out["consumed_job"] = CONSUMED_JOB
    out["consumed_attempt"] = CONSUMED_ATTEMPT
    out["previous_authorization_comment"] = PREVIOUS_AUTH_COMMENT
    out["authorization_heading"] = AUTHORIZATION_HEADING
    out["orchestration_guard_repair_only"] = True
    out["inherited_memory_safe_launcher_blob"] = ISSUE710_LAUNCHER_BLOB
    out["v26_10_impl_blob"] = V26_10_IMPL_BLOB
    return out


@app.function(image=image, cpu=4, memory=8192, timeout=180, volumes={"/vol": volume})
def preflight() -> dict[str, Any]:
    import tam_research.aera_v26_5_end_to_end_systems as base

    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue716 result already exists: {RESULT_PATH}")
    _frozen_blobs()
    hashes = base.checkpoint_hashes(base.CHECKPOINT_RELATIVE_DIR)
    if hashes != CHECKPOINT_HASHES:
        raise RuntimeError("issue716 checkpoint drift")
    with _successor_identity():
        inherited = frozen710.preflight.local()
    payload = _successor_metadata(inherited)
    if payload.get("result_absent") is not True:
        raise RuntimeError("issue716 inherited preflight did not prove fresh result absence")
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
            raise RuntimeError(f"issue716 forbidden preflight authority drift: {key}")
    return payload


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=MAX_GPU_SECONDS,
    volumes={"/vol": volume},
)
def run_microbenchmark() -> dict[str, Any]:
    volume.reload()
    if Path(RESULT_PATH).exists():
        raise RuntimeError(f"issue716 result already exists: {RESULT_PATH}")
    _frozen_blobs()
    with _successor_identity():
        summary = frozen710.run_microbenchmark.local()
    if not Path(RESULT_PATH).exists():
        raise RuntimeError("issue716 inherited benchmark returned without fresh durable result")
    result = json.loads(Path(RESULT_PATH).read_text())
    if result.get("research_issue") != RESEARCH_ISSUE:
        raise RuntimeError(f"issue716 durable result identity drift: {result.get('research_issue')}")
    if result.get("source_main") != SOURCE_MAIN or result.get("source_tree") != SOURCE_TREE:
        raise RuntimeError("issue716 durable source identity drift")
    if result.get("logit_compare_batch_chunk") != LOGIT_COMPARE_BATCH_CHUNK:
        raise RuntimeError("issue716 durable comparator drift")
    if set(result.get("rows", {})) != {"8", "64"}:
        raise RuntimeError("issue716 durable batch rows drift")
    return _successor_metadata(summary)


@app.local_entrypoint()
def preauth_main() -> None:
    evidence = preflight.remote()
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True))


@app.local_entrypoint()
def l4_main() -> None:
    evidence = preflight.remote()
    print(PREAUTH_MARKER + json.dumps(evidence, sort_keys=True))
    print(
        L4_START_MARKER
        + json.dumps(
            {
                "research_issue": RESEARCH_ISSUE,
                "gpu": "L4",
                "max_gpu_seconds": MAX_GPU_SECONDS,
                "result_path": RESULT_PATH,
                "authorization_heading": AUTHORIZATION_HEADING,
                "orchestration_guard_repair_only": True,
            },
            sort_keys=True,
        )
    )
    summary = run_microbenchmark.remote()
    print(SUMMARY_MARKER + json.dumps(summary, sort_keys=True))
