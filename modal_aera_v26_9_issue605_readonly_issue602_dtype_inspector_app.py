from __future__ import annotations

"""CPU-only Modal reader for the immutable #602 result."""

import hashlib
import json
from pathlib import Path

import modal

from tam_research.aera_v26_9_issue605_readonly_issue602_dtype_inspector import (
    ISSUE602_RESULT_PATH,
    ISSUE602_RESULT_SHA256,
    cpu_contract_preflight_issue605,
    inspect_issue602_result,
)

APP_NAME = "tam-research-aera-v26-9-issue605-readonly-issue602-dtype-inspector"
VOLUME_NAME = "tam-research-data"
MARKER = "AERA_V26_9_ISSUE605_READONLY_ISSUE602_DTYPE_INSPECTOR_JSON="

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_python_source("tam_research")
)


def precheck() -> dict:
    check = cpu_contract_preflight_issue605()
    return {
        "research_issue": 605,
        "source_result_path": ISSUE602_RESULT_PATH,
        "source_result_sha256": ISSUE602_RESULT_SHA256,
        "cpu_only": True,
        "gpu": False,
        "model_execution": False,
        "checkpoint_execution": False,
        "corpus_access": False,
        "volume_mutation": False,
        "volume_commit": False,
        "repair_authorized": False,
        "end_to_end_systems_authorized": False,
        "scientific_seed_consumed": check["scientific_seed_consumed"],
    }


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=60,
    volumes={"/vol": volume},
)
def inspect() -> dict:
    volume.reload()
    path = Path(ISSUE602_RESULT_PATH)
    if not path.exists():
        raise RuntimeError(f"issue605 source result missing: {ISSUE602_RESULT_PATH}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ISSUE602_RESULT_SHA256:
        raise RuntimeError(
            f"issue605 source SHA drift: got={digest} expected={ISSUE602_RESULT_SHA256}"
        )
    result = json.loads(raw)
    extracted = inspect_issue602_result(result)
    extracted["verified_source_sha256"] = digest
    return extracted


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(precheck(), sort_keys=True), flush=True)
    result = inspect.remote()
    print(MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
