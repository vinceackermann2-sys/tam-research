from __future__ import annotations

"""CPU-only read-only attribution for the consumed CHM-v2 #1115 OOM (#1127).

Never loads checkpoint tensors, never allocates a GPU, and never writes to the
consumed scientific result namespace.
"""

import json
from pathlib import Path
from typing import Any

import modal

ISSUE = 1127
SOURCE_CONTROL_ISSUE = 1115
SOURCE_TRIGGER_ISSUE = 1126
SOURCE_SEED = 2_011_121
PHASE = "chm-v2-100m-oom-attribution-1127-v1"
TRIGGER_TITLE = "[modal-chm-v2-100m-oom-attribution-1127-v1]"
SOURCE_ROOT = "/vol/chm-v2/100m-value-projected-eiem/issue-1115/seed-2011121-v2"
VOLUME_NAME = "tam-research-data"
FROZEN_STEPS = (512, 1024, 1536, 2048)
KINDS = ("local", "raw_eiem", "vp_eiem")

app = modal.App("chm-v2-100m-oom-attribution-1127-v1")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")


def _full_sha(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a full lowercase 40-hex SHA")
    return normalized


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"#1127 expected JSON object: {path}")
    return payload


def _checkpoint_record(root: Path, kind: str, step: int) -> dict[str, Any]:
    directory = root / "checkpoints" / kind
    metadata_path = directory / f"step-{step:04d}.json"
    checkpoint_path = directory / f"step-{step:04d}.pt"
    metadata = _read_json(metadata_path)
    allowed = (
        "kind",
        "step",
        "tokens_seen",
        "bytes",
        "sha256",
        "mean_train_nll",
        "lr",
        "resume_authorized",
        "scientific_evaluation_authorized",
    )
    metadata_summary = (
        {key: metadata.get(key) for key in allowed}
        if metadata is not None
        else None
    )
    return {
        "kind": kind,
        "step": step,
        "metadata_exists": metadata_path.is_file(),
        "checkpoint_exists": checkpoint_path.is_file(),
        "metadata": metadata_summary,
        "checkpoint_file_bytes": (
            int(checkpoint_path.stat().st_size) if checkpoint_path.is_file() else None
        ),
    }


def _inventory(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": int(path.stat().st_size),
                "suffix": path.suffix,
                "checkpoint_payload_opened": False,
            }
        )
    return rows


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=5 * 60,
    retries=0,
    volumes={"/vol": volume},
)
def inspect_consumed_oom(
    source_sha: str,
    source_tree: str,
    runner_sha: str,
    workflow_sha: str,
) -> str:
    source_sha = _full_sha(source_sha, "source_sha")
    source_tree = _full_sha(source_tree, "source_tree")
    runner_sha = _full_sha(runner_sha, "runner_sha")
    workflow_sha = _full_sha(workflow_sha, "workflow_sha")

    volume.reload()
    root = Path(SOURCE_ROOT)
    if not root.is_dir():
        raise RuntimeError("#1127 consumed result root is missing")

    markers = {}
    for name in (
        "ZERO_GPU_GATE.json",
        "DISPATCH_RESERVED.json",
        "ATTEMPT_CONSUMED.json",
        "RESULT.json",
        "ATTEMPT_FAILURE.json",
    ):
        markers[name] = _read_json(root / name)

    consumed = markers["ATTEMPT_CONSUMED.json"]
    failure = markers["ATTEMPT_FAILURE.json"]
    if consumed is None or consumed.get("scientific_seed_consumed") is not True:
        raise RuntimeError("#1127 consumed-attempt marker is missing or inconsistent")
    if int(consumed.get("scientific_seed", -1)) != SOURCE_SEED:
        raise RuntimeError("#1127 source seed provenance drift")
    if failure is None:
        raise RuntimeError("#1127 terminal infrastructure failure record is missing")
    if failure.get("scientific_interpretation") is not False:
        raise RuntimeError("#1127 failure must remain non-scientific")
    if markers["RESULT.json"] is not None:
        raise RuntimeError("#1127 unexpected scientific RESULT.json exists")

    checkpoints = {
        kind: [_checkpoint_record(root, kind, step) for step in FROZEN_STEPS]
        for kind in KINDS
    }
    inventory = _inventory(root)

    payload = {
        "classification": "CHM_V2_100M_VALUE_PROJECTED_EIEM_OOM_READONLY_ATTRIBUTION",
        "issue": ISSUE,
        "source_control_issue": SOURCE_CONTROL_ISSUE,
        "source_trigger_issue": SOURCE_TRIGGER_ISSUE,
        "source_scientific_seed": SOURCE_SEED,
        "source_seed_consumed": True,
        "source_root": SOURCE_ROOT,
        "source_sha": source_sha,
        "source_tree": source_tree,
        "runner_blob_sha": runner_sha,
        "workflow_blob_sha": workflow_sha,
        "gpu_allocated": False,
        "checkpoint_loaded": False,
        "writes_performed": False,
        "scientific_interpretation": False,
        "markers": markers,
        "checkpoints": checkpoints,
        "inventory": inventory,
        "inventory_file_count": len(inventory),
        "inventory_total_bytes": sum(item["bytes"] for item in inventory),
    }
    return json.dumps(payload, sort_keys=True)


@app.local_entrypoint()
def main(
    source_sha: str,
    source_tree: str,
    runner_sha: str,
    workflow_sha: str,
) -> None:
    payload = inspect_consumed_oom.remote(
        source_sha,
        source_tree,
        runner_sha,
        workflow_sha,
    )
    print(f"CHM_V2_100M_OOM_ATTRIBUTION={payload}")
