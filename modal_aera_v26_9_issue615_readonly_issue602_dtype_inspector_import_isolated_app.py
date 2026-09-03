from __future__ import annotations

"""Import-isolated CPU-only Modal reader for the immutable #602 result."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import modal

APP_NAME = "tam-research-aera-v26-9-issue615-readonly-issue602-dtype-inspector"
VOLUME_NAME = "tam-research-data"
MARKER = "AERA_V26_9_ISSUE605_READONLY_ISSUE602_DTYPE_INSPECTOR_JSON="
INSPECTOR_GIT_BLOB = "bcf22ae6e04b1a8cc2e39316627e5be7aec3e22b"
ISSUE602_RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"
ISSUE602_RESULT_SHA256 = "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"

LOCAL_INSPECTOR_PATH = (
    Path(__file__).resolve().parent
    / "tam_research"
    / "aera_v26_9_issue605_readonly_issue602_dtype_inspector.py"
)
REMOTE_INSPECTOR_PATH = Path("/root/aera_v26_9_issue605_frozen_inspector.py")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").add_local_file(
    LOCAL_INSPECTOR_PATH,
    str(REMOTE_INSPECTOR_PATH),
)


def _git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def _load_frozen_inspector(path: Path) -> ModuleType:
    raw = path.read_bytes()
    digest = _git_blob_sha(raw)
    if digest != INSPECTOR_GIT_BLOB:
        raise RuntimeError(
            f"issue615 inspector blob drift: got={digest} expected={INSPECTOR_GIT_BLOB}"
        )
    spec = importlib.util.spec_from_file_location(
        "aera_v26_9_issue605_frozen_inspector",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("issue615 could not create frozen inspector module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def precheck() -> dict:
    inspector = _load_frozen_inspector(LOCAL_INSPECTOR_PATH)
    check = inspector.cpu_contract_preflight_issue605()
    if inspector.ISSUE602_RESULT_PATH != ISSUE602_RESULT_PATH:
        raise RuntimeError("issue615 result path drift")
    if inspector.ISSUE602_RESULT_SHA256 != ISSUE602_RESULT_SHA256:
        raise RuntimeError("issue615 result SHA drift")
    return {
        "research_issue": 615,
        "source_inspector_issue": 605,
        "source_inspector_blob": INSPECTOR_GIT_BLOB,
        "source_result_path": ISSUE602_RESULT_PATH,
        "source_result_sha256": ISSUE602_RESULT_SHA256,
        "import_isolated": True,
        "tam_research_package_imported": False,
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
    inspector = _load_frozen_inspector(REMOTE_INSPECTOR_PATH)
    if inspector.ISSUE602_RESULT_PATH != ISSUE602_RESULT_PATH:
        raise RuntimeError("issue615 remote result path drift")
    if inspector.ISSUE602_RESULT_SHA256 != ISSUE602_RESULT_SHA256:
        raise RuntimeError("issue615 remote result SHA drift")
    volume.reload()
    path = Path(ISSUE602_RESULT_PATH)
    if not path.exists():
        raise RuntimeError(f"issue615 source result missing: {ISSUE602_RESULT_PATH}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ISSUE602_RESULT_SHA256:
        raise RuntimeError(
            f"issue615 source SHA drift: got={digest} expected={ISSUE602_RESULT_SHA256}"
        )
    result = json.loads(raw)
    extracted = inspector.inspect_issue602_result(result)
    extracted["verified_source_sha256"] = digest
    extracted["import_isolated_successor_issue"] = 615
    return extracted


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(precheck(), sort_keys=True), flush=True)
    result = inspect.remote()
    print(MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
