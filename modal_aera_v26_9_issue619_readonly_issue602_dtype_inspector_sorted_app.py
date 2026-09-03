from __future__ import annotations

"""CPU-only import-isolated reader for immutable #602 sorted-row evidence."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import modal

RESEARCH_ISSUE = 619
SOURCE_INSPECTOR_PATH = Path(
    "tam_research/aera_v26_9_issue619_readonly_issue602_dtype_inspector_sorted.py"
)
REMOTE_INSPECTOR_PATH = "/root/aera_v26_9_issue619_frozen_inspector.py"
SOURCE_INSPECTOR_BLOB = "b19744bf7e107409a90b62f26bde9e5cb88f36af"
ISSUE602_RESULT_PATH = "/vol/aera-v26/issue602-identity-weight-visibility/result.json"
ISSUE602_RESULT_SHA256 = "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb"
MARKER = "AERA_V26_9_ISSUE619_READONLY_ISSUE602_DTYPE_INSPECTOR_SORTED_JSON="

APP_NAME = "tam-research-aera-v26-9-issue619-readonly-issue602-dtype-inspector-sorted"
VOLUME_NAME = "tam-research-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").add_local_file(
    SOURCE_INSPECTOR_PATH, REMOTE_INSPECTOR_PATH
)


def _git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def _load_standalone(path: Path, module_name: str) -> ModuleType:
    if _git_blob_sha(path) != SOURCE_INSPECTOR_BLOB:
        raise RuntimeError("issue619 inspector blob drift")
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("issue619 could not create inspector import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    newly_loaded = set(sys.modules) - before
    if "tam_research" in sys.modules or any(
        name.startswith("tam_research.") for name in newly_loaded
    ):
        raise RuntimeError("issue619 unexpectedly imported tam_research package")
    return module


def precheck() -> dict:
    inspector = _load_standalone(SOURCE_INSPECTOR_PATH.resolve(), "issue619_precheck")
    check = inspector.cpu_contract_preflight_issue619()
    return {
        "research_issue": RESEARCH_ISSUE,
        "source_inspector_blob": SOURCE_INSPECTOR_BLOB,
        "source_result_path": inspector.ISSUE602_RESULT_PATH,
        "source_result_sha256": inspector.ISSUE602_RESULT_SHA256,
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
    inspector_path = Path(REMOTE_INSPECTOR_PATH)
    inspector = _load_standalone(inspector_path, "issue619_remote")
    volume.reload()
    path = Path(ISSUE602_RESULT_PATH)
    if not path.exists():
        raise RuntimeError(f"issue619 source result missing: {ISSUE602_RESULT_PATH}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ISSUE602_RESULT_SHA256:
        raise RuntimeError(
            f"issue619 source SHA drift: got={digest} expected={ISSUE602_RESULT_SHA256}"
        )
    result = json.loads(raw)
    extracted = inspector.inspect_issue602_result_sorted(result)
    extracted["verified_source_sha256"] = digest
    return extracted


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(precheck(), sort_keys=True), flush=True)
    result = inspect.remote()
    print(MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
