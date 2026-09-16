from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_990_v1.py"
IMPLEMENTATION = ROOT / "modal_chm_v1_100m_stage_c_990_v1_impl.py"
AUDIT_V2 = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-990-authority-audit-v2.yml"
SCIENTIFIC_WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-990-v1.yml"

OLD_IMPLEMENTATION_BLOB = "fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"
FROZEN_SCIENTIFIC_TRIGGER = "[modal-chm-v1-100m-stage-c-988-seed-977001-v1]"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def test_preserved_scientific_implementation_is_exact_pr991_blob() -> None:
    assert IMPLEMENTATION.is_file()
    assert _git_blob_sha(IMPLEMENTATION) == OLD_IMPLEMENTATION_BLOB


def test_public_runner_is_lazy_shim_with_no_top_level_torch_import() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert "torch" not in imports
    assert "_LazyTorchProxy" in source
    assert f'IMPLEMENTATION_BLOB_SHA = "{OLD_IMPLEMENTATION_BLOB}"' in source
    assert "exec(compile(_impl_bytes" in source


def test_runner_can_be_evaluated_with_torch_import_explicitly_blocked() -> None:
    script = r'''
import builtins
import runpy
import sys
import types

class FakeApp:
    def function(self, *args, **kwargs):
        def decorate(fn):
            return fn
        return decorate
    def local_entrypoint(self, *args, **kwargs):
        def decorate(fn):
            return fn
        return decorate

class FakeVolume:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

class FakeImage:
    @classmethod
    def debian_slim(cls, *args, **kwargs):
        return cls()
    def pip_install(self, *args, **kwargs):
        return self
    def add_local_python_source(self, *args, **kwargs):
        return self

modal = types.ModuleType("modal")
modal.App = lambda *args, **kwargs: FakeApp()
modal.Volume = FakeVolume
modal.Image = FakeImage
sys.modules["modal"] = modal
sys.modules.pop("torch", None)

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("host torch import is forbidden by #993 regression test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

namespace = runpy.run_path("modal_chm_v1_100m_stage_c_990_v1.py")
assert namespace["SCIENTIFIC_SEED"] == 977001
assert namespace["PHASE"] == "chm-v1-100m-stage-c-990-v1"
assert namespace["IMPLEMENTATION_BLOB_SHA"] == "fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"
assert callable(namespace["inspect_state"])
'''
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_v2_audit_is_fresh_read_only_pre_authority_surface() -> None:
    source = AUDIT_V2.read_text(encoding="utf-8")
    assert "[modal-chm-v1-100m-stage-c-990-authority-audit-v2]" in source
    assert "chm-v1-100m-stage-c-990-authority-audit-v2" in source
    assert "github.run_attempt" in source
    assert "35080849066" in source
    assert "supersedes_failed_audit_issue" in source
    assert "modal billing rates --json" in source
    assert "--phase inspect" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "CHM_V1_990_AUTHORITY_AUDIT_V2" in source
    assert "trigger_authorized=false" in source
    assert "gpu_allocated=false" in source
    assert "scientific_seed_consumed=false" in source
    assert OLD_IMPLEMENTATION_BLOB in source


def test_scientific_trigger_contract_remains_frozen() -> None:
    source = SCIENTIFIC_WORKFLOW.read_text(encoding="utf-8")
    assert FROZEN_SCIENTIFIC_TRIGGER in source
    assert "workflow_dispatch" not in source
    assert "scientific_seed=977001" in source
    assert "CHM_V1_990_FINAL_LAUNCHER_AUTHORITY_V1" in source
