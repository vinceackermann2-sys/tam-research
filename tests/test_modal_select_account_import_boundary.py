from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts" / "modal_select_account.py"
DUAL = ROOT / "tam_research" / "modal_dual_account.py"


def test_selector_cli_import_boundary_never_imports_package_init_or_torch() -> None:
    script = f"""
import builtins
import runpy

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("torch import forbidden in selector import-boundary test")
    if name == "tam_research" or name.startswith("tam_research."):
        raise RuntimeError("tam_research package import forbidden in selector import-boundary test")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
namespace = runpy.run_path({str(SELECTOR)!r}, run_name="selector_import_boundary_test")
assert callable(namespace["select_from_environment"])
assert namespace["_MODULE_PATH"] == {str(DUAL)!r}
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_selector_source_uses_direct_file_loader_not_package_import() -> None:
    source = SELECTOR.read_text(encoding="utf-8")
    assert "from tam_research.modal_dual_account import" not in source
    assert "import tam_research" not in source
    assert "spec_from_file_location" in source
    assert 'Path(__file__).resolve().parents[1] / "tam_research" / "modal_dual_account.py"' in source
    assert "sys.modules[_MODULE_NAME] = _module" in source
    assert "_spec.loader.exec_module(_module)" in source


def test_dual_account_policy_module_remains_torch_free() -> None:
    source = DUAL.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source
