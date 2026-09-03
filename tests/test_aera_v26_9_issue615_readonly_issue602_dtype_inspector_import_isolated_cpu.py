from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_9_issue615_readonly_issue602_dtype_inspector_import_isolated_app.py"
WORKFLOW = ROOT / ".github/workflows/aera-v26-9-issue615-readonly-issue602-dtype-inspector-import-isolated.yml"
INSPECTOR = ROOT / "tam_research/aera_v26_9_issue605_readonly_issue602_dtype_inspector.py"

INSPECTOR_BLOB = "bcf22ae6e04b1a8cc2e39316627e5be7aec3e22b"
LAUNCHER_BLOB = "f2930847c0acf82caab27a4d957157de877be432"


def _git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def test_issue615_frozen_inspector_is_exact_and_standalone() -> None:
    raw = INSPECTOR.read_bytes()
    assert _git_blob_sha(raw) == INSPECTOR_BLOB
    source = raw.decode()
    assert "import torch" not in source
    assert "from tam_research" not in source
    assert "import tam_research" not in source


def test_issue615_launcher_is_import_isolated_cpu_read_only() -> None:
    raw = LAUNCHER.read_bytes()
    assert _git_blob_sha(raw) == LAUNCHER_BLOB
    source = raw.decode()
    tree = ast.parse(source)

    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert all(not name.startswith("tam_research") for name in imported)

    assert "spec_from_file_location" in source
    assert ".add_local_file(" in source
    assert "add_local_python_source" not in source
    assert INSPECTOR_BLOB in source
    assert "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb" in source
    assert "gpu=" not in source
    assert "volume.commit(" not in source
    assert "write_bytes(" not in source
    assert "write_text(" not in source
    assert "open(" not in source
    assert 'cpu=1' in source
    assert '"tam_research_package_imported": False' in source
    assert '"model_execution": False' in source
    assert '"repair_authorized": False' in source
    assert '"end_to_end_systems_authorized": False' in source


def test_issue615_workflow_is_fresh_one_shot_attempt1_only() -> None:
    workflow = WORKFLOW.read_text()
    assert "[aera-v26-9-issue615-readonly-issue602-dtype-inspector]" in workflow
    assert "## #615 sole CPU inspector authorization" in workflow
    assert "33790448325" in workflow
    assert "100765501032" in workflow
    assert 'select(.name=="Run sole issue610 CPU inspection")' in workflow
    assert 'select(.name=="Authenticate Modal")' in workflow
    assert workflow.count(
        "modal run modal_aera_v26_9_issue615_readonly_issue602_dtype_inspector_import_isolated_app.py"
    ) == 1
    assert "workflow_dispatch" not in workflow
    assert "rerun" not in workflow.lower()
    assert "pip install torch" not in workflow.lower()
    assert INSPECTOR_BLOB in workflow
    assert LAUNCHER_BLOB in workflow
    assert "AERA_V26_9_ISSUE605_READONLY_ISSUE602_DTYPE_INSPECTOR_JSON=" in workflow
