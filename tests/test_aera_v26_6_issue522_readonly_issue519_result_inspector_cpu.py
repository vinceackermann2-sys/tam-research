from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modal_aera_v26_6_issue522_readonly_issue519_result_inspector_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-6-issue522-readonly-issue519-result-inspector.yml"
CANDIDATE = ROOT / "tam_research" / "aera_hardware_core_v26_6_ficem_write_materialize_cast.py"
PROBE = ROOT / "tam_research" / "aera_v26_6_issue519_ficem_write_materialize_cast_probe.py"

SOURCE_MAIN = "b4c117b01ae851327e9feca25ea3c12078831904"
SOURCE_RESULT_PATH = "/vol/aera-v26/issue519-ficem-write-materialize-cast/result.json"
SOURCE_RESULT_SHA256 = "b9fba0fca96644ef8db9bc46faf2c73d0c0cc1f1aaac6a321abe2411d3703cd5"
CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
PROBE_BLOB = "ec22807434192f58e292bffc3de9828be2b44272"
LAUNCHER_BLOB = "9fb159be98c006d7181445a5a9c8be64eb454e4d"
WORKFLOW_BLOB = "ed808cab079cbab81f636b2fedf19599ebb9d524"
FROZEN_PASS_MASKS = (
    0, 1, 2, 3, 36, 37, 38, 39, 72, 73, 74, 75, 108, 109, 110, 111,
    144, 145, 146, 147, 180, 181, 182, 183, 216, 217, 218, 219,
    252, 253, 254, 255,
)


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _literal_constant(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant {name}")


def test_issue522_freezes_consumed_issue519_evidence_and_reader_blobs() -> None:
    assert _blob(CANDIDATE) == CANDIDATE_BLOB
    assert _blob(PROBE) == PROBE_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB
    assert _literal_constant(LAUNCHER, "SOURCE_MAIN") == SOURCE_MAIN
    assert _literal_constant(LAUNCHER, "SOURCE_RESULT_PATH") == SOURCE_RESULT_PATH
    assert _literal_constant(LAUNCHER, "SOURCE_RESULT_SHA256") == SOURCE_RESULT_SHA256
    assert _literal_constant(LAUNCHER, "SOURCE_RUN") == 33672232063
    assert _literal_constant(LAUNCHER, "SOURCE_JOB") == 100388368044
    assert _literal_constant(LAUNCHER, "SOURCE_ATTEMPT") == 1
    assert _literal_constant(LAUNCHER, "CANDIDATE_BLOB") == CANDIDATE_BLOB
    assert _literal_constant(LAUNCHER, "PROBE_BLOB") == PROBE_BLOB


def test_issue522_freezes_exact_pass_partition_and_representative_rows() -> None:
    assert _literal_constant(LAUNCHER, "FROZEN_PASS_MASKS") == FROZEN_PASS_MASKS
    assert _literal_constant(LAUNCHER, "REPRESENTATIVE_MASKS") == (
        0, 4, 8, 16, 32, 36, 64, 72, 128, 144, 252, 255
    )
    assert len(FROZEN_PASS_MASKS) == 32
    expected = tuple(
        mask
        for mask in range(256)
        if ((mask >> 2) & 1) == ((mask >> 5) & 1)
        and ((mask >> 3) & 1) == ((mask >> 6) & 1)
        and ((mask >> 4) & 1) == ((mask >> 7) & 1)
    )
    assert FROZEN_PASS_MASKS == expected


def test_issue522_launcher_is_strictly_read_only_cpu_inspection() -> None:
    source = LAUNCHER.read_text()
    assert 'volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)' in source
    assert "volume.reload()" in source
    assert "path.read_bytes()" in source
    assert "hashlib.sha256(raw).hexdigest()" in source
    assert 'json.loads(raw)' in source
    assert 'gpu_used": False' in source
    assert 'experiment_rerun": False' in source
    assert 'repair_authorized": False' in source
    assert "gpu=" not in source
    assert "volume.commit" not in source
    for forbidden in (
        "write_text(",
        "write_bytes(",
        ".unlink(",
        ".rename(",
        ".replace(",
        "torch.",
        "run_mixed_dtype_write_probe",
        "run_materialize_cast_write_probe",
    ):
        assert forbidden not in source


def test_issue522_classification_covers_failure_categories_and_fields() -> None:
    source = LAUNCHER.read_text()
    for category in (
        "exception_or_error",
        "source_mutation",
        "validity_mismatch",
        "dtype_mismatch",
        "device_or_shape_mismatch",
        "nonfinite",
    ):
        assert f'"{category}"' in source
    for field in ("keys", "values", "strengths"):
        assert field in source
    for required in (
        "float_close_failures",
        "dtype_exact_failures",
        "pair_dtype_mismatch_rows",
        "max_abs_by_destination_dtype",
        "frozen_pass_masks_match",
        "representative_direct_rows",
        "edge_rows",
        "public_rows",
        "topology_rows",
    ):
        assert required in source


def test_issue522_workflow_is_one_shot_cpu_only_and_exact_bound_main() -> None:
    source = WORKFLOW.read_text()
    lowered = source.lower()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "cancel-in-progress: false" in source
    assert "[aera-v26-6-issue522-readonly-issue519-result-inspector]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert f"git merge-base --is-ancestor {SOURCE_MAIN} HEAD" in source
    assert "33672232063" in source
    assert "100388368044" in source
    assert CANDIDATE_BLOB in source
    assert PROBE_BLOB in source
    assert LAUNCHER_BLOB in source
    assert source.count("modal run modal_aera_v26_6_issue522_readonly_issue519_result_inspector_app.py") == 1
    assert 'gpu="' not in lowered
    assert "gpu: " not in lowered
    assert "modal deploy" not in source
    assert "volume.commit" not in source
    for forbidden_command in ("gh run rerun", "rerun-failed", "re-run-failed", "workflow_dispatch"):
        assert forbidden_command not in lowered
    assert "redispatch" not in lowered


def test_issue522_higher_authorizations_remain_false() -> None:
    source = LAUNCHER.read_text()
    for key in (
        "repair_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert f'"{key}": False' in source
