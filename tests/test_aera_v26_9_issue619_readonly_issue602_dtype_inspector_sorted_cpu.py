from __future__ import annotations

import ast
from pathlib import Path

INSPECTOR = Path("tam_research/aera_v26_9_issue619_readonly_issue602_dtype_inspector_sorted.py")
LAUNCHER = Path("modal_aera_v26_9_issue619_readonly_issue602_dtype_inspector_sorted_app.py")
WORKFLOW = Path(".github/workflows/aera-v26-9-issue619-readonly-issue602-dtype-inspector-sorted.yml")

EXPECTED_LOGICAL = (
    "fp32_source_bf16_projection_fp32_durable_batch8_mixed",
    "fp32_source_bf16_projection_fp32_durable_batch8_full",
    "fp32_source_bf16_projection_fp32_durable_batch64_mixed",
    "fp32_source_bf16_projection_fp32_durable_batch64_full",
)
EXPECTED_DURABLE = tuple(sorted(EXPECTED_LOGICAL))


def test_issue619_python_sources_parse() -> None:
    ast.parse(INSPECTOR.read_text())
    ast.parse(LAUNCHER.read_text())


def test_issue619_sorted_durable_contract_is_exact() -> None:
    ns: dict = {}
    exec(compile(INSPECTOR.read_text(), str(INSPECTOR), "exec"), ns)
    assert ns["LOGICAL_ROWS"] == EXPECTED_LOGICAL
    assert ns["DURABLE_ROW_ORDER"] == EXPECTED_DURABLE
    pre = ns["cpu_contract_preflight_issue619"]()
    assert pre["gpu_authorized_by_cpu_preflight"] is False
    assert pre["model_constructed"] is False
    assert pre["checkpoint_loaded"] is False
    assert pre["volume_mutated"] is False


def test_issue619_inspector_keeps_all_scientific_gates() -> None:
    text = INSPECTOR.read_text()
    for field in (
        "selection_semantically_equivalent",
        "pre_out_recalled_close",
        "final_out_close",
        "query_and_normalized_keys_bit_exact",
        "source_unchanged",
        "finite",
        "dtype_device_shape_exact",
        "direct_tail_topology_pass",
        "full_backend_no_reference_tail_ops",
    ):
        assert field in text
    assert 'row.get("pass") is not False' in text
    assert 'row.get("dtype_split_exact") is not False' in text
    assert "set(rows.keys()) != set(LOGICAL_ROWS)" in text
    assert "observed_order != DURABLE_ROW_ORDER" in text
    assert "dtype_inference_or_recomputation" in text


def test_issue619_launcher_is_import_isolated_cpu_read_only() -> None:
    text = LAUNCHER.read_text()
    assert "spec_from_file_location" in text
    assert ".add_local_file(" in text
    assert "add_local_python_source" not in text
    assert "from tam_research" not in text
    assert "import tam_research" not in text
    assert "gpu=" not in text
    assert "volume.commit(" not in text
    assert "write_text(" not in text
    assert "write_bytes(" not in text
    assert "inspect_issue602_result_sorted" in text
    assert "5ab64b2aa9750babebec6e681c7be587f079436436b5a3cda86ac809018256fb" in text


def test_issue619_workflow_is_single_attempt_single_modal_run() -> None:
    text = WORKFLOW.read_text()
    assert "workflow_dispatch" not in text
    assert "GITHUB_RUN_ATTEMPT" in text
    assert "## #619 sole CPU inspector authorization" in text
    assert "[aera-v26-9-issue619-readonly-issue602-dtype-inspector-sorted]" in text
    assert text.count("modal run modal_aera_v26_9_issue619_readonly_issue602_dtype_inspector_sorted_app.py") == 1
    assert "33793519431" in text
    assert "100775569626" in text
    assert "gpu=" not in text
