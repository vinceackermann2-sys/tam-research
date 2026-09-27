from __future__ import annotations

import ast
from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_label_direction_1099_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-label-direction-1099-v1.yml"
AUDIT = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-label-direction-1099-authority-audit-v2.yml"
PAYLOAD_GATE = ROOT / "tam_research" / "chm_v1_100m_stage_c_payload_gate_diagnostic.py"
MIRROR = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-payload-integration-1064-source-mirror-v5.yml"

V8_RESULT_ROOT = "/vol/chm-v1/100m-stage-c-payload-gate/issue-1017/v8"
CHECKPOINT_SHA256 = "846721098816af9aedd5bd9eb9bf855fdeb7a6f402cd3582912ee5775db64831"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            assert node.end_lineno is not None
            return "\n".join(lines[start:node.end_lineno])
    raise AssertionError(f"{name!r} not found in {path}")


def _heredocs(source: str) -> list[str]:
    lines = source.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if "python - <<'PY'" not in lines[i]:
            i += 1
            continue
        i += 1
        block: list[str] = []
        while i < len(lines) and lines[i].strip() != "PY":
            block.append(lines[i])
            i += 1
        assert i < len(lines), "unterminated Python heredoc"
        blocks.append(textwrap.dedent("\n".join(block)))
        i += 1
    return blocks


def _duplicate_direct_env_keys(source: str) -> list[tuple[int, str, int, int]]:
    lines = source.splitlines()
    duplicates: list[tuple[int, str, int, int]] = []
    for index, line in enumerate(lines):
        if line.strip() != "env:":
            continue
        base = len(line) - len(line.lstrip())
        seen: dict[str, int] = {}
        for child_index in range(index + 1, len(lines)):
            child = lines[child_index]
            if not child.strip():
                continue
            indent = len(child) - len(child.lstrip())
            if indent <= base:
                break
            if indent != base + 2 or ":" not in child.strip():
                continue
            key = child.strip().split(":", 1)[0]
            if key in seen:
                duplicates.append((index + 1, key, seen[key], child_index + 1))
            else:
                seen[key] = child_index + 1
    return duplicates


def test_runner_fresh_identities_and_frozen_payload_gate_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)

    assert 'PHASE = "chm-v1-100m-stage-c-label-direction-1099-v1"' in source
    assert "CONTROL_ISSUE = 1099" in source
    assert "PREREG_ISSUE = 1014" in source
    assert "SOURCE_SCIENTIFIC_ISSUE = 990" in source
    assert "SOURCE_POSTMORTEM_ISSUE = 1008" in source
    assert 'TRIGGER_TITLE = "[modal-chm-v1-100m-stage-c-label-direction-1099-v1]"' in source
    assert 'RESULT_ROOT = "/vol/chm-v1/100m-stage-c-label-direction/issue-1099/v1"' in source
    assert 'SOURCE_RESULT_ROOT = "/vol/chm-v1/source-mirror/issue-1064/v5"' in source

    assert 'PAYLOAD_GATE_BLOB = "32a5ea1baca45705e75fef4c99974668b59d73a6"' in source
    assert 'MODEL_BLOB = "b9b141c0e52d4fd0fff28b12a3588b2adc659b8f"' in source
    assert 'EVALUATOR_BLOB = "863bd038e60da5511503adb0c8e1046a680ed3bd"' in source
    assert (
        'SCIENTIFIC_IMPLEMENTATION_BLOB = '
        '"fb7fe5f8c2a597f7de9e4d03c6908dc0db9558ac"'
    ) in source
    assert 'DUAL_ACCOUNT_BLOB = "adb979e2ecaa7cdf1c0ee36e7a4d929783e078d6"' in source
    assert 'DUAL_ACCOUNT_CLI_BLOB = "440292942066d0b3d3a71d40ca8495092674ce6c"' in source
    assert CHECKPOINT_SHA256 in source
    assert "CHECKPOINT_BYTES = 407_424_818" in source
    assert "MAX_SECONDS = 7_200" in source
    assert "MAX_BILLED_COMPUTE_USD = 3.00" in source

    binding = _function_source(RUNNER, "_validate_bindings")
    assert "1099," in binding
    assert "1014," in binding
    assert "990," in binding
    assert "1008," in binding
    assert "PAYLOAD_GATE_BLOB" in binding


def test_runner_reuses_only_verified_v5_source_mirror_provenance() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    inspect = _function_source(RUNNER, "_inspect_source")
    assert V8_RESULT_ROOT not in source
    assert '"CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_SOURCE_MIRROR_V5_PASS"' in inspect
    assert '"repair_issue": 1085' in inspect
    assert '"control_issue": 1064' in inspect
    assert '"selected_modal_account": "secondary"' in inspect
    assert '"selected_credential_alias": "account2"' in inspect
    assert '"source_and_mirror_byte_identical": True' in inspect
    assert "_sha256_file(checkpoint)" in inspect
    assert "actual_sha != CHECKPOINT_SHA256" in inspect
    assert '"mirror_manifest_verified": True' in inspect
    assert '"diagnostic_result_namespace_unused": not Path(RESULT_ROOT).exists()' in inspect


def test_paid_runner_uses_exact_1014_scoring_and_384_long_range_cases() -> None:
    run = _function_source(RUNNER, "run_diagnostic")
    assert "from tam_research.chm_v1_100m_stage_c_payload_gate_diagnostic import (" in run
    for name in (
        "classify_payload_gate_signals",
        "gate_state_summary",
        "payload_gate_probe_row",
        "summarize_payload_gate_diagnostic",
        "validate_protocol_manifest",
    ):
        assert name in run

    assert "long_probes = [probe for probe in probes if probe.family in LONG_RANGE_FAMILIES]" in run
    assert "len(long_probes) != 384" in run
    assert "family_counts = Counter(probe.family for probe in long_probes)" in run
    assert "payload_gate_probe_row(model, probe)" in run
    assert "summary = summarize_payload_gate_diagnostic(rows, gate_state)" in run
    assert "decision = classify_payload_gate_signals(summary)" in run
    assert '"nll_benefits": decision["nll_benefits"]' in run
    assert '"gate_sigmoid_median": decision["gate_sigmoid_median"]' in run
    assert 'decision["benefits"]' not in run
    assert "payload_integration_probe_row" not in run
    assert "classify_payload_integration" not in run


def test_attempt_consumption_precedes_checkpoint_hash_load_and_scoring() -> None:
    run = _function_source(RUNNER, "run_diagnostic")
    consumed = run.index("_atomic_write(consumed_path, consumed)")
    commit = run.index("volume.commit()", consumed)
    checkpoint = run.index("checkpoint = Path(SOURCE_CHECKPOINT_PATH)")
    first_hash = run.index("_sha256_file(checkpoint)")
    load = run.index("torch.load(checkpoint")
    score = run.index("payload_gate_probe_row(model, probe)")
    assert consumed < commit < checkpoint < first_hash < load < score
    assert '"diagnostic_attempt_consumed": True' in run
    assert "retries=0" in run
    assert "gpu=GPU_CLASS" in run
    assert "timeout=MAX_SECONDS" in run


def test_runner_has_no_training_resume_or_checkpoint_mutation_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for forbidden in (
        "torch.optim",
        ".backward(",
        "zero_grad(",
        "optimizer.step",
        "torch.save(",
        "resume_from",
    ):
        assert forbidden not in source
    assert '"new_scientific_seed_created": False' in source
    assert '"source_scientific_seed_reused": False' in source
    assert '"checkpoint_resume_authorized": False' in source
    assert '"automatic_retry_authorized": False' in source
    assert '"stage_d_automatically_authorized": False' in source


def test_payload_gate_module_is_still_the_pre_result_1014_contract() -> None:
    source = PAYLOAD_GATE.read_text(encoding="utf-8")
    assert "ISSUE = 1014" in source
    assert 'CLASSIFICATION = (' in source
    assert '"NO_MEMORY"' in source
    assert '"ANSWER_TOKEN_LEARNED_GATE"' in source
    assert '"ANSWER_TOKEN_FULL_RESIDUAL"' in source
    assert '"LABEL_DIRECTION_LEARNED_GATE"' in source
    assert '"LABEL_DIRECTION_FULL_RESIDUAL"' in source
    assert "GATE_SUPPRESSION_FULL_MIN = 0.10" in source
    assert "RAW_PAYLOAD_FORMAT_LABEL_MIN = 0.10" in source
    assert "RAW_PAYLOAD_USABLE_MIN = 0.05" in source
    assert "RESIDUAL_CHANNEL_WEAK_MAX = 0.10" in source
    assert 'benefits["label_direction_full_residual"]' in source
    assert "PAYLOAD_GATE_DIAGNOSTIC_INCONCLUSIVE" in source


def test_launch_workflow_is_one_shot_authority_bound_and_concurrency_safe() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source and "types: [opened]" in source
    assert "[modal-chm-v1-100m-stage-c-label-direction-1099-v1]" in source
    assert '"phase":"chm-v1-100m-stage-c-label-direction-1099-v1"' in source
    assert '"control_issue":1099' in source
    assert '"prereg_issue":1014' in source
    assert '"/vol/chm-v1/100m-stage-c-label-direction/issue-1099/v1"' in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V1_100M_STAGE_C_LABEL_DIRECTION_FINAL_LAUNCHER_AUTHORITY_V1" in source

    assert 'git -C source merge-base --is-ancestor "$SOURCE_SHA" "$LIVE_MAIN"' in source
    assert "tam_research/chm_v1_100m_stage_c_payload_gate_diagnostic.py" in source
    assert "32a5ea1baca45705e75fef4c99974668b59d73a6" in source
    assert "modal_chm_v1_100m_stage_c_label_direction_1099_v1.py" in source
    assert "modal-chm-v1-100m-stage-c-payload-integration-1064-source-mirror-v5.yml" in source

    assert 'assert account=="secondary"' in source
    assert "scripts/modal_select_account_v3.py" in source
    assert "modal_runtime_admission_probe_1067_v1.py" in source
    assert "modal billing rates --json" in source
    assert "worst=hourly*2.0" in source
    assert "worst<=3.00" in source
    for phase in ("preflight", "reserve", "run", "state"):
        assert f"--phase {phase}" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "continue-on-error: true" in source
    assert "payload_integration_decomposition_verified=true" in source
    assert V8_RESULT_ROOT not in source


def test_terminal_serializer_uses_exact_1014_schema_not_retired_v8_schema() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "CHM_V1_100M_STAGE_C_LABEL_DIRECTION_AUTHORITATIVE_RESULT" in source
    assert 'benefits=result["nll_benefits"]' in source
    assert "answer_token_learned_gate_benefit=" in source
    assert "answer_token_full_residual_benefit=" in source
    assert "label_direction_learned_gate_benefit=" in source
    assert "label_direction_full_residual_benefit=" in source
    assert "raw_payload_answer_direction_cosine=" in source
    assert 'result["benefits"]' not in source
    assert 'result["aggregate"]' not in source
    assert "gh issue comment 1099 --repo" in source
    assert "--payload-gate-sha" in source
    assert "--decomposition-sha" not in source
    assert "36343365415" in source
    assert "1101" in source
    assert "actions: read" in source
    assert "gh issue close 1099 --repo" in source


def test_authority_audit_is_cpu_only_and_verifies_prior_authoritative_evidence() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v1-100m-stage-c-label-direction-1099-authority-audit-v2]" in source
    assert '"phase":"chm-v1-100m-stage-c-label-direction-1099-authority-audit-v2"' in source
    assert '"control_issue":1099' in source
    assert '"prereg_issue":1014' in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert 'git -C source merge-base --is-ancestor "$SOURCE_SHA" "$LIVE_MAIN"' in source

    assert "CHM_V1_100M_STAGE_C_POSTMORTEM_AUTHORITATIVE_RESULT" in source
    assert 'tags=[\"PAYLOAD_INTEGRATION_WEAK_SIGNAL\"]' in source
    assert "CHM_V1_100M_STAGE_C_PAYLOAD_INTEGRATION_AUTHORITATIVE_RESULT" in source
    assert 'tags=[\"PAYLOAD_DIRECTION_WEAK_SIGNAL\", \"LEARNED_GATE_HARM_SIGNAL\", \"UNIT_GATE_DESTABILIZATION_SIGNAL\"]' in source
    assert "learned_oracle_benefit=-0.030297408424" in source
    assert "unit_oracle_benefit=-1.879134013706" in source

    assert "--phase inspect-source" in source
    assert "--phase preflight" not in source
    assert "--phase reserve" not in source
    assert "--phase run " not in source
    assert "--phase state" not in source
    assert "result_namespace_unused=true" in source
    assert "gpu_allocated=false" in source
    assert "diagnostic_attempt_consumed=false" in source
    assert "trigger_authorized=false" in source
    assert "payload_integration_decomposition_verified=true" in source
    assert "retired_v8_partial_artifacts_excluded=true" in source
    assert "gh issue comment 1099 --repo" in source


def test_audit_may_reject_v8_literal_but_runner_and_launch_never_reference_it() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    assert V8_RESULT_ROOT not in runner
    assert V8_RESULT_ROOT not in workflow
    assert V8_RESULT_ROOT in audit  # static forbidden-pattern check only
    assert "grep -F" in audit


def test_source_mirror_remains_storage_only_and_is_not_recreated() -> None:
    source = MIRROR.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "modal run " not in source
    assert "modal shell " not in source
    assert "modal deploy " not in source
    assert "modal volume get" in source
    assert "modal volume put" in source
    assert CHECKPOINT_SHA256 in source
    assert "SOURCE_MIRROR_V5_PASS" in source


def test_all_new_workflow_embedded_python_is_syntax_valid() -> None:
    for path in (WORKFLOW, AUDIT):
        source = path.read_text(encoding="utf-8")
        blocks = _heredocs(source)
        assert blocks, path
        for index, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name} embedded Python block {index} is invalid: {exc}"
                ) from exc


def test_no_duplicate_direct_env_keys_or_top_level_sequence_regressions() -> None:
    for path in (WORKFLOW, AUDIT):
        source = path.read_text(encoding="utf-8")
        assert _duplicate_direct_env_keys(source) == [], path
        assert not any(line.startswith("- ") for line in source.splitlines()), path


def test_new_execution_files_preserve_no_training_no_new_seed_no_stage_d() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (RUNNER, WORKFLOW, AUDIT)
    )
    for forbidden in ("torch.optim", ".backward(", "torch.save(", "optimizer.step"):
        assert forbidden not in combined
    assert "new_scientific_seed_authorized=false" in combined
    assert "stage_d_authorized=false" in combined
    assert "automatic_retry_authorized=false" in combined


def test_authority_audit_durable_record_exports_every_shell_variable_it_uses() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    marker = "- name: Record immutable pre-authority audit"
    assert marker in source
    step = source.split(marker, 1)[1]
    if "\n      - name:" in step:
        step = step.split("\n      - name:", 1)[0]
    env_part, run_part = step.split("\n        run: |", 1)
    env_keys: set[str] = set()
    for line in env_part.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0]
        if key.isupper():
            env_keys.add(key)
    referenced = set(re.findall(r"\$([A-Z][A-Z0-9_]*)", run_part))
    assert referenced - {"GITHUB_REPOSITORY"} <= env_keys


def test_runner_cli_flag_matches_all_executable_workflows() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    audit = AUDIT.read_text(encoding="utf-8")
    assert "payload_gate_sha: str" in runner
    assert "--payload-gate-sha" in workflow
    assert "--payload-gate-sha" in audit
    assert "--decomposition-sha" not in workflow
    assert "--decomposition-sha" not in audit
