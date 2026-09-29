from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tam_research.chm_v2_100m_value_projected_eiem_host_staged_rerun import (
    AUDIT_TITLE,
    CONTROL_ISSUE,
    PARENT_ARCHITECTURE_ISSUE,
    PHASE,
    PREREG_ISSUE,
    RESULT_ROOT,
    SCIENTIFIC_SEED,
    SYSTEMS_EVIDENCE_ISSUE,
    TRIGGER_TITLE,
    assert_no_execution_authority,
    protocol_manifest,
    validate_contract,
    validate_seed_for_preparation,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v2_100m_value_projected_eiem_host_staged_1137_v1.py"


def test_1137_contract_freezes_fresh_seed_and_host_staged_transport() -> None:
    manifest = validate_contract()
    assert CONTROL_ISSUE == 1137
    assert PREREG_ISSUE == 1137
    assert PARENT_ARCHITECTURE_ISSUE == 1112
    assert SYSTEMS_EVIDENCE_ISSUE == 1133
    assert SCIENTIFIC_SEED == 2_011_371
    assert PHASE == "chm-v2-100m-value-projected-eiem-host-staged-1137-seed-2011371-v1"
    assert RESULT_ROOT == (
        "/vol/chm-v2/100m-value-projected-eiem-host-staged/issue-1137/seed-2011371-v1"
    )
    assert TRIGGER_TITLE == (
        "[modal-chm-v2-100m-value-projected-eiem-host-staged-1137-seed-2011371-v1]"
    )
    assert AUDIT_TITLE == (
        "[modal-chm-v2-100m-value-projected-eiem-host-staged-1137-authority-audit-v1]"
    )
    assert manifest["host_staged_transport_required"] is True
    assert manifest["full_source_cuda_cache_authorized"] is False
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["trigger_authorized_by_module"] is False
    assert manifest["gpu_allocation_authorized_by_module"] is False
    assert manifest["scientific_seed_consumed_by_module"] is False
    assert manifest["stage_d_authorized"] is False


def test_1137_seed_is_reserved_only_and_consumed_seeds_are_refused() -> None:
    assert validate_seed_for_preparation(2_011_371) == 2_011_371
    with pytest.raises(RuntimeError):
        validate_seed_for_preparation(2_011_371, request_execution=True)
    with pytest.raises(RuntimeError):
        validate_seed_for_preparation(2_011_121)
    with pytest.raises(RuntimeError):
        validate_seed_for_preparation(977_001)
    with pytest.raises(RuntimeError):
        validate_seed_for_preparation(2_011_372)


def test_1137_module_grants_no_execution_authority() -> None:
    assert_no_execution_authority()
    for kwargs in (
        {"gpu": True},
        {"training": True},
        {"trigger_creation": True},
        {"scientific_seed_consumption": True},
        {"stage_d": True},
    ):
        with pytest.raises(RuntimeError):
            assert_no_execution_authority(**kwargs)


def test_successor_runner_is_syntax_valid_and_uses_only_host_staged_corpus_gather() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)

    assert "_device_tokens" not in source
    assert "gather_batch_from_source" not in source
    assert source.count("host_staged_gather") >= 4
    assert "train_source = train_data.data" in source
    assert "val_source = val_data.data" in source
    assert "TokenBin device cache must be empty before host-staged training" in source
    assert "full-source CUDA cache populated during LOCAL phase" in source
    assert "full-source CUDA cache populated during RAW phase" in source
    assert "full-source CUDA cache populated during VP phase" in source
    assert "full-source CUDA cache populated before final classification" in source
    assert '"host_staged_transport": True' in source
    assert '"host_to_device_token_bytes_total"' in source


def test_successor_runner_excludes_retired_1115_executable_identity() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    forbidden = (
        "/vol/chm-v2/100m-value-projected-eiem/issue-1115/seed-2011121-v1",
        "/vol/chm-v2/100m-value-projected-eiem/issue-1115/seed-2011121-v2",
        "[modal-chm-v2-100m-value-projected-eiem-1115-seed-2011121-v1]",
        "[modal-chm-v2-100m-value-projected-eiem-1115-seed-2011121-v2]",
        "SCIENTIFIC_SEED = 2_011_121",
    )
    for token in forbidden:
        assert token not in source

    assert "SCIENTIFIC_SEED = 2_011_371" in source
    assert (
        'RESULT_ROOT = "/vol/chm-v2/100m-value-projected-eiem-host-staged/'
        'issue-1137/seed-2011371-v1"'
    ) in source


def test_successor_runner_preserves_frozen_three_way_science() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "CHMV1100MLocalLM()" in source
    assert "CHMV1100MEIEMLM()" in source
    assert "CHMV2100MValueProjectedEIEMLM()" in source
    assert "_seed_all(SCIENTIFIC_SEED)" in source
    assert 'kind not in {"local", "raw_eiem", "vp_eiem"}' in source
    assert "local_session_logits" in source
    assert "eiem_flat_training_session_logits" in source
    assert "vp_eiem_flat_training_session_logits" in source
    assert "eiem_exact_flat_two_chunk_logits" in source
    assert "vp_eiem_exact_flat_two_chunk_logits" in source
    assert "classify_development_gate" in source
    assert "torch.compile" not in source


def test_successor_runner_still_consumes_seed_before_model_construction_and_training() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    run = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_scientific"
    )
    lines = source.splitlines()
    start = min([run.lineno] + [d.lineno for d in run.decorator_list]) - 1
    end = run.end_lineno
    assert end is not None
    body = "\n".join(lines[start:end])

    consumed = body.index("_atomic_write(consumed_path, consumed)")
    local_construct = body.index("CHMV1100MLocalLM()")
    train_call = body.index("_train_model(")
    assert consumed < local_construct < train_call
    assert '"scientific_seed_consumed": True' in body
    assert '"automatic_retry_authorized": False' in body
    assert '"checkpoint_resume_authorized": False' in body


def test_protocol_manifest_remains_one_seed_development_only() -> None:
    manifest = protocol_manifest()
    assert manifest["training"]["tokens_per_model"] == 33_554_432
    assert manifest["training"]["optimizer_steps"] == 2_048
    assert manifest["training"]["warmup_steps"] == 40
    assert manifest["training"]["micro_batch"] == 4
    assert manifest["training"]["grad_accum"] == 4
    assert manifest["training"]["session_len"] == 1024
    assert manifest["training"]["soft_temperature"] == 0.10
    assert manifest["resource_envelope"]["gpu"] == "L4"
    assert manifest["resource_envelope"]["retries"] == 0
    assert manifest["resource_envelope"]["max_compute_usd"] == 6.00
    assert manifest["multi_seed_replication_authorized"] is False
