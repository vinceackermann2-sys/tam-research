from __future__ import annotations

import inspect

import pytest

import tam_research.chm_v1_100m_stage_c_run_control_prep as prep


def test_run_control_prep_manifest_is_exact_and_grants_zero_execution_authority() -> None:
    manifest = prep.validate_contract()
    assert manifest == prep.protocol_manifest()
    assert manifest["classification"] == (
        "CHM_V1_100M_STAGE_C_RUN_CONTROL_PREPARATION_NO_EXECUTION_AUTHORITY"
    )
    assert manifest["parent_research_issue"] == 977
    assert manifest["evaluator_issue"] == 986
    assert manifest["run_control_prep_issue"] == 988
    assert manifest["scientific_seed_future_eligible_not_authorized"] == 977_001
    assert manifest["consumed_stage_b_seed_never_reuse"] == 977_201

    for key in (
        "gpu_authorized",
        "modal_authorized",
        "paid_compute_authorized",
        "training_authorized",
        "trigger_creation_authorized",
        "scientific_execution_authorized",
        "scientific_seed_consumed",
        "stage_d_automatically_authorized",
    ):
        assert manifest[key] is False


def test_scientific_budget_is_33554432_per_model_and_67108864_for_pair() -> None:
    manifest = prep.validate_contract()
    training = manifest["training"]
    assert training["tokens_per_model"] == 33_554_432
    assert training["pair_tokens_total"] == 67_108_864
    assert training["tokens_per_optimizer_step"] == 16_384
    assert training["optimizer_steps_per_model"] == 2_048
    assert training["warmup_steps"] == 40
    assert training["session_len"] == 1_024
    assert training["micro_batch"] == 4
    assert training["grad_accum"] == 4
    assert training["train_stream_generator_seed"] == 987_001
    assert training["byte_identical_pair_stream_required"] is True
    assert training["compile_enabled"] is False


def test_checkpoint_schedule_is_audit_only_until_final_step() -> None:
    assert prep.CHECKPOINT_STEPS == (512, 1024, 1536, 2048)
    assert prep.FINAL_EVALUATION_STEP == prep.OPTIMIZER_STEPS_PER_MODEL == 2_048
    for step in (512, 1024, 1536):
        assert prep.checkpoint_role(step) == "audit-only"
    assert prep.checkpoint_role(2048) == "final-scientific-evaluation"
    with pytest.raises(ValueError, match="checkpoint schedule"):
        prep.checkpoint_role(1025)

    checkpoints = prep.protocol_manifest()["checkpoints"]
    assert checkpoints["resume_from_checkpoint_authorized"] is False
    assert checkpoints["model_selection_authorized"] is False


def test_seed_guard_records_977001_as_future_eligible_but_refuses_execution() -> None:
    assert prep.validate_seed_for_preparation(977_001) == 977_001
    with pytest.raises(RuntimeError, match="no scientific execution authority"):
        prep.validate_seed_for_preparation(977_001, request_execution=True)
    with pytest.raises(RuntimeError, match="consumed seed"):
        prep.validate_seed_for_preparation(977_201)
    with pytest.raises(RuntimeError, match="non-scientific"):
        prep.validate_seed_for_preparation(977_099)
    with pytest.raises(RuntimeError, match="recognizes only"):
        prep.validate_seed_for_preparation(977_999)


def test_frozen_source_and_evaluator_bindings_fail_closed() -> None:
    prep.validate_source_bindings(
        main_sha="72e432f3a0bbf1fb6a27da928f0078dab9491a73",
        main_tree="ae017fff0156fd9409025a672dd314d776285c7a",
        model_contract_blob="b9b141c0e52d4fd0fff28b12a3588b2adc659b8f",
        evaluator_blob="863bd038e60da5511503adb0c8e1046a680ed3bd",
    )
    with pytest.raises(RuntimeError, match="source binding mismatch"):
        prep.validate_source_bindings(
            main_sha="0" * 40,
            main_tree=prep.STARTING_MAIN_TREE,
            model_contract_blob=prep.MODEL_CONTRACT_BLOB,
            evaluator_blob=prep.EVALUATOR_BLOB,
        )

    evaluator = prep.protocol_manifest()["evaluator"]
    assert evaluator == {
        "generator_version": "chm-v1-100m-heldout-aligned-v4",
        "probe_seed": 977_301,
        "validation_seed": 977_302,
        "bootstrap_seed": 977_303,
        "cases_per_family": 128,
        "total_probes": 512,
        "long_range_probes": 384,
        "validation_tokens": 1_048_576,
        "bootstrap_resamples": 10_000,
    }


def test_future_resource_cap_is_pure_arithmetic_and_fails_closed() -> None:
    assert prep.projected_worst_case_compute_usd(2.0) == pytest.approx(24.0)
    accepted = prep.validate_live_rate_cap(2.0)
    assert accepted["within_cap"] is True
    assert accepted["worst_case_compute_usd"] == pytest.approx(24.0)
    with pytest.raises(RuntimeError, match="fail closed"):
        prep.validate_live_rate_cap(2.1)
    with pytest.raises(ValueError, match="positive"):
        prep.projected_worst_case_compute_usd(0.0)

    envelope = prep.protocol_manifest()["future_resource_envelope_not_authorized"]
    assert envelope["gpu"] == "1x NVIDIA L4"
    assert envelope["cpu_cores"] == 4
    assert envelope["ram_gib"] == 16
    assert envelope["max_gpu_seconds"] == 43_200
    assert envelope["max_compute_usd"] == 25.00
    assert envelope["retries"] == 0


def test_preparation_refuses_any_requested_execution_capability() -> None:
    prep.assert_no_execution_authority()
    for capability in (
        "gpu",
        "modal",
        "paid_compute",
        "training",
        "trigger_creation",
        "scientific_seed_consumption",
    ):
        with pytest.raises(RuntimeError, match="refuses execution authority"):
            prep.assert_no_execution_authority(**{capability: True})


def test_reserved_result_namespace_and_trigger_are_not_dispatched() -> None:
    manifest = prep.protocol_manifest()
    assert manifest["future_result_root_reserved_not_dispatched"] == (
        "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1"
    )
    assert manifest["future_trigger_title_reserved_not_created"] == (
        "[modal-chm-v1-100m-stage-c-988-seed-977001-v1]"
    )
    guards = manifest["one_shot_guards_required"]
    assert all(guards.values())
    assert guards["gpu_entry_consumes_scientific_seed"] is True
    assert guards["automatic_retry_refused"] is True
    assert guards["resume_after_consumption_refused"] is True


def test_prep_module_contains_no_gpu_modal_or_subprocess_execution_path() -> None:
    source = inspect.getsource(prep)
    banned = (
        "import modal",
        "from modal",
        "torch.cuda",
        "subprocess",
        ".remote(",
        "@app.function",
    )
    for token in banned:
        assert token not in source
