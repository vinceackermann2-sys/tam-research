from __future__ import annotations

import subprocess

import pytest

from tam_research.modal_dual_account import (
    AccountSnapshot,
    choose_account,
    probe_account,
    validate_frozen_account,
)


def _snapshot(
    label: str,
    *,
    healthy: bool = True,
    configured: bool = True,
    workspace: str | None = None,
    error: str | None = None,
) -> AccountSnapshot:
    return AccountSnapshot(
        label=label,
        configured=configured,
        auth_ok=healthy,
        rates_ok=healthy,
        billing_ok=healthy,
        required_volumes_ok=healthy,
        workspace_name=workspace or f"workspace-{label}",
        required_volumes=("tam-research-data",),
        present_required_volumes=("tam-research-data",) if healthy else (),
        metered_cost="1.25" if healthy else None,
        billed_cost="0.00" if healthy else None,
        adjustments={"plan_credit": "-1.25"} if healthy else None,
        error=error,
    )


def test_primary_is_default_when_both_accounts_are_healthy() -> None:
    result = choose_account((_snapshot("primary"), _snapshot("secondary")))
    assert result.selected_account == "primary"
    assert result.reason == "primary_healthy"


def test_low_credit_primary_selects_healthy_secondary() -> None:
    result = choose_account(
        (_snapshot("primary"), _snapshot("secondary")),
        low_credit_account="primary",
    )
    assert result.selected_account == "secondary"
    assert result.reason == "primary_marked_low_credit"
    assert result.low_credit_account == "primary"


def test_primary_failure_falls_back_to_secondary_before_reservation() -> None:
    result = choose_account(
        (
            _snapshot("primary", healthy=False, error="auth failed"),
            _snapshot("secondary"),
        )
    )
    assert result.selected_account == "secondary"
    assert result.reason == "primary_unavailable"


def test_force_account_is_fail_closed_and_cannot_override_low_credit_flag() -> None:
    result = choose_account(
        (_snapshot("primary"), _snapshot("secondary")),
        force_account="secondary",
    )
    assert result.selected_account == "secondary"
    assert result.reason == "explicit_force"

    with pytest.raises(RuntimeError, match="not healthy"):
        choose_account(
            (_snapshot("primary"), _snapshot("secondary", healthy=False)),
            force_account="secondary",
        )

    with pytest.raises(RuntimeError, match="low-credit"):
        choose_account(
            (_snapshot("primary"), _snapshot("secondary")),
            force_account="secondary",
            low_credit_account="secondary",
        )


def test_no_eligible_account_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="no eligible Modal account"):
        choose_account(
            (
                _snapshot("primary", healthy=False),
                _snapshot("secondary", healthy=False),
            )
        )
    with pytest.raises(RuntimeError, match="no eligible Modal account"):
        choose_account(
            (
                _snapshot("primary"),
                _snapshot("secondary", healthy=False),
            ),
            low_credit_account="primary",
        )


def test_account_switch_is_forbidden_after_durable_boundary() -> None:
    assert validate_frozen_account(
        selected_account="secondary",
        bound_account="primary",
        dispatch_reserved=False,
        attempt_consumed=False,
    ) == "secondary"

    with pytest.raises(RuntimeError, match="switch forbidden"):
        validate_frozen_account(
            selected_account="secondary",
            bound_account="primary",
            dispatch_reserved=True,
            attempt_consumed=False,
        )

    with pytest.raises(RuntimeError, match="switch forbidden"):
        validate_frozen_account(
            selected_account="secondary",
            bound_account="primary",
            dispatch_reserved=False,
            attempt_consumed=True,
        )

    assert validate_frozen_account(
        selected_account="primary",
        bound_account="primary",
        dispatch_reserved=True,
        attempt_consumed=True,
    ) == "primary"


def test_probe_account_records_sanitized_health_billing_and_volume_evidence() -> None:
    secret_id = "id-super-secret"
    secret_value = "secret-super-secret"

    def fake_runner(argv, *, env, timeout_seconds=45):
        assert env["MODAL_TOKEN_ID"] == secret_id
        assert env["MODAL_TOKEN_SECRET"] == secret_value
        if argv[:3] == ["modal", "token", "info"]:
            return subprocess.CompletedProcess(argv, 0, stdout="authenticated\n", stderr="")
        if argv[:3] == ["modal", "billing", "rates"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout='{"gpu_hour_cost_l4":"0.80","cpu_hour_cost":"0.05"}',
                stderr="",
            )
        if argv[:3] == ["modal", "billing", "summary"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout='{"metered_cost":"12.50","billed_cost":"0.00","adjustments":{"credits":"-12.50"}}',
                stderr="",
            )
        if argv[:3] == ["modal", "volume", "list"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout='[{"name":"tam-research-data"},{"name":"other"}]',
                stderr="",
            )
        if argv[0].endswith("python") or argv[0].endswith("python3"):
            return subprocess.CompletedProcess(argv, 0, stdout="workspace-two\n", stderr="")
        raise AssertionError(argv)

    snapshot = probe_account(
        label="secondary",
        token_id=secret_id,
        token_secret=secret_value,
        required_volumes=("tam-research-data",),
        command_runner=fake_runner,
    )
    evidence = snapshot.public_evidence()
    assert snapshot.healthy is True
    assert snapshot.workspace_name == "workspace-two"
    assert snapshot.metered_cost == "12.50"
    assert snapshot.billed_cost == "0.00"
    assert snapshot.present_required_volumes == ("tam-research-data",)
    serialized = repr(evidence)
    assert secret_id not in serialized
    assert secret_value not in serialized


def test_probe_account_marks_missing_required_volume_ineligible() -> None:
    def fake_runner(argv, *, env, timeout_seconds=45):
        if argv[:3] == ["modal", "token", "info"]:
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")
        if argv[:3] == ["modal", "billing", "rates"]:
            return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")
        if argv[:3] == ["modal", "billing", "summary"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout='{"metered_cost":"0","billed_cost":"0","adjustments":{}}', stderr=""
            )
        if argv[:3] == ["modal", "volume", "list"]:
            return subprocess.CompletedProcess(argv, 0, stdout='[{"name":"other"}]', stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="workspace-two\n", stderr="")

    snapshot = probe_account(
        label="secondary",
        token_id="id",
        token_secret="secret",
        required_volumes=("tam-research-data",),
        command_runner=fake_runner,
    )
    assert snapshot.healthy is False
    assert snapshot.auth_ok is True
    assert snapshot.billing_ok is True
    assert snapshot.required_volumes_ok is False
    assert snapshot.error == "required Modal volume missing"


def test_incomplete_or_absent_credentials_are_never_eligible() -> None:
    incomplete = probe_account(
        label="secondary",
        token_id="id",
        token_secret="",
    )
    absent = probe_account(
        label="secondary",
        token_id="",
        token_secret="",
    )
    assert incomplete.healthy is False
    assert incomplete.error == "incomplete credential pair"
    assert absent.healthy is False
    assert absent.error == "credentials not configured"
