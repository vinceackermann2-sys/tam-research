from __future__ import annotations

import subprocess

import pytest

from tam_research.modal_dual_account_v3 import (
    AccountSnapshot,
    choose_account,
    resolve_secondary_credentials,
    select_from_environment,
)


def _snapshot(
    label: str,
    *,
    runtime_ok: bool,
    configured: bool = True,
    static_ok: bool = True,
    error: str | None = None,
    aliases: tuple[str, ...] = (),
) -> AccountSnapshot:
    return AccountSnapshot(
        label=label,
        configured=configured,
        auth_ok=static_ok,
        rates_ok=static_ok,
        billing_ok=static_ok,
        required_volumes_ok=static_ok,
        runtime_admission_ok=runtime_ok,
        workspace_name=f"workspace-{label}" if configured else None,
        required_volumes=("tam-research-data",),
        present_required_volumes=("tam-research-data",) if static_ok else (),
        credential_aliases=aliases,
        error=error,
    )


def _runner(runtime_by_token: dict[str, bool], calls: list[tuple[list[str], str]]):
    def fake(argv, *, env, timeout_seconds=45):
        token = env.get("MODAL_TOKEN_ID", "")
        calls.append((list(argv), token))
        if argv[:3] == ["modal", "token", "info"]:
            return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")
        if argv[:3] == ["modal", "billing", "rates"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout='{"gpu_hour_cost_l4":"0.8","cpu_hour_cost":"0.05","mem_gib_hour_cost":"0.01"}',
                stderr="",
            )
        if argv[:3] == ["modal", "billing", "summary"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout='{"metered_cost":"0","billed_cost":"0","adjustments":{}}',
                stderr="",
            )
        if argv[:3] == ["modal", "volume", "list"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout='[{"name":"tam-research-data"}]', stderr=""
            )
        if argv[0].endswith("python") or argv[0].endswith("python3"):
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"workspace-{token}\n", stderr=""
            )
        if argv[:2] == ["modal", "run"]:
            ok = runtime_by_token[token]
            if not ok:
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout="",
                    stderr="Workspace exceeded its spend limit",
                )
            payload = (
                'MODAL_RUNTIME_ADMISSION={"gpu_allocated": false, '
                '"runtime_admission_ok": true, "volume_name": "tam-research-data", '
                '"writes_performed": false}\n'
            )
            return subprocess.CompletedProcess(argv, 0, stdout=payload, stderr="")
        raise AssertionError(argv)

    return fake


@pytest.mark.parametrize(
    ("id_name", "secret_name", "expected_alias"),
    [
        ("MODAL_TOKEN_ID_SECONDARY", "MODAL_TOKEN_SECRET_SECONDARY", "secondary"),
        ("MODAL_TOKEN_ID_2", "MODAL_TOKEN_SECRET_2", "account2"),
        ("MODAL_TOKEN_ID_B", "MODAL_TOKEN_SECRET_B", "legacy_b"),
    ],
)
def test_each_secondary_alias_is_supported(
    id_name: str, secret_name: str, expected_alias: str
) -> None:
    env = {id_name: "secondary-id", secret_name: "secondary-secret"}
    token_id, token_secret, aliases = resolve_secondary_credentials(env)
    assert token_id == "secondary-id"
    assert token_secret == "secondary-secret"
    assert aliases == (expected_alias,)


def test_multiple_identical_secondary_aliases_are_accepted_and_labels_only_recorded() -> None:
    env = {
        "MODAL_TOKEN_ID_SECONDARY": "same-id",
        "MODAL_TOKEN_SECRET_SECONDARY": "same-secret",
        "MODAL_TOKEN_ID_2": "same-id",
        "MODAL_TOKEN_SECRET_2": "same-secret",
        "MODAL_TOKEN_ID_B": "same-id",
        "MODAL_TOKEN_SECRET_B": "same-secret",
    }
    token_id, token_secret, aliases = resolve_secondary_credentials(env)
    assert token_id == "same-id"
    assert token_secret == "same-secret"
    assert aliases == ("secondary", "account2", "legacy_b")
    assert "same-id" not in repr(aliases)
    assert "same-secret" not in repr(aliases)


def test_conflicting_complete_aliases_fail_closed_without_secret_values() -> None:
    env = {
        "MODAL_TOKEN_ID_2": "id-a",
        "MODAL_TOKEN_SECRET_2": "secret-a",
        "MODAL_TOKEN_ID_B": "id-b",
        "MODAL_TOKEN_SECRET_B": "secret-b",
    }
    with pytest.raises(RuntimeError, match="conflicting complete secondary") as exc:
        resolve_secondary_credentials(env)
    message = str(exc.value)
    assert "id-a" not in message
    assert "id-b" not in message
    assert "secret-a" not in message
    assert "secret-b" not in message


@pytest.mark.parametrize(
    "partial",
    [
        {"MODAL_TOKEN_ID_2": "id-only"},
        {"MODAL_TOKEN_SECRET_2": "secret-only"},
        {"MODAL_TOKEN_ID_B": "id-only"},
        {"MODAL_TOKEN_SECRET_B": "secret-only"},
        {"MODAL_TOKEN_ID_SECONDARY": "id-only"},
        {"MODAL_TOKEN_SECRET_SECONDARY": "secret-only"},
    ],
)
def test_partial_alias_fails_closed_before_any_modal_command(partial: dict[str, str]) -> None:
    calls: list[tuple[list[str], str]] = []
    runner = _runner({"primary-id": True}, calls)
    env = {
        "MODAL_TOKEN_ID": "primary-id",
        "MODAL_TOKEN_SECRET": "primary-secret",
        **partial,
    }
    with pytest.raises(RuntimeError, match="incomplete pair"):
        select_from_environment(
            required_volumes=("tam-research-data",),
            environ=env,
            command_runner=runner,
        )
    assert calls == []



def test_conflicting_aliases_fail_closed_before_configured_primary_is_probed() -> None:
    calls: list[tuple[list[str], str]] = []
    runner = _runner({"primary-id": True}, calls)
    env = {
        "MODAL_TOKEN_ID": "primary-id",
        "MODAL_TOKEN_SECRET": "primary-secret",
        "MODAL_TOKEN_ID_2": "secondary-a",
        "MODAL_TOKEN_SECRET_2": "secret-a",
        "MODAL_TOKEN_ID_B": "secondary-b",
        "MODAL_TOKEN_SECRET_B": "secret-b",
    }
    with pytest.raises(RuntimeError, match="conflicting complete secondary"):
        select_from_environment(
            required_volumes=("tam-research-data",),
            environ=env,
            command_runner=runner,
        )
    assert calls == []


def test_primary_runtime_rejection_falls_back_to_established_account2_alias() -> None:
    calls: list[tuple[list[str], str]] = []
    runner = _runner({"primary-id": False, "secondary-id": True}, calls)
    env = {
        "MODAL_TOKEN_ID": "primary-id",
        "MODAL_TOKEN_SECRET": "primary-secret",
        "MODAL_TOKEN_ID_2": "secondary-id",
        "MODAL_TOKEN_SECRET_2": "secondary-secret",
    }
    selected = select_from_environment(
        required_volumes=("tam-research-data",),
        environ=env,
        command_runner=runner,
    )
    assert selected.selected_account == "secondary"
    assert selected.reason == "primary_unavailable"
    assert selected.accounts["primary"]["runtime_admission_ok"] is False
    assert selected.accounts["secondary"]["runtime_admission_ok"] is True
    assert selected.accounts["secondary"]["credential_aliases"] == ("account2",)


def test_primary_runtime_rejection_falls_back_to_legacy_b_alias() -> None:
    calls: list[tuple[list[str], str]] = []
    runner = _runner({"primary-id": False, "secondary-id": True}, calls)
    env = {
        "MODAL_TOKEN_ID": "primary-id",
        "MODAL_TOKEN_SECRET": "primary-secret",
        "MODAL_TOKEN_ID_B": "secondary-id",
        "MODAL_TOKEN_SECRET_B": "secondary-secret",
    }
    selected = select_from_environment(
        required_volumes=("tam-research-data",),
        environ=env,
        command_runner=runner,
    )
    assert selected.selected_account == "secondary"
    assert selected.accounts["secondary"]["credential_aliases"] == ("legacy_b",)


def test_alias_labels_are_public_evidence_but_credentials_are_not() -> None:
    secondary = _snapshot(
        "secondary",
        runtime_ok=True,
        aliases=("account2",),
    )
    primary = _snapshot("primary", runtime_ok=False, error="spend limit")
    selected = choose_account((primary, secondary))
    evidence = selected.public_evidence()
    assert evidence["accounts"]["secondary"]["credential_aliases"] == ("account2",)
    rendered = repr(evidence)
    assert "MODAL_TOKEN_SECRET_2" not in rendered
    assert "secondary-secret" not in rendered
