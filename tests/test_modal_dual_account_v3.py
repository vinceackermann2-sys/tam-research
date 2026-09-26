from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tam_research.modal_dual_account_v3 import (
    AccountSnapshot,
    choose_account,
    probe_account,
    resolve_secondary_credentials,
    select_from_environment,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "modal_runtime_admission_probe_1067_v1.py"


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
        error=error,
        credential_aliases=aliases,
    )


def _runner(runtime_by_token: dict[str, bool]):
    def fake(argv, *, env, timeout_seconds=45):
        token = env.get("MODAL_TOKEN_ID", "")
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
    ("env", "label"),
    (
        (
            {
                "MODAL_TOKEN_ID_SECONDARY": "secondary-id",
                "MODAL_TOKEN_SECRET_SECONDARY": "same-secret",
            },
            "secondary",
        ),
        (
            {
                "MODAL_TOKEN_ID_2": "secondary-id",
                "MODAL_TOKEN_SECRET_2": "same-secret",
            },
            "account_2",
        ),
        (
            {
                "MODAL_TOKEN_ID_B": "secondary-id",
                "MODAL_TOKEN_SECRET_B": "same-secret",
            },
            "legacy_b",
        ),
    ),
)
def test_each_supported_secondary_alias_resolves_without_exposing_secret(
    env: dict[str, str],
    label: str,
) -> None:
    resolved = resolve_secondary_credentials(env)
    assert resolved.configured is True
    assert resolved.token_id == "secondary-id"
    assert resolved.token_secret == "same-secret"
    assert resolved.alias_labels == (label,)


def test_multiple_identical_secondary_aliases_are_accepted_and_labels_recorded() -> None:
    env = {
        "MODAL_TOKEN_ID_SECONDARY": "secondary-id",
        "MODAL_TOKEN_SECRET_SECONDARY": "same-secret",
        "MODAL_TOKEN_ID_2": "secondary-id",
        "MODAL_TOKEN_SECRET_2": "same-secret",
        "MODAL_TOKEN_ID_B": "secondary-id",
        "MODAL_TOKEN_SECRET_B": "same-secret",
    }
    resolved = resolve_secondary_credentials(env)
    assert resolved.alias_labels == ("secondary", "account_2", "legacy_b")


def test_conflicting_secondary_aliases_fail_before_any_remote_probe() -> None:
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("remote probe must not run")

    env = {
        "MODAL_TOKEN_ID": "primary-id",
        "MODAL_TOKEN_SECRET": "primary-secret",
        "MODAL_TOKEN_ID_SECONDARY": "secondary-a",
        "MODAL_TOKEN_SECRET_SECONDARY": "secret-a",
        "MODAL_TOKEN_ID_2": "secondary-b",
        "MODAL_TOKEN_SECRET_2": "secret-b",
    }
    with pytest.raises(RuntimeError, match="aliases disagree"):
        select_from_environment(
            required_volumes=("tam-research-data",),
            command_runner=forbidden,
            environ=env,
        )
    assert calls == 0


def test_partial_secondary_alias_fails_before_any_remote_probe() -> None:
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("remote probe must not run")

    env = {
        "MODAL_TOKEN_ID": "primary-id",
        "MODAL_TOKEN_SECRET": "primary-secret",
        "MODAL_TOKEN_ID_2": "secondary-id",
    }
    with pytest.raises(RuntimeError, match="partial pair"):
        select_from_environment(
            required_volumes=("tam-research-data",),
            command_runner=forbidden,
            environ=env,
        )
    assert calls == 0


def test_primary_spend_limit_rejection_falls_back_to_established_account2_alias() -> None:
    runner = _runner({"primary-id": False, "secondary-id": True})
    selection = select_from_environment(
        required_volumes=("tam-research-data",),
        command_runner=runner,
        environ={
            "MODAL_TOKEN_ID": "primary-id",
            "MODAL_TOKEN_SECRET": "primary-secret",
            "MODAL_TOKEN_ID_2": "secondary-id",
            "MODAL_TOKEN_SECRET_2": "secondary-secret",
        },
    )
    assert selection.selected_account == "secondary"
    assert selection.reason == "primary_unavailable"
    assert selection.accounts["primary"]["runtime_admission_ok"] is False
    assert selection.accounts["secondary"]["runtime_admission_ok"] is True
    assert selection.accounts["secondary"]["credential_aliases"] == ("account_2",)
    public_text = repr(selection.public_evidence())
    assert "secondary-secret" not in public_text
    assert "primary-secret" not in public_text


def test_forced_runtime_unhealthy_account_is_rejected() -> None:
    primary = _snapshot("primary", runtime_ok=False, error="spend limit")
    secondary = _snapshot("secondary", runtime_ok=True, aliases=("account_2",))
    with pytest.raises(RuntimeError, match="not healthy"):
        choose_account((primary, secondary), force_account="primary")


def test_runtime_admission_is_required_for_health() -> None:
    snapshot = _snapshot("primary", runtime_ok=False)
    assert snapshot.configured is True
    assert snapshot.auth_ok is True
    assert snapshot.rates_ok is True
    assert snapshot.billing_ok is True
    assert snapshot.required_volumes_ok is True
    assert snapshot.healthy is False


def test_runtime_probe_is_cpu_only_retry_free_read_only_and_bounded() -> None:
    source = PROBE.read_text(encoding="utf-8")
    assert "gpu=" not in source
    assert "cpu=0.125" in source
    assert "memory=128" in source
    assert "timeout=60" in source
    assert "retries=0" in source
    assert "create_if_missing=False" in source
    assert "write_text(" not in source
    assert "open(" not in source
    assert "volume.commit" not in source
    assert '"writes_performed": False' in source
    assert '"gpu_allocated": False' in source
