from __future__ import annotations

"""Versioned dual-Modal-account selector with account-2 alias resolution (#1076).

Builds on v2 runtime-admission health and additionally resolves the secondary\naccount from canonical SECONDARY, established account-2 (_2), or legacy (_B)\ncredential aliases. Partial/conflicting aliases fail closed before any remote\nprobe. Secret token values never appear in returned evidence.
"""

from dataclasses import asdict, dataclass, replace
import json
import os
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

ACCOUNT_LABELS = ("primary", "secondary")
PRIMARY_TOKEN_ID_ENV = "MODAL_TOKEN_ID"
PRIMARY_TOKEN_SECRET_ENV = "MODAL_TOKEN_SECRET"
SECONDARY_CREDENTIAL_ALIASES = (
    ("secondary", "MODAL_TOKEN_ID_SECONDARY", "MODAL_TOKEN_SECRET_SECONDARY"),
    ("account_2", "MODAL_TOKEN_ID_2", "MODAL_TOKEN_SECRET_2"),
    ("legacy_b", "MODAL_TOKEN_ID_B", "MODAL_TOKEN_SECRET_B"),
)
RUNTIME_ADMISSION_PROBE = "modal_runtime_admission_probe_1067_v1.py"


@dataclass(frozen=True)
class AccountSnapshot:
    label: str
    configured: bool
    auth_ok: bool
    rates_ok: bool
    billing_ok: bool
    required_volumes_ok: bool
    runtime_admission_ok: bool
    workspace_name: str | None
    required_volumes: tuple[str, ...]
    present_required_volumes: tuple[str, ...]
    metered_cost: str | None = None
    billed_cost: str | None = None
    adjustments: Any = None
    error: str | None = None
    credential_aliases: tuple[str, ...] = ()

    @property
    def healthy(self) -> bool:
        return bool(
            self.configured
            and self.auth_ok
            and self.rates_ok
            and self.billing_ok
            and self.required_volumes_ok
            and self.runtime_admission_ok
        )

    def public_evidence(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["healthy"] = self.healthy
        return payload


@dataclass(frozen=True)
class AccountSelection:
    selected_account: str
    selected_workspace_name: str | None
    force_account: str | None
    low_credit_account: str | None
    reason: str
    accounts: dict[str, dict[str, Any]]

    def public_evidence(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SecondaryCredentialResolution:
    token_id: str | None
    token_secret: str | None
    alias_labels: tuple[str, ...]

    @property
    def configured(self) -> bool:
        return bool(self.token_id and self.token_secret)


def resolve_secondary_credentials(
    environ: Mapping[str, str],
) -> SecondaryCredentialResolution:
    """Resolve account-2 credentials without exposing secret values.

    Any partial alias pair fails closed. Multiple complete aliases are allowed
    only when every ID+secret pair is byte-identical.
    """
    complete: list[tuple[str, str, str]] = []
    for label, id_env, secret_env in SECONDARY_CREDENTIAL_ALIASES:
        token_id = environ.get(id_env)
        token_secret = environ.get(secret_env)
        id_present = bool(token_id)
        secret_present = bool(token_secret)
        if id_present != secret_present:
            raise RuntimeError(
                f"secondary Modal credential alias {label!r} is a partial pair"
            )
        if id_present:
            assert token_id is not None and token_secret is not None
            complete.append((label, token_id, token_secret))

    if not complete:
        return SecondaryCredentialResolution(None, None, ())

    first_id = complete[0][1]
    first_secret = complete[0][2]
    if any(
        token_id != first_id or token_secret != first_secret
        for _, token_id, token_secret in complete
    ):
        labels = tuple(label for label, _, _ in complete)
        raise RuntimeError(
            "secondary Modal credential aliases disagree; fail closed: "
            f"{labels}"
        )

    return SecondaryCredentialResolution(
        token_id=first_id,
        token_secret=first_secret,
        alias_labels=tuple(label for label, _, _ in complete),
    )


def _normalize_account_hint(
    value: str | None,
    *,
    name: str,
    allow_none_word: bool,
) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().lower()
    if allow_none_word and normalized == "none":
        return None
    if normalized not in ACCOUNT_LABELS:
        suffix = ", or none" if allow_none_word else ""
        raise ValueError(f"{name} must be primary, secondary{suffix}")
    return normalized


def choose_account(
    snapshots: Sequence[AccountSnapshot],
    *,
    force_account: str | None = None,
    low_credit_account: str | None = None,
) -> AccountSelection:
    by_label = {snapshot.label: snapshot for snapshot in snapshots}
    if set(by_label) != set(ACCOUNT_LABELS):
        raise ValueError("exactly one primary and one secondary snapshot are required")

    force = _normalize_account_hint(
        force_account, name="force_account", allow_none_word=False
    )
    low = _normalize_account_hint(
        low_credit_account, name="low_credit_account", allow_none_word=True
    )

    if force is not None:
        selected = by_label[force]
        if not selected.healthy:
            raise RuntimeError(f"forced Modal account {force!r} is not healthy")
        if low == force:
            raise RuntimeError(
                f"forced Modal account {force!r} is explicitly marked low-credit"
            )
        return AccountSelection(
            selected_account=force,
            selected_workspace_name=selected.workspace_name,
            force_account=force,
            low_credit_account=low,
            reason="explicit_force",
            accounts={
                label: by_label[label].public_evidence() for label in ACCOUNT_LABELS
            },
        )

    candidates = [
        by_label[label]
        for label in ACCOUNT_LABELS
        if label != low and by_label[label].healthy
    ]
    if not candidates:
        detail = {
            label: {
                "healthy": by_label[label].healthy,
                "configured": by_label[label].configured,
                "runtime_admission_ok": by_label[label].runtime_admission_ok,
                "error": by_label[label].error,
            }
            for label in ACCOUNT_LABELS
        }
        raise RuntimeError(f"no eligible Modal account remains: {detail}")

    selected = candidates[0]
    reason = (
        "primary_healthy"
        if selected.label == "primary"
        else ("primary_marked_low_credit" if low == "primary" else "primary_unavailable")
    )
    return AccountSelection(
        selected_account=selected.label,
        selected_workspace_name=selected.workspace_name,
        force_account=None,
        low_credit_account=low,
        reason=reason,
        accounts={label: by_label[label].public_evidence() for label in ACCOUNT_LABELS},
    )


def validate_frozen_account(
    *,
    selected_account: str,
    bound_account: str,
    dispatch_reserved: bool,
    attempt_consumed: bool,
) -> str:
    selected = _normalize_account_hint(
        selected_account, name="selected_account", allow_none_word=False
    )
    bound = _normalize_account_hint(
        bound_account, name="bound_account", allow_none_word=False
    )
    if (dispatch_reserved or attempt_consumed) and selected != bound:
        raise RuntimeError(
            f"Modal account switch forbidden after durable boundary: {bound!r} -> {selected!r}"
        )
    return selected


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_seconds: int = 45,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        env=dict(env),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _json_output(result: subprocess.CompletedProcess[str], name: str) -> Any:
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} did not return JSON") from exc


def _volume_names(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        for key in ("volumes", "items", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise RuntimeError("modal volume list --json returned an unsupported shape")
    names: set[str] = set()
    for item in payload:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict):
            for key in ("name", "Name"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    names.add(value)
                    break
    return names


def _runtime_admission_payload(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if result.returncode != 0:
        raise RuntimeError(
            f"Modal runtime admission probe failed with exit code {result.returncode}"
        )
    prefix = "MODAL_RUNTIME_ADMISSION="
    matches = [
        line.split(prefix, 1)[1]
        for line in result.stdout.splitlines()
        if prefix in line
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Modal runtime admission probe returned {len(matches)} marker lines"
        )
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Modal runtime admission probe marker was not JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Modal runtime admission probe returned non-object evidence")
    if payload.get("runtime_admission_ok") is not True:
        raise RuntimeError("Modal runtime admission probe did not report success")
    if payload.get("gpu_allocated") is not False:
        raise RuntimeError("Modal runtime admission probe unexpectedly reports GPU use")
    if payload.get("writes_performed") is not False:
        raise RuntimeError("Modal runtime admission probe unexpectedly reports writes")
    return payload


def _base_snapshot(
    *,
    label: str,
    configured: bool,
    auth_ok: bool,
    rates_ok: bool,
    billing_ok: bool,
    required_volumes_ok: bool,
    runtime_admission_ok: bool,
    workspace_name: str | None,
    required: tuple[str, ...],
    present: tuple[str, ...] = (),
    metered_cost: str | None = None,
    billed_cost: str | None = None,
    adjustments: Any = None,
    error: str | None = None,
    credential_aliases: tuple[str, ...] = (),
) -> AccountSnapshot:
    return AccountSnapshot(
        label=label,
        configured=configured,
        auth_ok=auth_ok,
        rates_ok=rates_ok,
        billing_ok=billing_ok,
        required_volumes_ok=required_volumes_ok,
        runtime_admission_ok=runtime_admission_ok,
        workspace_name=workspace_name,
        required_volumes=required,
        present_required_volumes=present,
        metered_cost=metered_cost,
        billed_cost=billed_cost,
        adjustments=adjustments,
        error=error,
        credential_aliases=credential_aliases,
    )


def probe_account(
    *,
    label: str,
    token_id: str | None,
    token_secret: str | None,
    required_volumes: Sequence[str] = (),
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
    runtime_probe_path: str = RUNTIME_ADMISSION_PROBE,
) -> AccountSnapshot:
    if label not in ACCOUNT_LABELS:
        raise ValueError(f"unknown Modal account label {label!r}")
    token_id = "" if token_id is None else str(token_id)
    token_secret = "" if token_secret is None else str(token_secret)
    required = tuple(sorted({str(name) for name in required_volumes if str(name)}))

    if bool(token_id) != bool(token_secret):
        return _base_snapshot(
            label=label,
            configured=False,
            auth_ok=False,
            rates_ok=False,
            billing_ok=False,
            required_volumes_ok=False,
            runtime_admission_ok=False,
            workspace_name=None,
            required=required,
            error="incomplete credential pair",
        )
    if not token_id:
        return _base_snapshot(
            label=label,
            configured=False,
            auth_ok=False,
            rates_ok=False,
            billing_ok=False,
            required_volumes_ok=False,
            runtime_admission_ok=False,
            workspace_name=None,
            required=required,
            error="credentials not configured",
        )

    env = dict(os.environ)
    env["MODAL_TOKEN_ID"] = token_id
    env["MODAL_TOKEN_SECRET"] = token_secret

    try:
        auth = command_runner(["modal", "token", "info"], env=env)
        if auth.returncode != 0:
            raise RuntimeError(
                f"modal token info failed with exit code {auth.returncode}"
            )

        workspace = command_runner(
            [
                sys.executable,
                "-c",
                "import modal; print(modal.Workspace.from_context().name)",
            ],
            env=env,
        )
        if workspace.returncode != 0 or not workspace.stdout.strip():
            raise RuntimeError(
                f"Modal workspace lookup failed with exit code {workspace.returncode}"
            )
        workspace_name = workspace.stdout.strip()

        rates = command_runner(["modal", "billing", "rates", "--json"], env=env)
        _json_output(rates, "modal billing rates")

        billing_result = command_runner(
            ["modal", "billing", "summary", "--for", "this month", "--json"],
            env=env,
        )
        billing = _json_output(billing_result, "modal billing summary")
        if not isinstance(billing, dict):
            raise RuntimeError("modal billing summary returned a non-object")
        metered_cost = billing.get("metered_cost")
        billed_cost = billing.get("billed_cost")
        adjustments = billing.get("adjustments")

        volumes_result = command_runner(
            ["modal", "volume", "list", "--json"], env=env
        )
        names = _volume_names(_json_output(volumes_result, "modal volume list"))
        present = tuple(name for name in required if name in names)
        required_ok = len(present) == len(required)
        if not required_ok:
            return _base_snapshot(
                label=label,
                configured=True,
                auth_ok=True,
                rates_ok=True,
                billing_ok=True,
                required_volumes_ok=False,
                runtime_admission_ok=False,
                workspace_name=workspace_name,
                required=required,
                present=present,
                metered_cost=None if metered_cost is None else str(metered_cost),
                billed_cost=None if billed_cost is None else str(billed_cost),
                adjustments=adjustments,
                error="required Modal volume missing",
            )

        runtime = command_runner(
            ["modal", "run", runtime_probe_path],
            env=env,
            timeout_seconds=75,
        )
        try:
            _runtime_admission_payload(runtime)
        except RuntimeError as exc:
            return _base_snapshot(
                label=label,
                configured=True,
                auth_ok=True,
                rates_ok=True,
                billing_ok=True,
                required_volumes_ok=True,
                runtime_admission_ok=False,
                workspace_name=workspace_name,
                required=required,
                present=present,
                metered_cost=None if metered_cost is None else str(metered_cost),
                billed_cost=None if billed_cost is None else str(billed_cost),
                adjustments=adjustments,
                error=str(exc),
            )

        return _base_snapshot(
            label=label,
            configured=True,
            auth_ok=True,
            rates_ok=True,
            billing_ok=True,
            required_volumes_ok=True,
            runtime_admission_ok=True,
            workspace_name=workspace_name,
            required=required,
            present=present,
            metered_cost=None if metered_cost is None else str(metered_cost),
            billed_cost=None if billed_cost is None else str(billed_cost),
            adjustments=adjustments,
            error=None,
        )
    except (RuntimeError, subprocess.SubprocessError) as exc:
        return _base_snapshot(
            label=label,
            configured=True,
            auth_ok=False,
            rates_ok=False,
            billing_ok=False,
            required_volumes_ok=False,
            runtime_admission_ok=False,
            workspace_name=None,
            required=required,
            error=str(exc),
        )


def select_from_environment(
    *,
    required_volumes: Sequence[str] = (),
    force_account: str | None = None,
    low_credit_account: str | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
    environ: Mapping[str, str] | None = None,
    runtime_probe_path: str = RUNTIME_ADMISSION_PROBE,
) -> AccountSelection:
    env = os.environ if environ is None else environ
    primary = probe_account(
        label="primary",
        token_id=env.get(PRIMARY_TOKEN_ID_ENV),
        token_secret=env.get(PRIMARY_TOKEN_SECRET_ENV),
        required_volumes=required_volumes,
        command_runner=command_runner,
        runtime_probe_path=runtime_probe_path,
    )
    secondary_credentials = resolve_secondary_credentials(env)
    secondary = probe_account(
        label="secondary",
        token_id=secondary_credentials.token_id,
        token_secret=secondary_credentials.token_secret,
        required_volumes=required_volumes,
        command_runner=command_runner,
        runtime_probe_path=runtime_probe_path,
    )
    secondary = replace(
        secondary,
        credential_aliases=secondary_credentials.alias_labels,
    )
    return choose_account(
        (primary, secondary),
        force_account=(
            force_account if force_account is not None else env.get("MODAL_FORCE_ACCOUNT")
        ),
        low_credit_account=(
            low_credit_account
            if low_credit_account is not None
            else env.get("MODAL_LOW_CREDIT_ACCOUNT")
        ),
    )
