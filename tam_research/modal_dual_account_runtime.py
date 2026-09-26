from __future__ import annotations

"""Runtime-admission-aware dual Modal account selection (#1067).

This is an additive/versioned wrapper around the frozen #1031 static selector.
It preserves the old policy while requiring a successful minimal remote CPU
function before an account is considered execution-eligible.
"""

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ACCOUNT_LABELS = ("primary", "secondary")
PRIMARY_TOKEN_ID_ENV = "MODAL_TOKEN_ID"
PRIMARY_TOKEN_SECRET_ENV = "MODAL_TOKEN_SECRET"
SECONDARY_TOKEN_ID_ENV = "MODAL_TOKEN_ID_SECONDARY"
SECONDARY_TOKEN_SECRET_ENV = "MODAL_TOKEN_SECRET_SECONDARY"

BASE_POLICY_BLOB = "b10c4a19d3c4114d084a721b1a84235dfef45259"
RUNTIME_PROBE_BLOB = "f758544147d365e1e16019a37edad2d558697b2b"
RUNTIME_PROBE_MARKER = "TAM_MODAL_RUNTIME_ADMISSION_PROBE="

_ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = Path(__file__).resolve().with_name("modal_dual_account.py")
_RUNTIME_PROBE_PATH = _ROOT / "scripts" / "modal_runtime_admission_probe.py"


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _require_blob(path: Path, expected: str, name: str) -> None:
    actual = _git_blob_sha(path.read_bytes())
    if actual != expected:
        raise RuntimeError(f"#1067 {name} blob drift: {actual} != {expected}")


_require_blob(_BASE_PATH, BASE_POLICY_BLOB, "base selector")
_require_blob(_RUNTIME_PROBE_PATH, RUNTIME_PROBE_BLOB, "runtime probe")

_spec = importlib.util.spec_from_file_location("_tam_modal_dual_account_frozen_1031", _BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load frozen base selector from {_BASE_PATH}")
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)


@dataclass(frozen=True)
class RuntimeAccountSnapshot:
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
    static_error: str | None = None
    runtime_error: str | None = None

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


AccountSelection = _base.AccountSelection
validate_frozen_account = _base.validate_frozen_account


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_seconds: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        env=dict(env),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _runtime_probe(
    *,
    env: Mapping[str, str],
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[bool, str | None]:
    result = command_runner(
        ["modal", "run", str(_RUNTIME_PROBE_PATH)],
        env=env,
        timeout_seconds=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "runtime admission probe failed").strip()
        detail = detail[-1000:]
        return False, detail

    matches = [
        line.split(RUNTIME_PROBE_MARKER, 1)[1]
        for line in result.stdout.splitlines()
        if RUNTIME_PROBE_MARKER in line
    ]
    if len(matches) != 1:
        return False, f"runtime admission probe marker count={len(matches)}"
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        return False, "runtime admission probe returned invalid JSON"

    ok = bool(
        payload.get("status") == "PASS"
        and payload.get("classification") == "TAM_MODAL_RUNTIME_ADMISSION_CPU_ONLY_PASS"
        and payload.get("volume_mount_visible") is True
        and payload.get("gpu_allocated") is False
        and payload.get("writes_performed") is False
    )
    return (True, None) if ok else (False, "runtime admission probe payload failed validation")


def probe_account_runtime(
    *,
    label: str,
    token_id: str | None,
    token_secret: str | None,
    required_volumes: Sequence[str] = (),
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
) -> RuntimeAccountSnapshot:
    static = _base.probe_account(
        label=label,
        token_id=token_id,
        token_secret=token_secret,
        required_volumes=required_volumes,
        command_runner=command_runner,
    )

    runtime_ok = False
    runtime_error: str | None = None
    if static.healthy:
        env = dict(os.environ)
        env[PRIMARY_TOKEN_ID_ENV] = "" if token_id is None else str(token_id)
        env[PRIMARY_TOKEN_SECRET_ENV] = "" if token_secret is None else str(token_secret)
        runtime_ok, runtime_error = _runtime_probe(env=env, command_runner=command_runner)
    else:
        runtime_error = "static account eligibility failed"

    return RuntimeAccountSnapshot(
        label=static.label,
        configured=static.configured,
        auth_ok=static.auth_ok,
        rates_ok=static.rates_ok,
        billing_ok=static.billing_ok,
        required_volumes_ok=static.required_volumes_ok,
        runtime_admission_ok=runtime_ok,
        workspace_name=static.workspace_name,
        required_volumes=static.required_volumes,
        present_required_volumes=static.present_required_volumes,
        metered_cost=static.metered_cost,
        billed_cost=static.billed_cost,
        adjustments=static.adjustments,
        static_error=static.error,
        runtime_error=runtime_error,
    )


def choose_account_runtime(
    snapshots: Sequence[RuntimeAccountSnapshot],
    *,
    force_account: str | None = None,
    low_credit_account: str | None = None,
) -> AccountSelection:
    # The frozen policy is intentionally reused by duck typing: it depends only
    # on label/healthy/workspace_name/public_evidence, all provided above.
    return _base.choose_account(
        snapshots,
        force_account=force_account,
        low_credit_account=low_credit_account,
    )


def select_from_environment(
    *,
    required_volumes: Sequence[str] = (),
    force_account: str | None = None,
    low_credit_account: str | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
    environ: Mapping[str, str] | None = None,
) -> AccountSelection:
    env = os.environ if environ is None else environ
    primary = probe_account_runtime(
        label="primary",
        token_id=env.get(PRIMARY_TOKEN_ID_ENV),
        token_secret=env.get(PRIMARY_TOKEN_SECRET_ENV),
        required_volumes=required_volumes,
        command_runner=command_runner,
    )
    secondary = probe_account_runtime(
        label="secondary",
        token_id=env.get(SECONDARY_TOKEN_ID_ENV),
        token_secret=env.get(SECONDARY_TOKEN_SECRET_ENV),
        required_volumes=required_volumes,
        command_runner=command_runner,
    )
    return choose_account_runtime(
        (primary, secondary),
        force_account=force_account if force_account is not None else env.get("MODAL_FORCE_ACCOUNT"),
        low_credit_account=(
            low_credit_account
            if low_credit_account is not None
            else env.get("MODAL_LOW_CREDIT_ACCOUNT")
        ),
    )
