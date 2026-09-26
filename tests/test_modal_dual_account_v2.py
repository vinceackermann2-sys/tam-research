from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tam_research.modal_dual_account_v2 import (
    AccountSnapshot,
    choose_account,
    probe_account,
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


def test_primary_spend_limit_rejection_falls_back_to_runtime_healthy_secondary() -> None:
    runner = _runner({"primary-id": False, "secondary-id": True})
    primary = probe_account(
        label="primary",
        token_id="primary-id",
        token_secret="p-secret",
        required_volumes=("tam-research-data",),
        command_runner=runner,
    )
    secondary = probe_account(
        label="secondary",
        token_id="secondary-id",
        token_secret="s-secret",
        required_volumes=("tam-research-data",),
        command_runner=runner,
    )
    assert primary.auth_ok is True
    assert primary.required_volumes_ok is True
    assert primary.runtime_admission_ok is False
    assert primary.healthy is False
    assert "runtime admission probe failed" in (primary.error or "")
    assert secondary.runtime_admission_ok is True
    assert secondary.healthy is True

    selected = choose_account((primary, secondary))
    assert selected.selected_account == "secondary"
    assert selected.reason == "primary_unavailable"
    assert selected.accounts["primary"]["runtime_admission_ok"] is False
    assert selected.accounts["secondary"]["runtime_admission_ok"] is True


def test_primary_runtime_rejection_and_unconfigured_secondary_has_no_eligible_account() -> None:
    runner = _runner({"primary-id": False})
    primary = probe_account(
        label="primary",
        token_id="primary-id",
        token_secret="p-secret",
        required_volumes=("tam-research-data",),
        command_runner=runner,
    )
    secondary = probe_account(
        label="secondary",
        token_id="",
        token_secret="",
        required_volumes=("tam-research-data",),
        command_runner=runner,
    )
    assert secondary.configured is False
    assert secondary.runtime_admission_ok is False
    with pytest.raises(RuntimeError, match="no eligible Modal account"):
        choose_account((primary, secondary))


def test_forced_runtime_unhealthy_account_is_rejected() -> None:
    primary = _snapshot("primary", runtime_ok=False, error="spend limit")
    secondary = _snapshot("secondary", runtime_ok=True)
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
    assert 'create_if_missing=False' in source
    assert "write_text(" not in source
    assert "open(" not in source
    assert "volume.commit" not in source
    assert '"writes_performed": False' in source
    assert '"gpu_allocated": False' in source
