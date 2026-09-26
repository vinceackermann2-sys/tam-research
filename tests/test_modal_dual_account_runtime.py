from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tam_research.modal_dual_account_runtime import (
    RUNTIME_PROBE_MARKER,
    choose_account_runtime,
    probe_account_runtime,
    select_from_environment,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "modal_runtime_admission_probe.py"
CLI = ROOT / "scripts" / "modal_select_account_runtime.py"


def _static_success(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    if argv[:3] == ["modal", "token", "info"]:
        return subprocess.CompletedProcess(argv, 0, stdout="authenticated\n", stderr="")
    if argv[:3] == ["modal", "billing", "rates"]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"gpu_hour_cost_l4":"0.80","cpu_hour_cost":"0.05","mem_gib_hour_cost":"0.01"}',
            stderr="",
        )
    if argv[:3] == ["modal", "billing", "summary"]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"metered_cost":"1.00","billed_cost":"0.00","adjustments":{"credits":"-1.00"}}',
            stderr="",
        )
    if argv[:3] == ["modal", "volume", "list"]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='[{"name":"tam-research-data"}]',
            stderr="",
        )
    if argv[0].endswith("python") or argv[0].endswith("python3"):
        token = "secondary" if argv and False else "workspace-runtime\n"
        return subprocess.CompletedProcess(argv, 0, stdout=token, stderr="")
    return None


def _runtime_pass(argv: list[str]) -> subprocess.CompletedProcess[str]:
    payload = (
        RUNTIME_PROBE_MARKER
        + '{"classification":"TAM_MODAL_RUNTIME_ADMISSION_CPU_ONLY_PASS",'
        + '"gpu_allocated":false,"status":"PASS","volume_mount_visible":true,'
        + '"writes_performed":false}\n'
    )
    return subprocess.CompletedProcess(argv, 0, stdout=payload, stderr="")


def test_spend_limited_primary_falls_back_to_runtime_eligible_secondary() -> None:
    def fake_runner(argv, *, env, timeout_seconds=45):
        static = _static_success(list(argv))
        if static is not None:
            if argv[0].endswith("python") or argv[0].endswith("python3"):
                workspace = "workspace-primary" if env["MODAL_TOKEN_ID"] == "p-id" else "workspace-secondary"
                return subprocess.CompletedProcess(argv, 0, stdout=workspace + "\n", stderr="")
            return static
        if argv[:2] == ["modal", "run"]:
            if env["MODAL_TOKEN_ID"] == "p-id":
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout="",
                    stderr="Workspace primary has exceeded its spend limit",
                )
            return _runtime_pass(list(argv))
        raise AssertionError(argv)

    result = select_from_environment(
        required_volumes=("tam-research-data",),
        command_runner=fake_runner,
        environ={
            "MODAL_TOKEN_ID": "p-id",
            "MODAL_TOKEN_SECRET": "p-secret",
            "MODAL_TOKEN_ID_SECONDARY": "s-id",
            "MODAL_TOKEN_SECRET_SECONDARY": "s-secret",
        },
    )
    assert result.selected_account == "secondary"
    assert result.selected_workspace_name == "workspace-secondary"
    assert result.reason == "primary_unavailable"
    assert result.accounts["primary"]["runtime_admission_ok"] is False
    assert "spend limit" in result.accounts["primary"]["runtime_error"]
    assert result.accounts["secondary"]["runtime_admission_ok"] is True
    assert result.accounts["secondary"]["healthy"] is True


def test_spend_limited_primary_and_unconfigured_secondary_fail_before_execution() -> None:
    def fake_runner(argv, *, env, timeout_seconds=45):
        static = _static_success(list(argv))
        if static is not None:
            return static
        if argv[:2] == ["modal", "run"]:
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr="workspace has exceeded its spend limit",
            )
        raise AssertionError(argv)

    with pytest.raises(RuntimeError, match="no eligible Modal account"):
        select_from_environment(
            required_volumes=("tam-research-data",),
            command_runner=fake_runner,
            environ={
                "MODAL_TOKEN_ID": "p-id",
                "MODAL_TOKEN_SECRET": "p-secret",
                "MODAL_TOKEN_ID_SECONDARY": "",
                "MODAL_TOKEN_SECRET_SECONDARY": "",
            },
        )


def test_forced_runtime_ineligible_account_is_rejected() -> None:
    def fake_runner(argv, *, env, timeout_seconds=45):
        static = _static_success(list(argv))
        if static is not None:
            return static
        if argv[:2] == ["modal", "run"]:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="spend limit")
        raise AssertionError(argv)

    snapshot = probe_account_runtime(
        label="primary",
        token_id="id",
        token_secret="secret",
        required_volumes=("tam-research-data",),
        command_runner=fake_runner,
    )
    assert snapshot.runtime_admission_ok is False
    assert snapshot.healthy is False
    with pytest.raises(RuntimeError, match="not healthy"):
        choose_account_runtime(
            (
                snapshot,
                type(snapshot)(
                    label="secondary",
                    configured=False,
                    auth_ok=False,
                    rates_ok=False,
                    billing_ok=False,
                    required_volumes_ok=False,
                    runtime_admission_ok=False,
                    workspace_name=None,
                    required_volumes=("tam-research-data",),
                    present_required_volumes=(),
                    static_error="credentials not configured",
                    runtime_error="static account eligibility failed",
                ),
            ),
            force_account="primary",
        )


def test_runtime_probe_is_cpu_only_read_only_and_bounded() -> None:
    source = PROBE.read_text(encoding="utf-8")
    assert 'modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)' in source
    assert "cpu=0.125" in source
    assert "memory=128" in source
    assert "timeout=60" in source
    assert "retries=0" in source
    assert "gpu=" not in source
    assert ".gpu" not in source
    assert "volume.commit" not in source
    assert ".write_text(" not in source
    assert ".write_bytes(" not in source
    assert ".open(" not in source
    assert 'Path("/vol").is_dir()' in source
    assert '"writes_performed": False' in source


def test_runtime_selector_cli_uses_verified_direct_file_loader() -> None:
    source = CLI.read_text(encoding="utf-8")
    assert "spec_from_file_location" in source
    assert "modal_dual_account_runtime.py" in source
    assert "modal_runtime_admission_probe.py" in source
    assert "e3489db84794471bb3ef00f1938939e92d0183a9" in source
    assert "f758544147d365e1e16019a37edad2d558697b2b" in source
    assert "import tam_research" not in source
    assert "from tam_research" not in source
