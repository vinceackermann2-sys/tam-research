from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch

from tam_research.chm_v1_100m_stage_c_execution import gather_batch_from_source
from tam_research.chm_v2_100m_host_staged_preflight import (
    ENGINEERING_SEED,
    MEASURED_TOKENS_PER_MODEL,
    PASS_CLASSIFICATION,
    RESULT_ROOT,
    STOP_CLASSIFICATION,
    TRIGGER_TITLE,
    classify_systems_preflight,
    engineering_start_plan,
    host_staged_gather,
    protocol_manifest,
    start_plan_sha256,
    validate_contract,
)


def test_contract_is_systems_only_and_freezes_identity() -> None:
    manifest = validate_contract()
    assert manifest == protocol_manifest()
    assert manifest["control_issue"] == 1133
    assert manifest["engineering_seed"] == 1133201 == ENGINEERING_SEED
    assert manifest["engineering_seed_is_scientific"] is False
    assert manifest["consumed_scientific_seed"] == 2011121
    assert manifest["consumed_scientific_seed_reusable"] is False
    assert manifest["scientific_execution_authorized"] is False
    assert manifest["fresh_scientific_seed_authorized"] is False
    assert manifest["stage_d_authorized"] is False
    assert RESULT_ROOT.endswith("/issue-1133/attempt-1133201-v1")
    assert TRIGGER_TITLE == "[modal-chm-v2-100m-host-staged-l4-preflight-1133-v1]"


def test_engineering_start_plan_is_deterministic_and_shared_shape() -> None:
    a = engineering_start_plan(2_000_000)
    b = engineering_start_plan(2_000_000)
    assert torch.equal(a, b)
    assert tuple(a.shape) == (10, 4, 4)
    assert start_plan_sha256(a) == start_plan_sha256(b)
    assert int(a.min()) >= 0
    assert int(a.max()) + 1024 < 2_000_000
    assert MEASURED_TOKENS_PER_MODEL == 131_072


def test_host_staged_gather_matches_existing_gather_semantics_on_cpu() -> None:
    source = (np.arange(20_000, dtype=np.uint32) % 50_257).astype(np.uint16)
    starts = torch.tensor([0, 17, 2000, 12_345], dtype=torch.int64)
    x, y, transferred = host_staged_gather(source, starts, seq_len=1024, device="cpu")

    reference_source = torch.from_numpy(source.astype(np.int64))
    rx, ry = gather_batch_from_source(reference_source, starts, seq_len=1024)
    assert torch.equal(x, rx)
    assert torch.equal(y, ry)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert transferred == 4 * 1025 * np.dtype(np.int64).itemsize


def test_host_staged_gather_reads_only_requested_windows() -> None:
    class GuardedSource:
        def __init__(self) -> None:
            self.array = np.arange(5000, dtype=np.uint16)
            self.slices: list[slice] = []

        def __len__(self) -> int:
            return len(self.array)

        def __getitem__(self, key):
            assert isinstance(key, slice), "host gather must use bounded slices"
            assert key.start is not None and key.stop is not None
            assert key.stop - key.start == 17
            self.slices.append(key)
            return self.array[key]

    source = GuardedSource()
    starts = torch.tensor([3, 101], dtype=torch.int64)
    x, y, transferred = host_staged_gather(source, starts, seq_len=16, device="cpu")
    assert tuple(x.shape) == (2, 16)
    assert tuple(y.shape) == (2, 16)
    assert len(source.slices) == 2
    assert transferred == 2 * 17 * 8


def test_host_staged_gather_refuses_invalid_offsets() -> None:
    source = np.arange(100, dtype=np.uint16)
    with pytest.raises(ValueError):
        host_staged_gather(source, [], seq_len=8, device="cpu")
    with pytest.raises(ValueError):
        host_staged_gather(source, [-1], seq_len=8, device="cpu")
    with pytest.raises(ValueError):
        host_staged_gather(source, [95], seq_len=8, device="cpu")


def _row(*, params: int, tps: float, allocated: float = 15.0, reserved: float = 16.0):
    return {
        "trainable_parameters": params,
        "warmup_steps": 2,
        "measured_steps": 8,
        "measured_tokens": 131_072,
        "finite_loss": True,
        "finite_gradients": True,
        "finite_parameters": True,
        "host_staged_transport": True,
        "full_source_cuda_cache_present": False,
        "peak_allocated_gib": allocated,
        "peak_reserved_gib": reserved,
        "tokens_per_second": tps,
    }


def test_positive_systems_gate_requires_memory_margin_and_projection() -> None:
    raw = _row(params=101_836_800, tps=12_000.0)
    # Use the live architecture's exact VP parameter count from manifest.
    vp_params = protocol_manifest()["models"]["vp_eiem_parameters"]
    vp = _row(params=vp_params, tps=10_000.0)
    decision = classify_systems_preflight(
        raw=raw,
        vp=vp,
        live_hourly_resource_usd=1.1172,
    )
    assert decision["classification"] == PASS_CLASSIFICATION
    assert decision["passed"] is True
    assert decision["stop_reasons"] == []
    assert decision["projection"]["three_model_projected_seconds"] < 14_400
    assert decision["projection"]["three_model_projected_compute_usd"] < 6.0


def test_systems_gate_stops_on_memory_cache_or_throughput_regression() -> None:
    raw = _row(params=101_836_800, tps=12_000.0, allocated=18.01)
    raw["full_source_cuda_cache_present"] = True
    vp_params = protocol_manifest()["models"]["vp_eiem_parameters"]
    vp = _row(params=vp_params, tps=1_999.0, reserved=19.01)
    decision = classify_systems_preflight(
        raw=raw,
        vp=vp,
        live_hourly_resource_usd=1.1172,
    )
    assert decision["classification"] == STOP_CLASSIFICATION
    reasons = set(decision["stop_reasons"])
    assert "raw_eiem_peak_allocated_above_gate" in reasons
    assert "raw_eiem_full_source_cuda_cache_present" in reasons
    assert "vp_eiem_peak_reserved_above_gate" in reasons
    assert "vp_eiem_throughput_below_gate" in reasons


def test_helper_has_no_modal_or_scientific_execution_surface() -> None:
    import tam_research.chm_v2_100m_host_staged_preflight as module

    source = inspect.getsource(module)
    assert "import modal" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "torch.save(" not in source
    assert "_device_tokens(" not in source
    assert "scientific_execution_authorized" in source
    assert "fresh_scientific_seed_authorized" in source


def _python_heredocs(source: str) -> list[str]:
    import textwrap

    lines = source.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if "python - <<'PY'" not in lines[index]:
            index += 1
            continue
        index += 1
        block: list[str] = []
        while index < len(lines) and lines[index].strip() != "PY":
            block.append(lines[index])
            index += 1
        assert index < len(lines), "unterminated Python heredoc"
        blocks.append(textwrap.dedent("\n".join(block)))
        index += 1
    return blocks


def test_modal_harness_preserves_host_staged_one_shot_boundary() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "modal_chm_v2_100m_host_staged_preflight_1133_v1.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert "_device_tokens(" not in source
    assert "host_staged_gather(" in source
    assert "ENGINEERING_SEED = 1_133_201" in source
    assert "CONSUMED_SCIENTIFIC_SEED = 2_011_121" in source
    assert "gpu=GPU_CLASS" in source
    assert "timeout=MAX_SECONDS" in source
    assert "retries=0" in source
    assert "_atomic_write(consumed, attempt)" in source
    assert source.index("_atomic_write(consumed, attempt)") < source.index(
        "CHMV1100MEIEMLM"
    )
    assert "torch.save(" not in source
    assert "checkpoint_resume" not in source
    assert '"scientific_seed_created": False' in source
    assert '"fresh_scientific_seed_authorized": False' in source


def test_engineering_workflow_is_issue_only_exact_authority_and_no_manual_rerun() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / ".github" / "workflows" / "modal-chm-v2-100m-host-staged-l4-preflight-1133-v1.yml"
    source = path.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "issues:" in source and "types: [opened]" in source
    assert "[modal-chm-v2-100m-host-staged-l4-preflight-1133-v1]" in source
    assert "github.run_attempt" in source
    assert 'test "$RUN_ATTEMPT" = "1"' in source
    assert "CHM_V2_100M_HOST_STAGED_CORPUS_L4_FINAL_ENGINEERING_AUTHORITY_V1" in source
    assert "--phase inspect" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase state" in source
    assert "steps.reserve.outcome == 'success'" in source
    assert "engineering_seed=1133201" in source
    assert "fresh_scientific_seed_authorized=false" in source
    assert "scientific_rerun_authorized=false" in source
    assert "stage_d_authorized=false" in source
    assert "MODAL_TOKEN_ID=%s" not in source
    assert "MODAL_TOKEN_SECRET=%s" not in source


def test_authority_audit_is_read_only_and_never_reserves_or_launches() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / ".github"
        / "workflows"
        / "modal-chm-v2-100m-host-staged-l4-preflight-1133-authority-audit-v1.yml"
    )
    source = path.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in source
    assert "[modal-chm-v2-100m-host-staged-l4-preflight-1133-authority-audit-v1]" in source
    assert "modal_select_account_v3.py" in source
    assert "--phase inspect" in source
    assert "--phase reserve" not in source
    assert "--phase run" not in source
    assert "--phase state" not in source
    assert "result_namespace_unused=true" in source
    assert "engineering_attempt_consumed=false" in source
    assert "trigger_authorized=false" in source
    assert "fresh_scientific_seed_authorized=false" in source


def test_all_1133_embedded_workflow_python_compiles() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = (
        root / ".github" / "workflows" / "modal-chm-v2-100m-host-staged-l4-preflight-1133-v1.yml",
        root
        / ".github"
        / "workflows"
        / "modal-chm-v2-100m-host-staged-l4-preflight-1133-authority-audit-v1.yml",
    )
    for path in paths:
        blocks = _python_heredocs(path.read_text(encoding="utf-8"))
        assert blocks, path
        for block in blocks:
            ast.parse(block)
