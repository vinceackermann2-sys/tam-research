"""PGW Stage-0 #1346: isolate the workspace path from predictor carry.

CPU-only diagnostic: imports frozen modules, changes no scientific trainer,
architecture, seed, threshold, checkpoint, or workflow.
"""

from __future__ import annotations

import torch

from tam_research.pgw_v2.model import TokenConditionedPredictiveEventWorkspace
from tam_research.pgw_v2_mechanism.model import (
    PGWControlConfig,
    RoutingControlWorkspace,
)


def _workspace_and_no_workspace() -> tuple[
    TokenConditionedPredictiveEventWorkspace, RoutingControlWorkspace
]:
    torch.manual_seed(1346)
    cfg = PGWControlConfig(
        vocab_size=127,
        d_model=32,
        n_layers=1,
        n_heads=4,
        max_seq_len=16,
        local_attn_inner=24,
        chunk_size=8,
        workspace_layers=(0,),
        workspace_dim=16,
        workspace_slots=2,
        workspace_heads=4,
        predictor_rank=4,
        workspace_ff_rank=5,
        selected_events=2,
        control_mode="no_workspace",
    )

    # Both modules are frozen repository implementations. This is NOT a new
    # architecture, trained result, or scientifically powered benchmark.
    enabled = TokenConditionedPredictiveEventWorkspace(cfg).eval()
    torch.nn.init.normal_(enabled.workspace_init, mean=0.0, std=0.02)
    disabled = RoutingControlWorkspace(
        cfg, layer_index=4, control_mode="no_workspace"
    ).eval()
    disabled.load_state_dict(enabled.state_dict(), strict=True)

    # A test-only intervention removes the alternative cross-chunk predictor
    # carry route while preserving identical instantiated parameter tensors.
    with torch.no_grad():
        enabled.predict_up.weight.zero_()
        disabled.predict_up.weight.zero_()
    return enabled, disabled


def test_workspace_only_causal_reach_without_predictor_carry() -> None:
    enabled, disabled = _workspace_and_no_workspace()
    torch.manual_seed(1347)
    x = torch.randn(1, 16, 32)
    local = torch.zeros_like(x)

    changed_past = x.clone()
    changed_past[:, :8, :] = -3.0 * x[:, :8, :]

    with torch.no_grad():
        on_before = enabled(x, local)
        on_after = enabled(changed_past, local)
        off_before = disabled(x, local)
        off_after = disabled(changed_past, local)

    # Completed-chunk write cannot affect that same chunk's output.
    assert torch.equal(on_before[:, :8], on_after[:, :8])
    # The only possible forward information path is the workspace read/write.
    assert (on_before[:, 8:] - on_after[:, 8:]).abs().max().item() > 1e-7
    # Control removes the workspace; the predictor is zeroed in both arms.
    assert torch.equal(off_before, off_after)
    assert torch.count_nonzero(off_before).item() == 0


def test_workspace_only_future_objective_backpropagates_to_past() -> None:
    enabled, _ = _workspace_and_no_workspace()
    torch.manual_seed(1348)
    x = torch.randn(1, 16, 32, requires_grad=True)
    local = torch.zeros_like(x)

    # Loss uses ONLY the subsequent chunk. The causal predictor is disabled.
    subsequent_output = enabled(x, local)[:, 8:, :]
    loss = subsequent_output.square().sum()
    past_gradient = torch.autograd.grad(loss, x)[0][:, :8, :]

    assert torch.isfinite(loss)
    assert torch.isfinite(past_gradient).all()
    assert past_gradient.abs().sum().item() > 0.0


def test_future_chunk_does_not_modify_completed_chunk_output() -> None:
    enabled, _ = _workspace_and_no_workspace()
    torch.manual_seed(1349)
    x = torch.randn(1, 16, 32)
    changed_future = x.clone()
    changed_future[:, 8:, :] += 9.0
    local = torch.zeros_like(x)

    with torch.no_grad():
        before = enabled(x, local)
        after = enabled(changed_future, local)
    assert torch.equal(before[:, :8], after[:, :8])
