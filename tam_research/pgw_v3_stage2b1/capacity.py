"""PGW-v3 Stage-2B1: no-optimizer, one-batch gradient-capacity DIAGNOSTIC.

One-shot backward probes are NOT training, active-parameter matching,
capacity-normalized evaluation, science seeds, or benchmark results.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tam_research.pgw_v3_stage0.model import ROUTE_MODES
from tam_research.pgw_v3_stage1b.oracle import make_example
from tam_research.pgw_v3_stage2a.model import (
    PGWV3Stage2A, PGWV3Stage2AConfig, batch_verified_examples,
    instantiated_parameter_count,
)
from tam_research.pgw_v3_stage2b0.auxiliary import predictor_auxiliary_loss


@dataclass(frozen=True)
class GradientCounts:
    grad_seen_elements: int
    nonzero_elements: int
    predictor_nonzero: int
    utility_nonzero: int
    other_workspace_nonzero: int
    nonworkspace_nonzero: int


@dataclass(frozen=True)
class ModeGradientReport:
    mode: str
    instantiated: int
    answer_ce: GradientCounts
    auxiliary: GradientCounts
    union_nonzero: int
    unchanged_after_backprop: bool


def _categorize(name: str) -> str:
    if name.startswith("workspace.predict_down.") or name.startswith("workspace.predict_up."):
        return "predictor"
    if name.startswith("workspace.utility."):
        return "utility"
    if name.startswith("workspace."):
        return "other_workspace"
    return "nonworkspace"


def _count_gradients(model: PGWV3Stage2A) -> tuple[GradientCounts, dict[str, torch.Tensor]]:
    seen = nonzero = 0
    groups = {
        "predictor": 0, "utility": 0,
        "other_workspace": 0, "nonworkspace": 0,
    }
    masks: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            masks[name] = torch.zeros_like(parameter, dtype=torch.bool)
        else:
            if not torch.isfinite(parameter.grad).all().item():
                raise ValueError("nonfinite structural probe gradient")
            mask = parameter.grad.detach().ne(0)
            masks[name] = mask
            seen += parameter.numel()
            k = int(mask.count_nonzero().item())
            nonzero += k
            groups[_categorize(name)] += k
    return GradientCounts(
        grad_seen_elements=seen,
        nonzero_elements=nonzero,
        predictor_nonzero=groups["predictor"],
        utility_nonzero=groups["utility"],
        other_workspace_nonzero=groups["other_workspace"],
        nonworkspace_nonzero=groups["nonworkspace"],
    ), masks


def gradient_capacity_preflight() -> tuple[ModeGradientReport, ...]:
    """Paired, shared-initialization CPU derivative probes, *no optimizer*.

    Does not choose an auxiliary weight lambda; runs answer CE and auxiliary
    separately. Counts nonzero grads for ONE deterministic fixture batch only.
    """
    present = make_example(
        split="validation", index=0, last_position=5,
        overwrite_count=2, interference=True, delay_chunks=1, missing=False,
    )
    absent = make_example(
        split="validation", index=0, last_position=5,
        overwrite_count=2, interference=True, delay_chunks=1, missing=True,
    )
    tokens, anchors, targets = batch_verified_examples([present, absent])
    if tokens.device.type != "cpu":
        raise ValueError("CPU-only structural probe")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1394)  # CPU test fixture init only; NOT a science seed.
        reference = PGWV3Stage2A(PGWV3Stage2AConfig(route_mode="hybrid")).cpu().eval()
        reference_state = {
            k: v.detach().clone() for k, v in reference.state_dict().items()
        }
        reports: list[ModeGradientReport] = []
        for mode in sorted(ROUTE_MODES):
            model = PGWV3Stage2A(PGWV3Stage2AConfig(route_mode=mode)).cpu().eval()
            model.load_state_dict(reference_state, strict=True)
            model.zero_grad(set_to_none=True)

            # Answer-only probe: predictor surprise is detached in Stage-0.
            ce = F.cross_entropy(model(tokens, anchors), targets)
            if not torch.isfinite(ce).item():
                raise ValueError("nonfinite answer-only structural probe")
            ce.backward()
            ce_counts, ce_masks = _count_gradients(model)

            # Separate auxiliary-only probe: no lambda, no gradient update.
            model.zero_grad(set_to_none=True)
            aux = predictor_auxiliary_loss(model, tokens, anchors)
            if not torch.isfinite(aux).item():
                raise ValueError("nonfinite auxiliary-only structural probe")
            aux.backward()
            aux_counts, aux_masks = _count_gradients(model)

            union_nonzero = sum(
                int((ce_masks[name] | aux_masks[name]).count_nonzero().item())
                for name, _ in model.named_parameters()
            )
            unchanged = all(
                torch.equal(value.detach(), reference_state[name])
                for name, value in model.state_dict().items()
            )
            reports.append(
                ModeGradientReport(
                    mode=mode,
                    instantiated=instantiated_parameter_count(model),
                    answer_ce=ce_counts,
                    auxiliary=aux_counts,
                    union_nonzero=union_nonzero,
                    unchanged_after_backprop=unchanged,
                )
            )
        return tuple(reports)
