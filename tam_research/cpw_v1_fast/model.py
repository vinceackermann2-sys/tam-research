from __future__ import annotations

import torch

from architectures.cortex_s.affine_scan_triton_candidate import (
    affine_scan_triton_candidate,
    triton_available,
)
from tam_research.cpw_v1.model import (
    CPWV1Config,
    CPWV1ResearchLM,
    RecurrentWorldPredictor,
)
from tam_research.models import parameter_count


class TritonScanWorldPredictor(RecurrentWorldPredictor):
    """Same learned WORLD recurrence with a CUDA Triton scan backend."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(self.candidate(x))
        keep = torch.sigmoid(self.keep(x))
        update = (1.0 - keep) * candidate
        state = affine_scan_triton_candidate(keep, update, None)
        self.last_state_norm = state.float().norm(dim=-1).mean().detach()
        return self.out(state)


@torch.no_grad()
def convert_cpw_v1_to_triton_scan(
    model: CPWV1ResearchLM,
) -> CPWV1ResearchLM:
    before_count = parameter_count(model)
    before = {name: p for name, p in model.named_parameters()}

    for block in model.blocks:
        old = block.mixer.world
        if not isinstance(old, RecurrentWorldPredictor):
            raise TypeError("expected frozen CPW-v1 RecurrentWorldPredictor")
        if isinstance(old, TritonScanWorldPredictor):
            raise TypeError("conversion may only be applied once")

        with torch.random.fork_rng(devices=[]):
            replacement = TritonScanWorldPredictor(
                old.candidate.in_features,
                old.candidate.out_features,
            )
        replacement.candidate = old.candidate
        replacement.keep = old.keep
        replacement.out = old.out
        block.mixer.world = replacement

    after_count = parameter_count(model)
    after = {name: p for name, p in model.named_parameters()}
    if before_count != after_count:
        raise RuntimeError("scan conversion changed parameter count")
    if before.keys() != after.keys():
        raise RuntimeError("scan conversion changed parameter names")
    for name in before:
        if before[name] is not after[name]:
            raise RuntimeError(f"scan conversion replaced parameter object: {name}")
    return model


class CPWV1FastResearchLM(CPWV1ResearchLM):
    def __init__(self, cfg: CPWV1Config = CPWV1Config()):
        super().__init__(cfg)
        convert_cpw_v1_to_triton_scan(self)


def fast_parameter_count() -> int:
    return parameter_count(CPWV1FastResearchLM())


def fast_backend_status() -> dict[str, object]:
    return {
        "cuda_triton_available": triton_available(),
        "scientific_equations_changed": False,
        "parameter_count_delta": 0,
        "source_kernel": "architectures.cortex_s.affine_scan_triton_candidate",
    }
