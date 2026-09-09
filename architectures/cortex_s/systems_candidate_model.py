from __future__ import annotations

import torch

from .language_model import CortexSLM, TrulySparseMoE, parameter_count
from .systems_optimization import GroupedSparseMoECandidate


@torch.no_grad()
def convert_to_grouped_moe_candidate(model: CortexSLM) -> CortexSLM:
    """Replace only MoE execution modules with the systems candidate in-place.

    This is an engineering conversion seam, not the production architecture.  It
    copies every router/expert parameter exactly, preserves the global RNG state,
    preserves total trainable parameter count, and leaves recurrent state,
    attention, normalization, embeddings and the language head untouched.
    """

    before = parameter_count(model)
    for block in model.blocks:
        legacy = block.moe
        if not isinstance(legacy, TrulySparseMoE):
            raise TypeError("model contains a non-legacy MoE; refusing ambiguous conversion")
        first = legacy.experts[0][0]
        device = legacy.router.weight.device
        dtype = legacy.router.weight.dtype
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

        # Candidate initialization is immediately overwritten.  fork_rng prevents
        # this engineering conversion from perturbing any scientific/global RNG.
        with torch.random.fork_rng(devices=cuda_devices):
            candidate = GroupedSparseMoECandidate(
                d_model=first.in_features,
                num_experts=legacy.num_experts,
                top_k=legacy.top_k,
                hidden=first.out_features,
            ).to(device=device, dtype=dtype)
        candidate.copy_from_legacy(legacy)
        block.moe = candidate

    after = parameter_count(model)
    if after != before:
        raise RuntimeError(f"grouped-MoE conversion changed parameter count: {before} -> {after}")
    return model
