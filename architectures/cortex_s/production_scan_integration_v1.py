from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
from torch import nn

from architectures.cortex_s.affine_scan_triton_candidate import (
    affine_scan_triton_candidate,
    triton_available,
)
from architectures.cortex_s.grouped_moe_v4 import build_memory_lean_grouped_cortex_100m
from architectures.cortex_s.language_model import (
    CortexSLM,
    PersistentWorldState,
    affine_scan,
    parameter_count,
)


SYSTEMS_VARIANT = "memory_lean_grouped_v4_triton_scan_v1"
CONSUMED_PREDECESSOR_ISSUE = 817
CONSUMED_PREDECESSOR_RUN = 34_450_068_844
CONSUMED_PREDECESSOR_JOB = 102_783_560_405
CONSUMED_PREDECESSOR_SEED = 2_026_090_907
ISOLATED_SCAN_SPEEDUP = 3.055218648823612


def _should_use_triton(a: torch.Tensor) -> bool:
    """Select Triton only for a real CUDA tensor when Triton is importable."""

    return bool(a.is_cuda and triton_available())


def production_affine_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Training-safe scan dispatcher for the production-integration candidate.

    CUDA+Triton uses the #817-validated custom-autograd candidate. CPU and any
    unsupported environment keep the existing production ``affine_scan`` exactly;
    they deliberately do not route through the candidate's sequential CPU oracle.
    """

    if _should_use_triton(a):
        return affine_scan_triton_candidate(a, b, initial)
    return affine_scan(a, b, initial)


class ProductionScanPersistentWorldState(PersistentWorldState):
    """PersistentWorldState with only the scan execution backend changed."""

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate = torch.tanh(self.candidate(x))
        keep = torch.sigmoid(self.keep(x))
        if state is None:
            state = torch.zeros(
                x.size(0),
                self.state_size,
                device=candidate.device,
                dtype=candidate.dtype,
            )
        else:
            state = state.to(device=candidate.device, dtype=candidate.dtype)
        states = production_affine_scan(keep, (1.0 - keep) * candidate, state)
        return self.out(states), states[:, -1], states


def _replacement_world(legacy: PersistentWorldState) -> ProductionScanPersistentWorldState:
    first_parameter = next(legacy.parameters())
    device = first_parameter.device
    dtype = first_parameter.dtype
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    # Construction is immediately discarded in favour of the existing submodules.
    # fork_rng prevents this systems-only conversion from perturbing global RNG.
    with torch.random.fork_rng(devices=cuda_devices):
        replacement = ProductionScanPersistentWorldState(
            d_model=legacy.candidate.in_features,
            state_size=legacy.state_size,
        ).to(device=device, dtype=dtype)

    replacement.candidate = legacy.candidate
    replacement.keep = legacy.keep
    replacement.out = legacy.out
    return replacement


@torch.no_grad()
def convert_world_state_to_triton_scan(model: CortexSLM) -> CortexSLM:
    """Replace only recurrent scan execution while preserving all parameters."""

    before_count = parameter_count(model)
    before_parameters = {name: parameter for name, parameter in model.named_parameters()}

    for block in model.blocks:
        legacy = block.world
        if not isinstance(legacy, PersistentWorldState):
            raise TypeError("expected PersistentWorldState before scan conversion")
        if isinstance(legacy, ProductionScanPersistentWorldState):
            raise TypeError("scan conversion is single-use per model")
        block.world = _replacement_world(legacy)

    after_count = parameter_count(model)
    after_parameters = {name: parameter for name, parameter in model.named_parameters()}
    if after_count != before_count:
        raise RuntimeError(
            f"scan conversion changed parameter count: {before_count} -> {after_count}"
        )
    if before_parameters.keys() != after_parameters.keys():
        raise RuntimeError("scan conversion changed parameter names")
    for name in before_parameters:
        if before_parameters[name] is not after_parameters[name]:
            raise RuntimeError(f"scan conversion replaced parameter object: {name}")
    return model


def build_memory_lean_grouped_triton_scan_cortex_100m() -> CortexSLM:
    """Build the exact grouped-v4 100M model plus the guarded scan seam."""

    from architectures.cortex_s.experiments.scale100m_2b.protocol import (
        EXPECTED_CORTEX_PARAMS,
    )

    model = build_memory_lean_grouped_cortex_100m()
    before = parameter_count(model)
    model = convert_world_state_to_triton_scan(model)
    after = parameter_count(model)
    if before != EXPECTED_CORTEX_PARAMS or after != EXPECTED_CORTEX_PARAMS:
        raise RuntimeError(
            "100M parameter count drift across scan integration: "
            f"{before} -> {after}, expected {EXPECTED_CORTEX_PARAMS}"
        )
    return model


@contextmanager
def memory_lean_grouped_triton_scan_training_builder() -> Iterator[None]:
    """Temporarily expose the integrated builder to an existing training harness.

    This context manager itself allocates no GPU and launches no training. A later
    preregistered preflight may use it to test the exact integrated production graph.
    """

    import architectures.cortex_s.experiments.scale100m_2b.train as training_module

    original = training_module.build_cortex_100m
    training_module.build_cortex_100m = build_memory_lean_grouped_triton_scan_cortex_100m
    try:
        yield
    finally:
        training_module.build_cortex_100m = original


def integration_status() -> dict[str, object]:
    return {
        "systems_variant": SYSTEMS_VARIANT,
        "classification": "ENGINEERING_PRODUCTION_INTEGRATION_ZERO_CREDIT_ONLY",
        "production_dispatcher_wired_on_converted_models": True,
        "default_language_model_unchanged": True,
        "cuda_backend": "affine_scan_triton_candidate",
        "cpu_fallback": "existing_production_affine_scan",
        "parameter_count_delta": 0,
        "consumed_predecessor_issue": CONSUMED_PREDECESSOR_ISSUE,
        "consumed_predecessor_run": CONSUMED_PREDECESSOR_RUN,
        "consumed_predecessor_job": CONSUMED_PREDECESSOR_JOB,
        "consumed_predecessor_seed": CONSUMED_PREDECESSOR_SEED,
        "isolated_scan_speedup": ISOLATED_SCAN_SPEEDUP,
        "gpu_dispatch_authorized": False,
        "production_preflight_authorized": False,
        "full_training_authorized": False,
        "scientific_claim_authorized": False,
    }
