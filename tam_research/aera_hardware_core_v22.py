from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState, FastMemoryState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v18 import PretrainableDeltaFastMemory
from .aera_hardware_core_v21 import EventPairFastMemoryStage, HardwareAwareAERATextLMV21


@dataclass
class DualDeltaFastMemoryState(FastMemoryState):
    """Session-local associative matrix plus inverse observed-key covariance."""

    inverse_key_covariance: torch.Tensor

    def detach(self) -> "DualDeltaFastMemoryState":
        return DualDeltaFastMemoryState(
            self.matrix.detach(),
            self.inverse_key_covariance.detach(),
        )


class PretrainableDualDeltaFastMemory(PretrainableDeltaFastMemory):
    """V21 projections/read path with extra non-parametric covariance state."""

    def empty_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DualDeltaFastMemoryState:
        matrix = torch.zeros(
            batch_size,
            self.memory_dim,
            self.memory_dim,
            device=device,
            dtype=dtype,
        )
        eye = torch.eye(self.memory_dim, device=device, dtype=dtype)
        inverse = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return DualDeltaFastMemoryState(matrix, inverse)


def interference_corrected_dual_delta_update(
    matrix: torch.Tensor,
    inverse_key_covariance: torch.Tensor,
    keys: torch.Tensor,
    targets: torch.Tensor,
    strengths: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential covariance-preconditioned error correction.

    ``keys`` are expected normalized.  For a strength-one write, the dual
    direction is normalized so the current key's own prediction is corrected by
    the full residual.  The Sherman-Morrison state makes future dual directions
    account for key geometry already observed in this session.
    """
    if matrix.ndim != 3 or inverse_key_covariance.shape != matrix.shape:
        raise ValueError("matrix and inverse covariance must match [batch,dim,dim]")
    if keys.ndim != 3 or targets.shape != keys.shape:
        raise ValueError("keys and targets must match [batch,candidates,dim]")
    if strengths.shape != (*keys.shape[:-1], 1):
        raise ValueError("strengths must be [batch,candidates,1]")
    if matrix.size(0) != keys.size(0) or matrix.size(1) != keys.size(2):
        raise ValueError("state/key dimension mismatch")

    m = matrix
    p = inverse_key_covariance
    for t in range(keys.size(1)):
        key = keys[:, t]
        target = targets[:, t]
        strength = strengths[:, t].clamp(0.0, 1.0)

        pk = torch.einsum("bij,bj->bi", p, key)
        key_metric = (key * pk).sum(dim=-1, keepdim=True).clamp_min(eps)
        dual = pk / key_metric

        prediction = torch.einsum("bi,bij->bj", key, m)
        error = target - prediction
        m = m + torch.einsum("bi,bj->bij", dual * strength, error)

        # P <- (P^-1 + s*k*k^T)^-1 via weighted Sherman-Morrison.
        covariance_denom = (1.0 + strength * key_metric).clamp_min(eps)
        scale = (strength / covariance_denom).unsqueeze(-1)
        p = p - scale * torch.einsum("bi,bj->bij", pk, pk)
        # Keep numerical drift from breaking the intended symmetric metric.
        p = 0.5 * (p + p.transpose(-1, -2))

    return m, p


class InterferenceCorrectedEventPairFastMemoryStage(EventPairFastMemoryStage):
    """V21 event-pair stage with dual-delta session-local writes."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__(source)
        if hasattr(source, "pair_write_gate"):
            self.pair_write_gate.load_state_dict(source.pair_write_gate.state_dict())

        old = source.memory
        if not isinstance(old, PretrainableDeltaFastMemory):
            raise TypeError("v22 source memory must be PretrainableDeltaFastMemory")
        new = PretrainableDualDeltaFastMemory(
            self.cfg.d_model,
            self.cfg.memory_dim,
            lr=1.0,
            decay=1.0,
        ).to(device=old.q.weight.device, dtype=old.q.weight.dtype)
        new.load_state_dict(old.state_dict(), strict=True)
        new.set_differentiable_pretraining(old.differentiable_pretraining)
        self.memory = new

    def _event_pair_update(
        self,
        address_source: torch.Tensor,
        payload_source: torch.Tensor,
        write_strength: torch.Tensor,
        state: FastMemoryState,
    ) -> DualDeltaFastMemoryState:
        memory = self.memory
        if not isinstance(memory, PretrainableDualDeltaFastMemory):
            raise TypeError("v22 requires PretrainableDualDeltaFastMemory")
        if not isinstance(state, DualDeltaFastMemoryState):
            raise TypeError("v22 requires DualDeltaFastMemoryState")
        if address_source.shape != payload_source.shape or address_source.ndim != 3:
            raise ValueError("address/payload sources must match [batch,candidates,d_model]")
        if write_strength.shape != (*address_source.shape[:-1], 1):
            raise ValueError("write_strength must be [batch,candidates,1]")

        def update(*, detach_inputs: bool) -> DualDeltaFastMemoryState:
            a = address_source.detach() if detach_inputs else address_source
            payload = payload_source.detach() if detach_inputs else payload_source
            strength = write_strength.detach() if detach_inputs else write_strength
            matrix = state.matrix.detach().clone() if detach_inputs else state.matrix
            inverse = (
                state.inverse_key_covariance.detach().clone()
                if detach_inputs
                else state.inverse_key_covariance
            )
            keys = F.normalize(memory.k(a), dim=-1)
            targets = torch.tanh(memory.v(payload))
            next_matrix, next_inverse = interference_corrected_dual_delta_update(
                matrix,
                inverse,
                keys,
                targets,
                strength,
            )
            if detach_inputs:
                next_matrix = next_matrix.detach()
                next_inverse = next_inverse.detach()
            return DualDeltaFastMemoryState(next_matrix, next_inverse)

        if memory.differentiable_pretraining:
            return update(detach_inputs=False)
        with torch.no_grad():
            return update(detach_inputs=True)


class HardwareAwareAERATextLMV22(HardwareAwareAERATextLMV21):
    """V21 plus interference-corrected dual-delta fast-memory writes."""

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stages = nn.ModuleList(
            InterferenceCorrectedEventPairFastMemoryStage(stage)
            for stage in self.stages
        )


def dual_delta_memory_protocol() -> dict[str, object]:
    return {
        "version": "aera-v22-interference-corrected-dual-delta",
        "source": "aera-v21",
        "qkvout_dimension_changed": False,
        "learned_parameter_count_changed": False,
        "read_path_changed": False,
        "event_pair_candidates_changed": False,
        "routing_changed": False,
        "stream_changed": False,
        "extra_session_state": "one memory_dim x memory_dim inverse-key-covariance matrix per stage",
        "write_direction": "P*k/(k^T*P*k)",
        "write_correction": "M += strength * dual * (target - k^T M)",
        "covariance_update": "weighted Sherman-Morrison",
        "blanket_matrix_decay": False,
        "deployment_local_update_detached": True,
        "base_pretraining_update_differentiable": True,
        "gpu_authorized": False,
    }
