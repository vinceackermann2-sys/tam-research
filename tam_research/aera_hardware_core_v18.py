from __future__ import annotations

import torch
import torch.nn.functional as F

from .aera import FastMemoryState
from .aera_delta_memory import DeltaFastMemory
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v17 import HardwareAwareAERATextLMV17


class PretrainableDeltaFastMemory(DeltaFastMemory):
    """Delta memory with identical pretraining and deployment update equations.

    Deployment keeps the established ``DeltaFastMemory.local_update`` semantics:
    the write is detached/no-grad and therefore never mutates or backpropagates
    through base-model parameters.

    Base pretraining may opt into a differentiable form of the *same* delta rule.
    That lets later-chunk task loss teach q/k/v/out and the controller's write
    strength while leaving deployment local adaptation gradient-free.
    """

    def __init__(
        self,
        d_model: int,
        memory_dim: int,
        *,
        lr: float = 0.2,
        decay: float = 0.999,
    ) -> None:
        super().__init__(d_model, memory_dim, lr=lr, decay=decay)
        self.differentiable_pretraining = False

    def set_differentiable_pretraining(self, enabled: bool) -> None:
        self.differentiable_pretraining = bool(enabled)

    def differentiable_update(
        self,
        x: torch.Tensor,
        write_strength: torch.Tensor,
        state: FastMemoryState,
    ) -> FastMemoryState:
        """Apply the deployment delta rule without detaching the training graph."""
        if write_strength.shape != (*x.shape[:-1], 1):
            raise ValueError("write_strength must be [batch,time,1]")
        if state.matrix.ndim != 3 or state.matrix.size(0) != x.size(0):
            raise ValueError("memory state batch mismatch")

        keys = F.normalize(self.k(x), dim=-1)
        targets = torch.tanh(self.v(x))
        strength = write_strength.clamp(0.0, 1.0)
        matrix = state.matrix

        for t in range(x.size(1)):
            matrix = self.decay * matrix
            key = keys[:, t]
            target = targets[:, t]
            pred = torch.einsum("bi,bij->bj", key, matrix)
            error = target - pred
            eta = self.lr * strength[:, t]
            matrix = matrix + torch.einsum("bi,bj->bij", key * eta, error)
        return FastMemoryState(matrix)

    def local_update(
        self,
        x: torch.Tensor,
        write_strength: torch.Tensor,
        state: FastMemoryState,
    ) -> FastMemoryState:
        if self.differentiable_pretraining:
            return self.differentiable_update(x, write_strength, state)
        return super().local_update(x, write_strength, state)


class HardwareAwareAERATextLMV18(HardwareAwareAERATextLMV17):
    """V17 routing core with trainable-through-time fast-memory writes.

    This revision does not change routing, experts, latent depth, target rates,
    thresholding, state shape, memory equation, or stored parameter count. It only
    makes the already-existing local delta memory differentiable during *base
    pretraining* when explicitly enabled. Inference defaults to the established
    detached local-update rule.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        for stage in self.stages:
            old = stage.memory
            new = PretrainableDeltaFastMemory(
                cfg.d_model,
                cfg.memory_dim,
                lr=cfg.fast_memory_lr,
                decay=cfg.fast_memory_decay,
            ).to(device=old.q.weight.device, dtype=old.q.weight.dtype)
            new.load_state_dict(old.state_dict(), strict=True)
            stage.memory = new
        self.set_memory_pretraining_mode(False)

    def set_memory_pretraining_mode(self, enabled: bool) -> None:
        for stage in self.stages:
            memory = stage.memory
            if not isinstance(memory, PretrainableDeltaFastMemory):
                raise TypeError("v18 stage memory must be PretrainableDeltaFastMemory")
            memory.set_differentiable_pretraining(enabled)

    def memory_pretraining_mode(self) -> bool:
        flags = []
        for stage in self.stages:
            memory = stage.memory
            if not isinstance(memory, PretrainableDeltaFastMemory):
                raise TypeError("v18 stage memory must be PretrainableDeltaFastMemory")
            flags.append(memory.differentiable_pretraining)
        if len(set(flags)) != 1:
            raise RuntimeError("v18 memory pretraining flags disagree across stages")
        return flags[0]


def memory_pretraining_protocol() -> dict[str, object]:
    return {
        "memory_equation_changed": False,
        "stored_parameter_count_changed": False,
        "routing_changed_from_v17": False,
        "deployment_local_update_detached": True,
        "base_pretraining_update_differentiable": True,
        "memory_update_scope": "within-sequence causal chunk state",
        "gpu_authorized": False,
    }
