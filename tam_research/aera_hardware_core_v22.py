from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState, FastMemoryState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v8 import StageRouteGate
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

    ``keys`` are expected normalized. For a strength-one write, the dual
    direction is normalized so the current key's own prediction is corrected by
    the full residual. The Sherman-Morrison state makes future dual directions
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


def _as_dual(memory: FastMemoryState) -> DualDeltaFastMemoryState:
    if not isinstance(memory, DualDeltaFastMemoryState):
        raise TypeError("v22 routing requires DualDeltaFastMemoryState")
    return memory


def _select_dual_state(state: AERAState, idx: torch.Tensor) -> AERAState:
    memory = _as_dual(state.memory)
    return AERAState(
        stream=state.stream.index_select(0, idx),
        memory=DualDeltaFastMemoryState(
            memory.matrix.index_select(0, idx),
            memory.inverse_key_covariance.index_select(0, idx),
        ),
    )


def _restore_dual_state_dtype(base: AERAState, update: AERAState) -> AERAState:
    base_memory = _as_dual(base.memory)
    update_memory = _as_dual(update.memory)
    return AERAState(
        stream=update.stream.to(dtype=base.stream.dtype),
        memory=DualDeltaFastMemoryState(
            update_memory.matrix.to(dtype=base_memory.matrix.dtype),
            update_memory.inverse_key_covariance.to(
                dtype=base_memory.inverse_key_covariance.dtype
            ),
        ),
    )


def _merge_dual_state(base: AERAState, update: AERAState, idx: torch.Tensor) -> AERAState:
    update = _restore_dual_state_dtype(base, update)
    base_memory = _as_dual(base.memory)
    update_memory = _as_dual(update.memory)
    return AERAState(
        stream=base.stream.index_copy(0, idx, update.stream),
        memory=DualDeltaFastMemoryState(
            base_memory.matrix.index_copy(0, idx, update_memory.matrix),
            base_memory.inverse_key_covariance.index_copy(
                0, idx, update_memory.inverse_key_covariance
            ),
        ),
    )


def _blend_dual_state(base: AERAState, update: AERAState, gate: torch.Tensor) -> AERAState:
    base_memory = _as_dual(base.memory)
    update_memory = _as_dual(update.memory)
    g1 = gate.to(base.stream.dtype)
    g2 = gate[:, :, None].to(base_memory.matrix.dtype)
    gp = gate[:, :, None].to(base_memory.inverse_key_covariance.dtype)
    return AERAState(
        stream=base.stream + g1 * (update.stream - base.stream),
        memory=DualDeltaFastMemoryState(
            base_memory.matrix + g2 * (update_memory.matrix - base_memory.matrix),
            base_memory.inverse_key_covariance
            + gp
            * (
                update_memory.inverse_key_covariance
                - base_memory.inverse_key_covariance
            ),
        ),
    )


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

    def _route_one_stage(
        self,
        x: torch.Tensor,
        stage: nn.Module,
        stage_state: AERAState,
        router: StageRouteGate,
        *,
        route_mode: str,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, object]]:
        """Preserve both M and P across v21's sparse/calibration routing semantics."""
        gate, logits = router(x[:, 0], stage_state.stream, mode=route_mode)
        prob = torch.sigmoid(logits)

        if route_mode == "hard_sparse":
            run_idx = (gate[:, 0] >= 0.5).nonzero(as_tuple=False).squeeze(-1)
            if run_idx.numel() == 0:
                return x, stage_state, {
                    "stage_route_probability": prob,
                    "stage_route_gate": gate,
                    "executed_fraction": 0.0,
                    "start": None,
                    "end": None,
                }

            selected_x = x.index_select(0, run_idx)
            selected_state = _select_dual_state(stage_state, run_idx)
            selected_y, selected_new_state, selected_controls = stage.forward_chunk(
                selected_x,
                selected_state,
                hard=True,
                update_memory=update_memory,
            )
            selected_y = selected_y.to(dtype=x.dtype)
            selected_new_state = _restore_dual_state_dtype(
                selected_state, selected_new_state
            )
            y = x.index_copy(0, run_idx, selected_y)
            new_state = _merge_dual_state(stage_state, selected_new_state, run_idx)
            return y, new_state, {
                "stage_route_probability": prob,
                "stage_route_gate": gate,
                "executed_fraction": float(run_idx.numel() / x.size(0)),
                "start": selected_controls["start"],
                "end": selected_controls["end"],
            }

        isolate = bool(
            route_mode == "straight_through"
            and getattr(self, "_isolate_router_task_gradient", False)
        )
        task_gate = gate.detach() if isolate else gate
        processed, processed_state, controls = stage.forward_chunk(
            x,
            stage_state,
            hard=False,
            update_memory=update_memory,
        )
        processed = processed.to(dtype=x.dtype)
        processed_state = _restore_dual_state_dtype(stage_state, processed_state)
        gate_for_residual = task_gate.to(dtype=x.dtype)
        y = x + gate_for_residual[:, None, :] * (processed - x)
        new_state = _blend_dual_state(stage_state, processed_state, task_gate)
        info: dict[str, object] = {
            "stage_route_probability": prob,
            "stage_route_gate": task_gate,
            "executed_fraction": 1.0,
            "start": controls["start"],
            "end": controls["end"],
        }
        if isolate:
            info["task_router_gradient_isolated"] = True
        return y, new_state, info


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
