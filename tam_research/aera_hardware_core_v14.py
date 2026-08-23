from __future__ import annotations

import torch

from .aera import AERAState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v8 import StageRouteGate, _blend_state
from .aera_hardware_core_v9 import _restore_state_dtype
from .aera_hardware_core_v12 import HardwareAwareAERATextLMV12


class HardwareAwareAERATextLMV14(HardwareAwareAERATextLMV12):
    """AERA-v14: isolate optional-router learning from primary task gradients.

    V12/v13 use a straight-through gate during router calibration.  That gives the
    language/task objective a gradient through the binary stage decision, so task
    loss can reward executing more stages at the same time that explicit difficulty
    supervision asks the router to respect a compute budget.  Two corrected real-
    language development seeds converged to the same 75% stage-execution fixed point.

    V14 keeps the forward decision binary and causal but can detach that gate from
    the task graph on calibration passes.  The router logit/probability remains in
    the returned routing record with its graph intact, so explicit difficulty,
    budget, and polarization losses still train the router.  Model architecture,
    inference semantics, parameter count, experts, state/memory, and hard-sparse
    execution are unchanged.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ) -> None:
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self._isolate_router_task_gradient = False

    def set_router_task_gradient_isolation(self, enabled: bool) -> None:
        self._isolate_router_task_gradient = bool(enabled)

    def _route_one_stage(
        self,
        x: torch.Tensor,
        stage,
        stage_state: AERAState,
        router: StageRouteGate,
        *,
        route_mode: str,
        update_memory: bool,
    ) -> tuple[torch.Tensor, AERAState, dict[str, object]]:
        if route_mode != "straight_through" or not self._isolate_router_task_gradient:
            return super()._route_one_stage(
                x,
                stage,
                stage_state,
                router,
                route_mode=route_mode,
                update_memory=update_memory,
            )

        # Ask the established router for its ordinary straight-through binary gate
        # and differentiable logit.  Detach only the gate used by the expensive task
        # path; keep `prob` differentiable for explicit routing supervision.
        gate, logits = router(x[:, 0], stage_state.stream, mode="straight_through")
        prob = torch.sigmoid(logits)
        task_gate = gate.detach()

        processed, processed_state, controls = stage.forward_chunk(
            x,
            stage_state,
            hard=False,
            update_memory=update_memory,
        )
        # Match the v10/v11 mixed-precision calibration contract.
        processed = processed.to(dtype=x.dtype)
        processed_state = _restore_state_dtype(stage_state, processed_state)
        gate_for_residual = task_gate.to(dtype=x.dtype)
        y = x + gate_for_residual[:, None, :] * (processed - x)
        new_state = _blend_state(stage_state, processed_state, task_gate)
        return y, new_state, {
            "stage_route_probability": prob,
            "stage_route_gate": task_gate,
            "executed_fraction": 1.0,
            "start": controls["start"],
            "end": controls["end"],
            "task_router_gradient_isolated": True,
        }
