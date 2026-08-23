from __future__ import annotations

import torch

from .aera import AERAState, FastMemoryState
from .aera_hardware_core import HardwareAERAConfig
from .aera_hardware_core_v6 import BMMHardSparseExpertBank
from .aera_hardware_core_v8 import HardwareAwareAERATextLMV8, StageRouteGate, _select_state


def _restore_state_dtype(base: AERAState, update: AERAState) -> AERAState:
    """Restore an executed sparse branch to the skipped branch's state dtypes.

    Mixed-precision kernels are allowed to return FP32 even when the residual/state
    entering a hard-sparse merge is BF16. Sparse index_copy requires exact dtype
    equality, and—more importantly—the executed and skipped examples must leave the
    router with one coherent representation dtype. This cast is differentiable and
    changes no routing semantics.
    """

    return AERAState(
        stream=update.stream.to(dtype=base.stream.dtype),
        memory=FastMemoryState(update.memory.matrix.to(dtype=base.memory.matrix.dtype)),
    )


def _merge_state_dtype_safe(base: AERAState, update: AERAState, idx: torch.Tensor) -> AERAState:
    update = _restore_state_dtype(base, update)
    return AERAState(
        stream=base.stream.index_copy(0, idx, update.stream),
        memory=FastMemoryState(base.memory.matrix.index_copy(0, idx, update.memory.matrix)),
    )


class HardwareAwareAERATextLMV9(HardwareAwareAERATextLMV8):
    """Pre-scale AERA candidate with dtype-safe sparse merges and verified BMM MoE.

    This is an implementation-hardening revision of v8, not an architecture change:
    causal whole-stage routing, predictive stream state, fast memory, expert count,
    latent depth and objectives are unchanged. The hard stage merge explicitly
    restores executed residual/state tensors to the skipped path dtype, and the L4-
    verified v6 BMM expert backend is installed by default instead of native grouped
    GEMM (which was slower at the tested shapes).
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ):
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        # Preserve the initialized expert weights while changing only execution backend.
        for stage in self.stages:
            old = stage.experts
            new = BMMHardSparseExpertBank(cfg).to(device=old.w1.device, dtype=old.w1.dtype)
            new.load_state_dict(old.state_dict())
            stage.experts = new

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
        if route_mode != "hard_sparse":
            return super()._route_one_stage(
                x,
                stage,
                stage_state,
                router,
                route_mode=route_mode,
                update_memory=update_memory,
            )

        gate, logits = router(x[:, 0], stage_state.stream, mode=route_mode)
        prob = torch.sigmoid(logits)
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
        selected_state = _select_state(stage_state, run_idx)
        selected_y, selected_new_state, selected_controls = stage.forward_chunk(
            selected_x,
            selected_state,
            hard=True,
            update_memory=update_memory,
        )

        # Executed kernels may promote to FP32 under autocast; hard sparse merging
        # must match the skipped residual/state dtypes exactly.
        selected_y = selected_y.to(dtype=x.dtype)
        selected_new_state = _restore_state_dtype(selected_state, selected_new_state)
        y = x.index_copy(0, run_idx, selected_y)
        new_state = _merge_state_dtype_safe(stage_state, selected_new_state, run_idx)
        return y, new_state, {
            "stage_route_probability": prob,
            "stage_route_gate": gate,
            "executed_fraction": float(run_idx.numel() / x.size(0)),
            "start": selected_controls["start"],
            "end": selected_controls["end"],
        }
