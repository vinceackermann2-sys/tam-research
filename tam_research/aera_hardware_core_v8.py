from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aera import AERAState, FastMemoryState
from .aera_hardware_core import HardwareAERAConfig, HardwareAERAState
from .aera_hardware_core_v7 import HardwareAwareAERATextLMV7


class StageRouteGate(nn.Module):
    """Cheap causal run/skip decision for one expensive AERA stage.

    The gate sees only the first representation in the current chunk plus that
    stage's carried stream state. It therefore cannot inspect future tokens before
    deciding whether the stage runs for the chunk.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, 1)
        self.last_probability: torch.Tensor | None = None
        self.last_hard_gate: torch.Tensor | None = None

    def forward(
        self,
        first_event: torch.Tensor,
        stream: torch.Tensor,
        *,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if first_event.ndim != 2 or stream.shape != first_event.shape:
            raise ValueError("stage router expects [batch,d_model] event and stream")
        if mode not in {"soft", "straight_through", "hard_sparse"}:
            raise ValueError(f"unknown stage routing mode: {mode}")

        logits = self.proj(torch.cat((first_event, stream), dim=-1))
        prob = torch.sigmoid(logits)
        hard = (prob >= 0.5).to(prob.dtype)
        if mode == "soft":
            gate = prob
        elif mode == "straight_through":
            # Binary forward decision with sigmoid gradient. This calibration mode
            # still evaluates both paths; it exists to teach the router, not to
            # claim training-FLOP savings.
            gate = hard.detach() - prob.detach() + prob
        else:
            gate = hard

        self.last_probability = prob.detach().cpu()
        self.last_hard_gate = hard.detach().cpu()
        return gate, logits

    def stats(self) -> dict[str, float] | None:
        if self.last_probability is None or self.last_hard_gate is None:
            return None
        return {
            "mean_run_probability": float(self.last_probability.float().mean()),
            "hard_run_fraction": float(self.last_hard_gate.float().mean()),
        }


@dataclass(frozen=True)
class MostlyHardRoutingSchedule:
    """Training schedule that makes hard sparse execution the common path.

    Every `calibration_every` step uses a straight-through calibration pass so
    task gradients can teach the stage gates. Other steps execute only selected
    stages. A later learned policy/REINFORCE variant can replace this schedule,
    but this already prevents the final architecture from requiring dense-stage
    execution on every training step.
    """

    calibration_every: int = 8

    def __post_init__(self) -> None:
        if self.calibration_every < 2:
            raise ValueError("calibration_every must be >=2")

    def mode_for_step(self, step: int) -> str:
        if step < 0:
            raise ValueError("step must be >=0")
        return "straight_through" if step % self.calibration_every == 0 else "hard_sparse"

    @property
    def nominal_hard_sparse_fraction(self) -> float:
        return (self.calibration_every - 1) / self.calibration_every


def _select_state(state: AERAState, idx: torch.Tensor) -> AERAState:
    return AERAState(
        stream=state.stream.index_select(0, idx),
        memory=FastMemoryState(state.memory.matrix.index_select(0, idx)),
    )


def _merge_state(base: AERAState, update: AERAState, idx: torch.Tensor) -> AERAState:
    return AERAState(
        stream=base.stream.index_copy(0, idx, update.stream),
        memory=FastMemoryState(base.memory.matrix.index_copy(0, idx, update.memory.matrix)),
    )


def _blend_state(base: AERAState, update: AERAState, gate: torch.Tensor) -> AERAState:
    g1 = gate.to(base.stream.dtype)
    g2 = gate[:, :, None].to(base.memory.matrix.dtype)
    return AERAState(
        stream=base.stream + g1 * (update.stream - base.stream),
        memory=FastMemoryState(
            base.memory.matrix + g2 * (update.memory.matrix - base.memory.matrix)
        ),
    )


class HardwareAwareAERATextLMV8(HardwareAwareAERATextLMV7):
    """AERA pre-scale candidate with causal whole-stage conditional execution.

    Routing modes:
      * soft: all stages execute and are mixed by probabilities; controller warmup.
      * straight_through: all stages execute, binary forward gates carry sigmoid
        gradients; sparse-router calibration.
      * hard_sparse: only examples whose gate says RUN execute that stage; skipped
        examples preserve both residual representation and per-stage state exactly.

    The intended training regime is mostly hard_sparse steps with periodic
    straight_through calibration. Inference uses hard_sparse.
    """

    def __init__(
        self,
        cfg: HardwareAERAConfig = HardwareAERAConfig(),
        *,
        stream_forecast_tokens: int = 4,
    ):
        super().__init__(cfg, stream_forecast_tokens=stream_forecast_tokens)
        self.stage_routers = nn.ModuleList(StageRouteGate(cfg.d_model) for _ in range(cfg.n_stages))
        self.last_stage_execution: list[dict[str, float]] = []

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
        gate, logits = router(x[:, 0], stage_state.stream, mode=route_mode)
        prob = torch.sigmoid(logits)

        if route_mode == "hard_sparse":
            run_idx = (gate[:, 0] >= 0.5).nonzero(as_tuple=False).squeeze(-1)
            if run_idx.numel() == 0:
                info: dict[str, object] = {
                    "stage_route_probability": prob,
                    "stage_route_gate": gate,
                    "executed_fraction": 0.0,
                    "start": None,
                    "end": None,
                }
                return x, stage_state, info

            selected_x = x.index_select(0, run_idx)
            selected_state = _select_state(stage_state, run_idx)
            selected_y, selected_new_state, selected_controls = stage.forward_chunk(
                selected_x,
                selected_state,
                hard=True,
                update_memory=update_memory,
            )
            y = x.index_copy(0, run_idx, selected_y)
            new_state = _merge_state(stage_state, selected_new_state, run_idx)
            info = {
                "stage_route_probability": prob,
                "stage_route_gate": gate,
                "executed_fraction": float(run_idx.numel() / x.size(0)),
                "start": selected_controls["start"],
                "end": selected_controls["end"],
            }
            return y, new_state, info

        # Calibration modes evaluate the stage for every example. `gate` is either
        # a soft probability or a straight-through binary value.
        processed, processed_state, controls = stage.forward_chunk(
            x,
            stage_state,
            hard=False,
            update_memory=update_memory,
        )
        y = x + gate[:, None, :] * (processed - x)
        new_state = _blend_state(stage_state, processed_state, gate)
        info = {
            "stage_route_probability": prob,
            "stage_route_gate": gate,
            "executed_fraction": 1.0,
            "start": controls["start"],
            "end": controls["end"],
        }
        return y, new_state, info

    def forward(
        self,
        tokens: torch.Tensor,
        state: HardwareAERAState | None = None,
        *,
        hard: bool = False,
        update_memory: bool = False,
        return_block_logits: bool = False,
        route_mode: str | None = None,
    ) -> dict[str, object]:
        if tokens.ndim != 2 or tokens.size(1) < 1:
            raise ValueError("tokens must be nonempty [batch,time]")
        if route_mode is None:
            route_mode = "hard_sparse" if hard else "soft"
        if route_mode not in {"soft", "straight_through", "hard_sparse"}:
            raise ValueError(f"unknown route_mode: {route_mode}")
        if state is None:
            state = self.empty_state(tokens)
        if len(state.stages) != len(self.stages):
            raise ValueError("state stage count mismatch")

        outputs: list[torch.Tensor] = []
        control_history: list[list[dict[str, object]]] = []
        stream_history: list[list[torch.Tensor]] = []
        current_state = state
        execution_accum = [0.0 for _ in self.stages]
        execution_chunks = 0

        for start in range(0, tokens.size(1), self.cfg.chunk_size):
            chunk = tokens[:, start : start + self.cfg.chunk_size]
            pos = torch.arange(chunk.size(1), device=tokens.device)
            x = self.token_emb(chunk) + self.local_pos(pos)[None]
            new_states: list[AERAState] = []
            stage_controls: list[dict[str, object]] = []
            stage_streams: list[torch.Tensor] = []

            for i, (stage, stage_state, router) in enumerate(
                zip(self.stages, current_state.stages, self.stage_routers)
            ):
                x, new_state, info = self._route_one_stage(
                    x,
                    stage,
                    stage_state,
                    router,
                    route_mode=route_mode,
                    update_memory=update_memory,
                )
                new_states.append(new_state)
                stage_controls.append(info)
                stage_streams.append(new_state.stream)
                execution_accum[i] += float(info["executed_fraction"])

            outputs.append(x)
            control_history.append(stage_controls)
            stream_history.append(stage_streams)
            current_state = HardwareAERAState(new_states)
            execution_chunks += 1

        self.last_stage_execution = [
            {
                "stage": float(i),
                "mean_executed_fraction": execution_accum[i] / max(execution_chunks, 1),
            }
            for i in range(len(self.stages))
        ]

        hidden = self.norm(torch.cat(outputs, dim=1))
        result: dict[str, object] = {
            "logits": self.lm_head(hidden),
            "hidden": hidden,
            "state": current_state,
            "next_event_prediction": self.next_event(hidden),
            "controls": control_history,
            "stage_routes": control_history,
            "stream_history": stream_history,
            "routing_mode": route_mode,
        }
        if return_block_logits:
            result["block_logits"] = self.block_draft(hidden, self.lm_head)
        return result

    def soft_objective(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
        *,
        stage_compute_weight: float = 0.002,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if output.get("routing_mode") == "hard_sparse":
            raise ValueError("use hard_sparse_task_loss for hard_sparse training steps")

        # Reconstruct the base control structure expected by the inherited
        # objective while preserving v3's predictive stream-history objective.
        routes = output["stage_routes"]
        assert isinstance(routes, list)
        base_controls: list[list[dict[str, dict[str, torch.Tensor]]]] = []
        stage_probs: list[torch.Tensor] = []
        for chunk in routes:
            base_chunk = []
            for item in chunk:
                start = item["start"]
                end = item["end"]
                assert isinstance(start, dict) and isinstance(end, dict)
                base_chunk.append({"start": start, "end": end})
                stage_probs.append(item["stage_route_probability"].float().mean())
            base_controls.append(base_chunk)

        base_output = dict(output)
        base_output["controls"] = base_controls
        terms = super().objective(tokens, base_output, **kwargs)
        stage_compute = torch.stack(stage_probs).mean()
        total = terms["total"] + stage_compute_weight * stage_compute
        return {**terms, "stage_compute": stage_compute, "total": total}

    def _stream_forecast_loss(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
    ) -> torch.Tensor:
        history = output.get("stream_history")
        if not isinstance(history, list):
            raise ValueError("v8 output missing stream_history")
        terms: list[torch.Tensor] = []
        for chunk_index, stage_streams in enumerate(history[:-1]):
            next_start = (chunk_index + 1) * self.cfg.chunk_size
            remaining = tokens.size(1) - next_start
            k = min(self.stream_forecast_tokens, remaining)
            if k <= 0:
                continue
            target = tokens[:, next_start : next_start + k]
            for stage_index, stream in enumerate(stage_streams):
                pred = self.stream_forecast_heads[stage_index](stream)
                pred = pred.view(
                    stream.size(0), self.stream_forecast_tokens, self.cfg.vocab_size
                )[:, :k]
                terms.append(
                    F.cross_entropy(
                        pred.float().reshape(-1, self.cfg.vocab_size), target.reshape(-1)
                    )
                )
        return (
            torch.stack(terms).mean()
            if terms
            else torch.zeros((), device=tokens.device, dtype=torch.float32)
        )

    def hard_sparse_task_loss(
        self,
        tokens: torch.Tensor,
        output: dict[str, object],
        *,
        event_weight: float = 0.05,
        block_weight: float = 0.25,
        stream_forecast_weight: float = 0.20,
    ) -> dict[str, torch.Tensor]:
        """Task loss for majority hard-sparse training steps.

        Router calibration is intentionally supplied by periodic soft/ST passes;
        this loss trains only the actually executed model path, avoiding the hidden
        dense-stage counterfactual that would erase training-compute savings. The
        predictive-stream forecast remains active so sparse training does not drop
        AERA's central compressed-state learning signal.
        """
        if output.get("routing_mode") != "hard_sparse":
            raise ValueError("hard_sparse_task_loss requires route_mode='hard_sparse'")
        logits = output["logits"]
        event_pred = output["next_event_prediction"]
        assert isinstance(logits, torch.Tensor) and isinstance(event_pred, torch.Tensor)
        lm = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        target_event = self.token_emb(tokens[:, 1:]).detach()
        event = F.mse_loss(event_pred[:, :-1].float(), target_event.float())
        block = torch.zeros((), device=tokens.device)
        block_logits = output.get("block_logits")
        if isinstance(block_logits, torch.Tensor):
            valid = tokens.size(1) - self.cfg.block_size
            if valid > 0:
                terms = []
                for offset in range(self.cfg.block_size):
                    pred = block_logits[:, :valid, offset]
                    target = tokens[:, offset + 1 : offset + 1 + valid]
                    terms.append(
                        F.cross_entropy(
                            pred.float().reshape(-1, self.cfg.vocab_size), target.reshape(-1)
                        )
                    )
                block = torch.stack(terms).mean()
        forecast = self._stream_forecast_loss(tokens, output)
        total = (
            lm
            + event_weight * event
            + block_weight * block
            + stream_forecast_weight * forecast
        )
        return {
            "total": total,
            "next_token": lm,
            "next_event": event,
            "block": block,
            "stream_forecast": forecast,
        }

    def stats(self) -> dict[str, object]:
        return {
            "stages": [stage.stats() for stage in self.stages],
            "stage_routers": [router.stats() for router in self.stage_routers],
            "stage_execution": self.last_stage_execution,
        }
