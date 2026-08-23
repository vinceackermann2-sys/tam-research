from __future__ import annotations

import torch
import torch.nn as nn

from .aera import AERAState
from .aera_hardware_core import HardwareAERAStage, HardwareAwareAERATextLM, HardwareAERAConfig


class UnconditionalStreamAERAStage(HardwareAERAStage):
    """Hardware-aware stage where compressed stream state is always readable.

    The persistent stream is the architecture's canonical cross-chunk working state,
    not an optional retrieval source. Making it a residual prevents a controller
    from suppressing the only information path that spans chunk boundaries. Fast
    neural memory remains gated because it is optional, higher-latency retrieval.
    """

    def forward_chunk(
        self,
        events: torch.Tensor,
        state: AERAState | None,
        *,
        hard: bool,
        update_memory: bool,
    ):
        if events.ndim != 3 or events.size(1) > self.cfg.chunk_size:
            raise ValueError("events must be [batch,time,d_model] within chunk_size")
        if state is None:
            state = self.empty_state(events)

        h = self.norm(events)
        start_control = self.controller(h[:, 0], state.stream)
        self.last_start_controls = {
            k: v.detach() for k, v in start_control.items() if "logits" not in k
        }

        memory_read = self.memory.read(h[:, :1], state.memory).squeeze(1)
        carried = self.state_to_chunk(state.stream)
        # Architectural correction from learned CPU gate #176:
        # stream state is an unconditional residual; optional fast memory stays gated.
        context = carried + start_control["memory_read"] * memory_read
        h = h + context[:, None, :]

        h = h + self.attn(h)
        h = h + self.experts(
            h,
            start_control["expert_logits"],
            start_control["expert_count_logits"],
            hard=hard,
        )

        end_summary = h[:, -1]
        end_control = self.controller(end_summary, state.stream)
        self.last_end_controls = {
            k: v.detach() for k, v in end_control.items() if "logits" not in k
        }
        reasoned = self.reasoner(end_summary, end_control["depth_logits"], hard=hard)

        last_mask = torch.zeros(h.size(1), device=h.device, dtype=h.dtype)
        last_mask[-1] = 1
        h = h + self.reason_to_chunk(reasoned)[:, None, :] * last_mask[None, :, None]
        h = self.out_norm(h)

        final_stream = self.stream_cell(reasoned, state.stream)
        memory_state = state.memory
        if update_memory:
            write = (end_control["novelty"] * end_control["memory_write"]).clamp(0.0, 1.0)
            memory_state = self.memory.local_update(
                reasoned[:, None, :], write[:, None, :], state.memory
            )

        return h, AERAState(final_stream, memory_state), {
            "start": start_control,
            "end": end_control,
        }


class HardwareAwareAERATextLMV2(HardwareAwareAERATextLM):
    """Canonical hardware-aware AERA v2 with unconditional stream residual."""

    def __init__(self, cfg: HardwareAERAConfig = HardwareAERAConfig()):
        super().__init__(cfg)
        self.stages = nn.ModuleList(UnconditionalStreamAERAStage(cfg) for _ in range(cfg.n_stages))
