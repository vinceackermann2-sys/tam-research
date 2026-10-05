from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.cpw_v0.model import (
    CPWMixer,
    CPWV0Config,
    CPWV0ResearchLM,
    _norm_mse,
)


CPW_ARMS = {
    "full",
    "no_aux",
    "uniform_all",
    "no_memory",
    "no_sequence",
    "no_world",
    "no_attention",
}

_DISABLED_INDEX = {
    "no_attention": 0,
    "no_world": 1,
    "no_sequence": 2,
    "no_memory": 3,
}


class CPWAblationMixer(CPWMixer):
    def __init__(self, cfg: CPWV0Config, arm: str):
        if arm not in CPW_ARMS:
            raise ValueError(f"unknown CPW ablation arm: {arm}")
        super().__init__(cfg)
        self.arm = arm

    def forward(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape != workspace.shape:
            raise ValueError("workspace must match the token-state shape")

        context = x + self.cfg.workspace_mix * workspace
        attention = self.attention(context)
        world, world_state = self.world(context)

        previous = torch.cat(
            (torch.zeros_like(context[:, :1]), context[:, :-1]),
            dim=1,
        )
        sequence = self.sequence(previous)

        prefix = torch.cumsum(context, dim=1)
        denom = torch.arange(
            1,
            context.size(1) + 1,
            device=context.device,
            dtype=context.dtype,
        ).view(1, -1, 1)
        prefix = prefix / denom
        previous_prefix = torch.cat(
            (torch.zeros_like(prefix[:, :1]), prefix[:, :-1]),
            dim=1,
        )
        memory = self.memory(previous_prefix)

        branches = torch.stack(
            (attention, world, sequence, memory),
            dim=2,
        )
        router_logits = self.router(context)

        if self.arm == "uniform_all":
            mixed = branches.mean(dim=2)
            mask = torch.ones_like(router_logits)
        else:
            routed_logits = router_logits.float()
            disabled = _DISABLED_INDEX.get(self.arm)
            if disabled is not None:
                routed_logits = routed_logits.clone()
                routed_logits[..., disabled] = float("-inf")
            top_values, top_indices = torch.topk(
                routed_logits,
                k=self.cfg.router_top_k,
                dim=-1,
            )
            top_weights = torch.softmax(top_values, dim=-1).to(context.dtype)
            gather_index = top_indices.unsqueeze(-1).expand(
                -1, -1, -1, context.size(-1)
            )
            selected = torch.gather(branches, 2, gather_index)
            mixed = (selected * top_weights.unsqueeze(-1)).sum(dim=2)
            mask = torch.zeros_like(router_logits)
            mask.scatter_(-1, top_indices, 1.0)

        aux_terms: list[torch.Tensor] = []
        if self.arm != "no_aux":
            if context.size(1) > 1 and self.arm != "no_sequence":
                aux_terms.append(_norm_mse(sequence[:, 1:], context[:, 1:]))
            if context.size(1) > 1 and self.arm != "no_world":
                aux_terms.append(_norm_mse(world[:, :-1], context[:, 1:]))
            h = self.cfg.memory_horizon
            if context.size(1) > h and self.arm != "no_memory":
                aux_terms.append(_norm_mse(memory[:, :-h], context[:, h:]))

        aux = (
            torch.stack(aux_terms).mean()
            if aux_terms
            else context.float().sum() * 0.0
        )

        if self.arm == "no_world":
            next_workspace = workspace
        else:
            next_workspace = (
                (1.0 - self.cfg.workspace_mix) * workspace
                + self.cfg.workspace_mix * world
            )

        full_probs = torch.softmax(router_logits.float(), dim=-1)
        entropy = -(full_probs * full_probs.clamp_min(1e-9).log()).sum(-1)

        self.last_aux_loss = aux
        self.last_stats = {
            "active_fraction": mask.mean().detach(),
            "router_entropy": entropy.mean().detach(),
            "attention_usage": mask[..., 0].mean().detach(),
            "world_usage": mask[..., 1].mean().detach(),
            "sequence_usage": mask[..., 2].mean().detach(),
            "memory_usage": mask[..., 3].mean().detach(),
            "world_state_norm": world_state.float().norm(dim=-1).mean().detach(),
        }
        return mixed, next_workspace


class CPWV0AblationLM(CPWV0ResearchLM):
    """Parameter-identical CPW-v0 graph with one frozen mechanism intervention."""

    def __init__(
        self,
        arm: str,
        cfg: CPWV0Config = CPWV0Config(),
    ):
        if arm not in CPW_ARMS:
            raise ValueError(f"unknown CPW ablation arm: {arm}")
        super().__init__(cfg)
        self.arm = arm

        # Replace only computation semantics. Copy the already initialized
        # frozen-v0 tensors so every arm retains exactly the same parameter
        # names, shapes, values and count for a given seed.
        for block in self.blocks:
            old = block.mixer
            replacement = CPWAblationMixer(cfg, arm)
            replacement.load_state_dict(old.state_dict(), strict=True)
            block.mixer = replacement
