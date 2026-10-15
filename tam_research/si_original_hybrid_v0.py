"""SI Original Hybrid Workspace v0: CPU-first mechanism study, not an SI model.

Preserves the user's architecture ingredients as separate, ablatable components:
causal local attention + recurrent state, top-1 sparse experts, adaptive
refinement, modality tags, world-state prediction, uncertainty verification,
versioned external memory and bounded fast-weight adaptation interfaces.

Only a single numerical state/action modality is trained in this pilot.
Additional modality tags are scaffolding, not demonstrated multimodality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SIConfig:
    obs_dim: int = 3
    action_dim: int = 4
    d_model: int = 40
    layers: int = 3
    heads: int = 4
    window: int = 6
    experts: int = 2
    max_seq_len: int = 64


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, window: Optional[int]):
        super().__init__()
        assert width % heads == 0
        self.heads, self.width, self.window = heads, width, window
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = [z.view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in self.qkv(x).chunk(3, -1)]
        idx = torch.arange(t, device=x.device)
        causal = idx[None, :] <= idx[:, None]
        if self.window is not None:
            causal = causal & (idx[None, :] > idx[:, None] - self.window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class StateSpaceMixer(nn.Module):
    """Causal diagonal gated recurrence; reference path favors correctness over speed."""
    def __init__(self, width: int):
        super().__init__()
        self.keep = nn.Linear(width, width)
        self.write = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = 0.98 * torch.sigmoid(self.keep(x))
        val = torch.tanh(self.write(x))
        state = torch.zeros_like(x[:, 0, :])
        states = []
        for j in range(x.size(1)):
            state = gate[:, j] * state + (1 - gate[:, j]) * val[:, j]
            states.append(state)
        return self.out(torch.stack(states, 1))


class SparseExperts(nn.Module):
    """Actually evaluate one expert per token, not a dense weighted combination."""
    def __init__(self, width: int, experts: int):
        super().__init__()
        self.router = nn.Linear(width, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
            for _ in range(experts)
        ])

    def forward(self, x: torch.Tensor):
        probs = torch.softmax(self.router(x), dim=-1)
        chosen = probs.argmax(dim=-1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = chosen == i
            if mask.any():
                out[mask] = expert(x[mask]) * probs[..., i][mask, None]
        uniform = 1.0 / len(self.experts)
        balance_loss = (probs.mean(dim=(0, 1)) - uniform).square().sum()
        return out, balance_loss, chosen


class SIHybridBlock(nn.Module):
    def __init__(self, config: SIConfig, layer: int, *, no_moe: bool = False):
        super().__init__()
        self.mixer = (CausalAttention(config.d_model, config.heads, config.window)
                      if layer % 2 == 0 else StateSpaceMixer(config.d_model))
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = (nn.Sequential(nn.Linear(config.d_model, 2 * config.d_model),
                                  nn.GELU(), nn.Linear(2 * config.d_model, config.d_model))
                    if no_moe else SparseExperts(config.d_model, config.experts))
        self.no_moe = no_moe

    def forward(self, x: torch.Tensor):
        x = x + self.mixer(self.norm1(x))
        if self.no_moe:
            return x + self.ffn(self.norm2(x)), x.new_zeros(()), None
        y, balance, route = self.ffn(self.norm2(x))
        return x + y, balance, route


class DenseBlock(nn.Module):
    def __init__(self, config: SIConfig):
        super().__init__()
        self.attn = CausalAttention(config.d_model, config.heads, None)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(nn.LayerNorm(config.d_model),
                                 nn.Linear(config.d_model, 4 * config.d_model + 10),
                                 nn.GELU(), nn.Linear(4 * config.d_model + 10, config.d_model))
        self.residual_gain = nn.Parameter(torch.ones(config.d_model))

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(x) * self.residual_gain


class FastAdapter(nn.Module):
    def __init__(self, width: int, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


class ObservationHeads(nn.Module):
    def __init__(self, width: int, output: int):
        super().__init__()
        self.world = nn.Linear(width, output)
        self.log_variance = nn.Linear(width, output)
        self.verifier = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor):
        return {
            "prediction": self.world(x),
            "log_variance": self.log_variance(x).clamp(-6., 3.),
            "verified_error": F.softplus(self.verifier(x)),
        }


class OriginalSIHybrid(nn.Module):
    """Original hybrid concept with explicit ablations and honest compute semantics."""
    def __init__(self, config: SIConfig, *, no_moe: bool = False,
                 no_refine: bool = False, no_recurrence: bool = False):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.obs_dim + config.action_dim, config.d_model)
        self.modality = nn.Embedding(4, config.d_model)
        self.position = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([
            SIHybridBlock(config, j if not no_recurrence else 0, no_moe=no_moe)
            for j in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.refiner = nn.Sequential(nn.Linear(config.d_model, config.d_model),
                                     nn.GELU(), nn.Linear(config.d_model, config.d_model))
        self.halt = nn.Linear(config.d_model, 1)
        nn.init.constant_(self.halt.bias, -1.)
        self.no_refine = no_refine
        self.fast_adapter = FastAdapter(config.d_model)
        self.heads = ObservationHeads(config.d_model, config.obs_dim)

    def forward(self, obs, actions, *, modality=None, use_fast=False,
                skip_optional_refine=False):
        b, t, _ = obs.shape
        assert t <= self.config.max_seq_len
        if modality is None:
            modality = torch.zeros(b, t, device=obs.device, dtype=torch.long)
        indices = torch.arange(t, device=obs.device)
        x = self.encoder(torch.cat([obs, actions], -1)) + self.modality(modality) + self.position(indices)[None]
        balances, routes = [], []
        for block in self.blocks:
            x, loss, route = block(x)
            balances.append(loss)
            if route is not None:
                routes.append(route)
        # Soft gates allow gradients. This is not a compute saving in training.
        # Whole-sequence early exit in evaluation actually skips refinement.
        halting = torch.sigmoid(self.halt(x))
        did_refine = not self.no_refine and not skip_optional_refine
        if did_refine:
            x = x + (1 - halting) * self.refiner(x)
        if use_fast:
            x = x + self.fast_adapter(x)
        out = self.heads(self.final_norm(x))
        out.update(balance=torch.stack(balances).sum(),
                   halt_probability=halting.mean(),
                   expert_routes=routes, did_refine=did_refine)
        return out

    @torch.no_grad()
    def plan_one_step(self, history_obs, history_actions, candidate_actions, goal):
        """Evaluate action candidates via imagined single-step observation.

        Interface smoke only; no evidence of effective long-horizon planning.
        """
        errors = []
        for i in range(candidate_actions.size(1)):
            acts = history_actions.clone()
            acts[:, -1] = candidate_actions[:, i]
            pred = self(history_obs, acts)["prediction"][:, -1]
            errors.append((pred - goal).square().sum(dim=-1))
        return torch.stack(errors, 1).argmin(1)


class TransformerControl(nn.Module):
    def __init__(self, config: SIConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.obs_dim + config.action_dim, config.d_model)
        self.modality = nn.Embedding(4, config.d_model)
        self.position = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([DenseBlock(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.heads = ObservationHeads(config.d_model, config.obs_dim)

    def forward(self, obs, actions, *, modality=None, **_):
        b, t, _ = obs.shape
        if modality is None:
            modality = torch.zeros(b, t, device=obs.device, dtype=torch.long)
        indices = torch.arange(t, device=obs.device)
        x = self.encoder(torch.cat([obs, actions], -1)) + self.modality(modality) + self.position(indices)[None]
        for block in self.blocks:
            x = block(x)
        out = self.heads(self.final_norm(x))
        out.update(balance=obs.new_zeros(()), halt_probability=obs.new_zeros(()), expert_routes=[], did_refine=False)
        return out


class EpisodicMemory:
    """Versioned external facts; only caller-verified past observations are stored."""
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.items: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def write_verified(self, world_version: int, key: torch.Tensor, value: torch.Tensor):
        self.items.append((world_version, key.detach().cpu().flatten(), value.detach().cpu().flatten()))
        self.items = self.items[-self.capacity:]

    def read(self, world_version: int, query: torch.Tensor):
        eligible = [(k, v) for version, k, v in self.items if version == world_version]
        if not eligible:
            return None
        q = query.detach().cpu().flatten()
        score = [F.cosine_similarity(q[None], k[None]).item() for k, _ in eligible]
        return eligible[max(range(len(score)), key=lambda i: score[i])][1].clone()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def prediction_loss(outputs, targets):
    error = outputs["prediction"] - targets
    mse = error.square().mean()
    gaussian = 0.5 * (outputs["log_variance"] + error.square() * (-outputs["log_variance"]).exp()).mean()
    calibrated = (outputs["verified_error"] - error.detach().square().mean(-1, keepdim=True)).square().mean()
    return mse + .05 * gaussian + .05 * calibrated + .01 * outputs["balance"], mse.detach()


def generate_world_batch(batch: int, seq_len: int, seed: int, mode: str = "stable", device="cpu"):
    """Causal per-episode dynamical systems; targets are future observations only."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=generator)
    dynamics = rnd(batch, 3, 3) * .28 + .48 * torch.eye(3)[None]
    alternate = rnd(batch, 3, 3) * .28 + .48 * torch.eye(3)[None]
    controls = rnd(batch, 4, 3) * .30
    latent = rnd(batch, 3) * .35
    observations, actions, targets = [], [], []
    for t in range(seq_len):
        current = latent
        choice = torch.randint(0, 4, (batch,), generator=generator)
        action = F.one_hot(choice, 4).float()
        matrix = alternate if mode == "switch" and t >= seq_len // 2 else dynamics
        proposal = torch.bmm(current[:, None, :], matrix).squeeze(1) + torch.bmm(action[:, None, :], controls).squeeze(1)
        if mode == "nonlinear":
            proposal = proposal + .38 * current.roll(1, dims=-1) * current.roll(2, dims=-1)
        next_state = .60 * current + .40 * torch.tanh(proposal)
        observations.append(current)
        actions.append(action)
        targets.append(next_state)
        latent = next_state
    return torch.stack(observations, 1).to(device), torch.stack(actions, 1).to(device), torch.stack(targets, 1).to(device)
