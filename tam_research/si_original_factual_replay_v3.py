"""Factual sensor-level replay for the user's ORIGINAL integrated SI architecture.

Engineering-only successor to V2. The V1 replay trainer supervised only
DETACHED frozen latent vectors, so its loss could not update the modality
encoders, original attention/recurrent/sparse-MoE backbone or recurrent
reasoning. V3 re-encodes complete PUBLIC observed before/after histories
with live gradients while using the actually executed action and reward.

Not a scientific experiment, online actor-critic solution, trusted receipt,
full joint policy/search training, benchmark, or demonstrated SI advance.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tam_research.si_original_integrated_v1 import KINDS, PROVENANCE
from tam_research.si_original_world_scoped_v2 import SIOriginalWorldScopedV2


@dataclass(frozen=True)
class PublicSensoryTransition:
    world: str
    before: tuple[tuple[str, torch.Tensor], ...]
    action: int
    reward: float
    after: tuple[tuple[str, torch.Tensor], ...]
    provenance: str


def _copy_public(
    seq: Sequence[tuple[str, torch.Tensor]],
) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple((kind, value.detach().cpu().clone()) for kind, value in seq)


class SIOriginalFactualReplayV3(SIOriginalWorldScopedV2):
    """Sensor-level experience stream for supervised world-model gradients.

    This builds on the original interleaved SIHybridBlock and V2 per-world
    rank-2 state. Raw histories are in RAM, and are deliberately NOT part of
    V2's persistent state schema. Model usage is sequential, never concurrent.
    """

    def __init__(self, *, sensory_replay_capacity: int = 32,
                 **kwargs: Any) -> None:
        if (type(sensory_replay_capacity) is not int or
                not 1 <= sensory_replay_capacity <= 128):
            raise ValueError("bounded factual sensory-replay capacity required")
        super().__init__(**kwargs)
        self.sensory_replay: deque[PublicSensoryTransition] = deque(
            maxlen=sensory_replay_capacity)

    @torch.no_grad()
    def ingest_executed(
        self, before: Sequence[tuple[str, torch.Tensor]],
        action: int, reward: float,
        after: Sequence[tuple[str, torch.Tensor]], *,
        world: str, skill: str | None = None,
        provenance: str = PROVENANCE,
    ) -> None:
        # Parent validates exact history prefix, a single witnessed next
        # public observation, action range, finite reward and caller claims;
        # only THEN can it be saved as a bounded raw trajectory record.
        super().ingest_executed(
            before, action, reward, after,
            world=world, skill=skill, provenance=provenance)
        self.sensory_replay.append(PublicSensoryTransition(
            world=world, before=_copy_public(before), action=action,
            reward=float(reward), after=_copy_public(after),
            provenance=provenance))

    def train_sensor_world_replay(
        self, optimizer: torch.optim.Optimizer, *, world: str,
        count: int = 1,
    ) -> dict[str, Any]:
        """Train encoders/backbone/reasoner + factual predictive heads.

        The fixed objective is action-conditioned SINGLE-STEP next public
        representation and actual reward. Re-encodes past raw observations,
        with stop-gradient AFTER observation target. The direct action head
        is intentionally NOT trained as though an executed action were an
        optimal expert label. Learned search policy is NOT trained here.
        Returns engineering metrics; no scientific score or reward advantage.
        """
        if (type(world) is not str or not world or
                type(count) is not int or not 1 <= count <= 16):
            raise ValueError("valid named world and bounded replay count required")
        observed = [t for t in self.sensory_replay if t.world == world]
        if len(observed) < count:
            raise ValueError("insufficient own-world executed public replay")
        chosen = observed[-count:]
        device = next(self.parameters()).device

        self.train()
        optimizer.zero_grad(set_to_none=True)
        terms: list[torch.Tensor] = []
        pred_mse: list[float] = []
        pred_reward_mse: list[float] = []
        latent_steps: list[int] = []
        for experience in chosen:
            assert experience.provenance == PROVENANCE
            assert experience.world == world
            assert math.isfinite(experience.reward)
            prior_seq = [(kind, value.to(device)) for kind, value
                         in experience.before]
            later_seq = [(kind, value.to(device)) for kind, value
                         in experience.after]
            prior, balance, _ = self._backbone(prior_seq)
            # Planning forward uses _reason() before _world_step().
            # Learning must train EXACTLY that same latent transition path.
            reasoned, iterations = self._reason(
                prior, steps_limit=self.max_steps)
            latent_steps.append(int(iterations[0]))
            with torch.no_grad():
                real_future, _, _ = self._backbone(later_seq)
                real_future, _ = self._reason(
                    real_future, steps_limit=self.max_steps)
            future_pred, reward_pred = self._world_step(
                reasoned, experience.action)
            reward = torch.tensor(
                [experience.reward], dtype=reward_pred.dtype, device=device)
            next_mse = (future_pred - real_future).square().mean()
            reward_mse = F.mse_loss(reward_pred, reward)
            # One-step factual reward is NOT a long-horizon value teacher.
            value_mse = F.mse_loss(
                self.value_head(reasoned).flatten(), reward)
            # Verifier gates actual search from its BEFORE-action state.
            error_target = next_mse.detach().clamp(0, 1)
            predicted_uncertainty = torch.sigmoid(
                self.verifier(reasoned)).flatten()
            uncertainty_loss = (predicted_uncertainty - error_target).square().mean()
            # Discrete physical halting receives an auxiliary supervised
            # target from actual prediction error; no straight-through
            # gradient or fictional counterfactual action is introduced.
            halt_target = (next_mse.detach() < 0.05).float().expand_as(
                predicted_uncertainty)
            halt_logit = self.halt(reasoned).flatten() + self.halt_bias
            halt_loss = F.binary_cross_entropy_with_logits(
                halt_logit, halt_target)
            objective = (next_mse + reward_mse + 0.25 * value_mse +
                         0.1 * uncertainty_loss + 0.05 * halt_loss +
                         0.001 * balance)
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("nonfinite public sensory replay objective")
            terms.append(objective)
            pred_mse.append(float(next_mse.detach()))
            pred_reward_mse.append(float(reward_mse.detach()))
        total = torch.stack(terms).mean()
        total.backward()
        encoder_gradient_norms = {}
        for kind in KINDS:
            encoder = getattr(self, f"{kind}_encoder")
            encoder_gradient_norms[kind] = float(sum(
                p.grad.detach().abs().sum().item()
                for p in encoder.parameters() if p.grad is not None))
        hybrid_gradient_norm = float(sum(
            p.grad.detach().abs().sum().item()
            for p in self.blocks.parameters() if p.grad is not None))
        reasoning_gradient_norm = float(sum(
            p.grad.detach().abs().sum().item()
            for p in self.latent_step.parameters() if p.grad is not None))
        gradnorm = nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        if not bool(torch.isfinite(gradnorm)):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("nonfinite joint replay gradients")
        optimizer.step()
        return {
            "status": "DEVELOPMENT_ONLY_FACTUAL_WORLD_REPLAY",
            "world": world, "executed_experiences": count,
            "objective": float(total.detach()),
            "next_representation_mse": sum(pred_mse) / count,
            "executed_reward_mse": sum(pred_reward_mse) / count,
            "backbone_gradient_l1": hybrid_gradient_norm,
            "latent_reasoner_gradient_l1": reasoning_gradient_norm,
            "encoder_gradient_l1": encoder_gradient_norms,
            "latent_steps_per_sample": latent_steps,
            "direct_policy_trained": False,
            "long_horizon_search_trained": False,
            "scientific_gate_passed": False,
            "paid_compute_used": False,
        }
