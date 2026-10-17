"""PCMA-v1 factual learning / baseline engineering. NOT a scientific experiment.

Only executed factual transitions supply targets. The neural reward predictors
are offline-pretrained on such data and frozen during evaluation; all cross-
session updates are confined to the explicit bounded PCMA adapter/memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from tam_research.si_original_hybrid_v0 import OriginalSIHybrid, SIConfig, TransformerControl
from tam_research.si_pcma_v1 import (
    FactualTransition, PersistentWorld, PublicObservation, PCMAState,
    WorldConfig, FactualMemory,
)


@dataclass(frozen=True)
class CausalRewardSample:
    """One label for an action ACTUALLY executed, never an unchosen candidate."""
    observation: PublicObservation
    action: int
    prior: tuple[FactualTransition, ...]
    observed_net_reward: float
    logical_time: int


def causal_samples(rows: Sequence[FactualTransition]) -> tuple[CausalRewardSample, ...]:
    """Construct training examples with only strictly earlier in-session history.

    Input rows must be an ordered actual rollout. End-of-session feedback may
    train a target but cannot enter the next session's short context.
    """
    samples: list[CausalRewardSample] = []
    history: list[FactualTransition] = []
    previous_time = 0
    for e in rows:
        if e.time <= previous_time:
            raise ValueError("nonmonotonic or duplicate factual time")
        if not (0 <= e.action < 4 and 0 <= e.before.symbol < 4):
            raise ValueError("bad public record")
        if history and history[-1].done:
            history.clear()
        samples.append(CausalRewardSample(
            e.before, e.action, tuple(history), e.reward - e.fee, e.time
        ))
        history.append(e)
        previous_time = e.time
    return tuple(samples)


def causal_tokens(
    observation: PublicObservation,
    action: int,
    prior: Sequence[FactualTransition],
    *,
    logical_time: int | None = None,
    max_context: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode public history and an unlabeled query; target NEVER enters query.

    Previous observed rewards are legitimate feedback. Unchosen counterfactual
    rewards, latent regimes and current/future query rewards cannot be inputs.
    """
    if observation.symbol not in range(4) or action not in range(4):
        raise ValueError("invalid public observation or action")
    if not 1 <= max_context <= 63:
        raise ValueError("bad causal context length")
    history = tuple(prior)[-(max_context - 1):] if max_context > 1 else ()
    if logical_time is not None and any(e.time >= logical_time for e in history):
        raise ValueError("current or future outcome in causal context")
    if any(history[i].time >= history[i + 1].time for i in range(len(history) - 1)):
        raise ValueError("unordered public history")
    features: list[list[float]] = []
    action_rows: list[list[float]] = []
    modalities: list[int] = []
    for event in history:
        features.append([
            event.before.symbol / 3.0,
            event.before.steps_remaining / 16.0,
            float(event.reward - event.fee),
        ])
        action_rows.append([float(i == event.action) for i in range(4)])
        modalities.append(0)
    # The target is not an argument and the query reward channel stays blank.
    features.append([observation.symbol / 3.0,
                     observation.steps_remaining / 16.0, 0.0])
    action_rows.append([float(i == action) for i in range(4)])
    modalities.append(1)
    return (
        torch.tensor([features], dtype=torch.float32),
        torch.tensor([action_rows], dtype=torch.float32),
        torch.tensor([modalities], dtype=torch.long),
    )


class FactualRewardPredictor(nn.Module):
    """Identical reward head, modality semantics and causal inputs by backbone."""
    def __init__(self, kind: Literal["hybrid", "transformer"]):
        super().__init__()
        if kind not in ("hybrid", "transformer"):
            raise ValueError("unsupported backbone")
        self.kind = kind
        cfg = SIConfig()  # Canonical original config; original source unchanged.
        self.backbone = OriginalSIHybrid(cfg) if kind == "hybrid" else TransformerControl(cfg)
        self.reward_head = nn.Linear(cfg.obs_dim, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor,
                modality: torch.Tensor) -> torch.Tensor:
        base = self.backbone(obs, actions, modality=modality)
        return self.reward_head(base["prediction"][:, -1]).flatten()

    def predict_sample(self, sample: CausalRewardSample) -> torch.Tensor:
        obs, action, mode = causal_tokens(
            sample.observation, sample.action, sample.prior,
            logical_time=sample.logical_time,
        )
        return self(obs, action, mode)


def fit_factual_predictor(
    predictor: FactualRewardPredictor,
    samples: Sequence[CausalRewardSample],
    *,
    updates: int = 2,
    lr: float = 0.001,
) -> tuple[float, ...]:
    """Tiny bounded CPU DEVELOPER routine. No heldout test or scientific seeds.

    Deterministic pass through already-observed factual rows only; no oracle
    dataset, counterfactual labels or privileged simulator-state access.
    """
    if not 1 <= updates <= 32 or not (0 < lr <= 0.01) or not samples:
        raise ValueError("invalid engineering fit configuration")
    predictor.train()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    for i in range(updates):
        sample = samples[i % len(samples)]
        optimizer.zero_grad(set_to_none=True)
        predicted = predictor.predict_sample(sample)
        target = torch.tensor([sample.observed_net_reward], dtype=predicted.dtype)
        loss = F.mse_loss(predicted, target)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    predictor.eval()
    return tuple(losses)


class FrozenNeuralFactualPolicy:
    """Frozen offline backbone plus optionally persistent shared PCMA state.

    The backbone is NEVER updated after deployment. Online learning is only via
    the budgeted PCMAState.adapter with feedback from executed actions.
    """
    def __init__(
        self, predictor: FactualRewardPredictor, *,
        memory_enabled: bool = True,
        adaptation_enabled: bool = True,
        pcma_weight: float = 0.5,
    ):
        if not (0 <= pcma_weight <= 1):
            raise ValueError("pcma_weight must lie in [0,1]")
        self.predictor = predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.state = PCMAState(memory_enabled=memory_enabled,
                               adaptation_enabled=adaptation_enabled)
        self.pcma_weight = pcma_weight
        self.history: list[FactualTransition] = []

    @torch.no_grad()
    def action_scores(self, observation: PublicObservation) -> tuple[float, ...]:
        scores = []
        for action in range(4):
            obs, acts, mode = causal_tokens(observation, action, self.history)
            neural_score = float(self.predictor(obs, acts, mode)[0])
            # Same combination and explicitly identical external storage for
            # hybrid and Transformer; simple non-learned fusion at this stage.
            memory_score = self.state.value(observation, action)
            scores.append(
                (1 - self.pcma_weight) * neural_score +
                self.pcma_weight * memory_score
            )
        return tuple(scores)

    def choose(self, observation: PublicObservation) -> int:
        scores = self.action_scores(observation)
        return max(range(4), key=lambda i: scores[i])

    def observe(self, event: FactualTransition) -> None:
        self.state.observe(event)
        self.history.append(event)

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        self.history.clear()
        self.state.restart_session(erase_persistent=erase_persistent)


class ExactMatchValuePolicy:
    """Nonprivileged Beta-smoothed value lookup: cheap v26-style comparator."""
    def __init__(self, capacity: int = 128):
        self.memory = FactualMemory(capacity)

    def choose(self, observation: PublicObservation) -> int:
        scores = []
        for action in range(4):
            rows = self.memory.retrieve(observation, action)
            successes = sum(e.reward for e in rows)
            prior_mean = (1.0 + successes) / (2.0 + len(rows))
            # Probe has known public charge and no immediate reward.
            scores.append(prior_mean - (0.1 if action == 3 else 0.0))
        return max(range(4), key=lambda i: scores[i])

    def observe(self, event: FactualTransition) -> None:
        self.memory.record(event)

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if erase_persistent:
            self.memory = FactualMemory(self.memory.capacity)


class OnlineTabularTDPolicy:
    """Nonprivileged factual TD(0) baseline, with optional cross-session reset."""
    def __init__(self, alpha: float = 0.2, discount: float = 0.8):
        if not (0 < alpha <= 1 and 0 <= discount < 1):
            raise ValueError("invalid TD settings")
        self.alpha, self.discount = alpha, discount
        self.q: dict[tuple[int, int], float] = {}

    def choose(self, observation: PublicObservation) -> int:
        return max(range(4), key=lambda a: self.q.get((observation.symbol, a), 0.0))

    def observe(self, event: FactualTransition) -> None:
        key = (event.before.symbol, event.action)
        previous = self.q.get(key, 0.0)
        future = 0.0 if event.done else max(
            self.q.get((event.after.symbol, a), 0.0) for a in range(4)
        )
        target = event.reward - event.fee + self.discount * future
        self.q[key] = previous + self.alpha * (target - previous)

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if erase_persistent:
            self.q.clear()


def collect_factual_rollouts(
    configs: Sequence[WorldConfig],
    *,
    exploration_seed: int,
    sessions: int = 2,
) -> tuple[FactualTransition, ...]:
    """Collect rewards from executed actions only, never from latent state."""
    if sessions < 1 or not configs:
        raise ValueError("empty collection plan")
    rng = random.Random(exploration_seed)
    records: list[FactualTransition] = []
    for i, config in enumerate(configs):
        world = PersistentWorld(config)
        for session in range(sessions):
            if session:
                world.restart_session()
            for _ in range(config.max_steps):
                event = world.step(rng.randrange(4))
                # Namespaced provenance and time, so worlds cannot falsely
                # collide if training logs later share one factual store.
                records.append(FactualTransition(
                    event.before, event.action, event.after, event.reward,
                    event.fee, event.time + i * 1_000_000,
                    f"world-{i}/{event.provenance}", event.done,
                ))
    return tuple(records)


def run_policy_sessions(policy, config: WorldConfig, *, sessions: int = 3,
                        erase_at_restart: bool = False) -> tuple[tuple[FactualTransition, ...], ...]:
    """Each policy interacts with its OWN world; no shared fake outcomes.

    This utility intentionally returns FACTUAL trajectories, not CI benchmark
    scores or scientific significance. World seed equality couples initial
    settings but differing actions necessarily create differing trajectories.
    """
    if sessions < 1:
        raise ValueError("sessions must be positive")
    world = PersistentWorld(config)
    out = []
    for session in range(sessions):
        if session:
            world.restart_session()
            policy.restart_session(erase_persistent=erase_at_restart)
        events = []
        for _ in range(config.max_steps):
            action = policy.choose(world.public_observation())
            event = world.step(action)
            policy.observe(event)
            events.append(event)
        out.append(tuple(events))
    return tuple(out)
