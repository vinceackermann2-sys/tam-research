"""Original SI/PCMA published ten-arm neural adapter — CPU DEVELOPMENT ONLY.

Reuses EXACT OriginalSIHybrid and TransformerControl source, with a new
common ten-action interface. Both see identical causal public chosen-action
feedback; neither sees the POPGym privileged reset/step info containing true
arm probabilities, hidden world state, unexecuted returns or an oracle.

This is a tiny supervised **developer smoke**, not adequate pretraining,
full online reinforcement learning, scientific validation, matched-FLOPs
work, or evidence for SI. Strong Thompson sampling is a separate control.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from time import perf_counter
import json
import math
from typing import Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from tam_research.si_original_hybrid_v0 import (
    OriginalSIHybrid, TransformerControl, SIConfig, count_parameters,
)
from tam_research.si_pcma_v1_external_bandit import (
    ARMS, HORIZON, PACKAGE_VERSION, PublicBanditFeedback,
    FactualOnlineBanditPolicy,
)

Kind = Literal["hybrid", "transformer"]
ENGINEERING_TRAIN_EPISODES = (101, 102, 103, 104)
ENGINEERING_VALIDATION_EPISODES = (201, 202)
TOKEN_WINDOW = 8
BATCH_SIZE = 4
DEVELOPER_UPDATES = 4


@dataclass(frozen=True)
class PublicBanditExample:
    """Predict one executed arm payout using strictly earlier factual data."""
    previous: tuple[PublicBanditFeedback, ...]
    action: int
    step: int
    observed_payout: int


def _check_feedback(item: PublicBanditFeedback) -> None:
    if (type(item) is not PublicBanditFeedback
            or type(item.action) is not int or item.action not in range(ARMS)
            or type(item.observed_payout) is not int
            or item.observed_payout not in (0, 1)
            or type(item.step) is not int or not 1 <= item.step <= HORIZON
            or type(item.reward) is not float or not math.isfinite(item.reward)
            or abs(item.reward - (2*item.observed_payout-1)/HORIZON) > 1e-12):
        raise ValueError("invalid executed factual bandit event")


def bandit_tokens(
    prior: Sequence[PublicBanditFeedback], candidate: int, step: int,
    *, window: int = TOKEN_WINDOW,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same masked causal feature layout for BOTH original-SI backbones.

    Query reward=0, query modality=1. All history rewards are actual PUBLIC
    feedback. Query's correct/incorrect payout is never an input.
    """
    if (type(candidate) is not int or candidate not in range(ARMS)
            or type(step) is not int or not 1 <= step <= HORIZON
            or type(window) is not int or not 2 <= window <= 16):
        raise ValueError("invalid bounded public query")
    history = tuple(prior)
    # Reject future, duplicates and cross-session/noncausal feedback before
    # truncation, not only in the visible last few tokens.
    last = 0
    for event in history:
        _check_feedback(event)
        if event.step <= last or event.step >= step:
            raise ValueError("unordered, duplicated or future factual outcome")
        last = event.step
    if history and history[-1].step != step - 1:
        raise ValueError("missing immediate public feedback")
    history = history[-(window-1):]
    observations = [
        [float(e.observed_payout), (HORIZON-e.step)/HORIZON, 0.0]
        for e in history
    ] + [[0.0, (HORIZON-step)/HORIZON, 1.0]]
    actions = [
        [float(a == e.action) for a in range(ARMS)] for e in history
    ] + [[float(a == candidate) for a in range(ARMS)]]
    modality = [0]*len(history) + [1]
    return (
        torch.tensor([observations], dtype=torch.float32),
        torch.tensor([actions], dtype=torch.float32),
        torch.tensor([modality], dtype=torch.long),
    )


class BanditPublicPayoutPredictor(nn.Module):
    """Original source backbone, equal adapter, equal 10-arm token semantics."""

    def __init__(self, kind: Kind):
        super().__init__()
        if kind not in ("hybrid", "transformer"):
            raise ValueError("only original hybrid and Transformer comparator")
        self.kind = kind
        cfg = SIConfig(obs_dim=3, action_dim=ARMS, max_seq_len=16)
        self.backbone = (
            OriginalSIHybrid(cfg) if kind == "hybrid"
            else TransformerControl(cfg)
        )
        # Same dense payout prediction head, same objective, no hidden labels.
        self.payout_head = nn.Linear(cfg.obs_dim, 1)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor,
        modality: torch.Tensor,
    ) -> torch.Tensor:
        if (obs.ndim != 3 or actions.ndim != 3 or modality.ndim != 2
                or obs.shape[:2] != actions.shape[:2]
                or obs.shape[:2] != modality.shape
                or obs.shape[2] != 3 or actions.shape[2] != ARMS
                or not 1 <= obs.shape[1] <= 16):
            raise ValueError("incompatible common public token interface")
        output = self.backbone(obs, actions, modality=modality)
        return self.payout_head(output["prediction"][:, -1]).flatten()


def development_bandit_examples(
    episode_ids: Sequence[int],
) -> tuple[PublicBanditExample, ...]:
    """Collect once from the externally authored task; no hidden-state info.

    Fixed round-robin logging gives all arms observational support. No
    environmental oracle, offline simulated unchosen arm or policy-dependent
    test metric is synthesized. IDs are developer fixtures, NOT scientific
    seed reservations.
    """
    if (not episode_ids or any(
            type(i) is not int or i not in (
                ENGINEERING_TRAIN_EPISODES +
                ENGINEERING_VALIDATION_EPISODES) for i in episode_ids)
            or len(set(episode_ids)) != len(episode_ids)):
        raise ValueError("unapproved or duplicated developer episode IDs")
    if version("popgym") != PACKAGE_VERSION:
        raise RuntimeError("wrong external benchmark version")
    import popgym
    records: list[PublicBanditExample] = []
    for episode_id in episode_ids:
        world = popgym.envs.MultiarmedBanditEasy()
        policy = FactualOnlineBanditPolicy("round_robin")
        try:
            obs, _discard_privileged = world.reset(seed=episode_id)
            for step in range(1, HORIZON+1):
                candidate = policy.choose(int(obs))
                history = tuple(policy.history)
                obs, reward, terminated, truncated, _discard_privileged = (
                    world.step(candidate))
                payout = int(obs)
                event = PublicBanditFeedback(candidate, payout, float(reward),
                                             step)
                policy.observe(event)
                if step >= TOKEN_WINDOW and step % 5 == 0:
                    records.append(PublicBanditExample(
                        previous=history, action=candidate,
                        step=step, observed_payout=payout,
                    ))
                if (terminated or truncated) != (step == HORIZON):
                    raise RuntimeError("external episode boundary changed")
        finally:
            world.close()
    return tuple(records)


def stack_factual_batch(
    examples: Sequence[PublicBanditExample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("no chosen-arm developer targets")
    rows = []
    for e in examples:
        if (type(e) is not PublicBanditExample
                or type(e.observed_payout) is not int
                or e.observed_payout not in (0, 1)):
            raise ValueError("invalid developer factual target")
        token = bandit_tokens(e.previous, e.action, e.step)
        if token[0].shape[1] != TOKEN_WINDOW:
            raise ValueError("batch must have identical complete context")
        rows.append(token)
    return (
        torch.cat([x[0] for x in rows]),
        torch.cat([x[1] for x in rows]),
        torch.cat([x[2] for x in rows]),
        torch.tensor([float(e.observed_payout) for e in examples]),
    )


@dataclass(frozen=True)
class DevelopmentNeuralForecast:
    kind: str
    model_parameters: int
    optimizer_updates: int
    training_examples: int
    validation_examples: int
    last_training_bce: float
    factual_validation_bce: float
    training_cpu_wall_seconds: float
    matched_active_flops: bool = False


@dataclass(frozen=True)
class DevelopmentNeuralReadout:
    status: str
    fixed_training_episodes: tuple[int, ...]
    fixed_validation_episodes: tuple[int, ...]
    results: tuple[DevelopmentNeuralForecast, ...]
    reward_policy_evaluated: bool
    strong_thompson_comparison_run: bool
    science_seeds_reserved: int
    heldout_worlds_evaluated: int
    may_claim_architecture_advantage: bool


def development_neural_forecast_screen() -> DevelopmentNeuralReadout:
    """CPU-only bounded supervised smoke on IDENTICAL logged chosen arms.

    Not a fair capacity/FLOP-matched experiment. Does NOT evaluate policy
    return or compare neural value to Thompson's online action selection.
    A single tiny 4-update fit has no credible scientific interpretation.
    """
    torch.set_num_threads(min(2, torch.get_num_threads()))
    training = development_bandit_examples(ENGINEERING_TRAIN_EPISODES)
    validation = development_bandit_examples(
        ENGINEERING_VALIDATION_EPISODES)
    if not training or not validation:
        raise RuntimeError("no developer factual examples")
    valid_tensors = stack_factual_batch(validation)
    rows: list[DevelopmentNeuralForecast] = []
    for kind in ("hybrid", "transformer"):
        torch.manual_seed(901)  # Developer parameter initialization, NOT science.
        predictor = BanditPublicPayoutPredictor(kind)
        optimizer = torch.optim.AdamW(
            predictor.parameters(), lr=0.001, weight_decay=0.01)
        generator = torch.Generator().manual_seed(902)
        order = torch.randperm(len(training), generator=generator).tolist()
        losses: list[float] = []
        started = perf_counter()
        predictor.train()
        for i in range(DEVELOPER_UPDATES):
            batch = [
                training[order[(i*BATCH_SIZE+j) % len(order)]]
                for j in range(BATCH_SIZE)
            ]
            obs, acts, mode, target = stack_factual_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            predicted = predictor(obs, acts, mode)
            loss = F.binary_cross_entropy_with_logits(predicted, target)
            if not torch.isfinite(loss):
                raise RuntimeError("invalid developer supervised loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        wall = perf_counter()-started
        predictor.eval()
        with torch.no_grad():
            obs, acts, mode, target = valid_tensors
            val = F.binary_cross_entropy_with_logits(
                predictor(obs, acts, mode), target)
        if not torch.isfinite(val):
            raise RuntimeError("nonfinite developer validation loss")
        rows.append(DevelopmentNeuralForecast(
            kind=kind, model_parameters=count_parameters(predictor),
            optimizer_updates=DEVELOPER_UPDATES,
            training_examples=len(training),
            validation_examples=len(validation),
            last_training_bce=losses[-1],
            factual_validation_bce=float(val),
            training_cpu_wall_seconds=wall,
        ))
    return DevelopmentNeuralReadout(
        status="SMALL_CPU_DEVELOPMENT_FORECAST_ONLY",
        fixed_training_episodes=ENGINEERING_TRAIN_EPISODES,
        fixed_validation_episodes=ENGINEERING_VALIDATION_EPISODES,
        results=tuple(rows), reward_policy_evaluated=False,
        strong_thompson_comparison_run=False, science_seeds_reserved=0,
        heldout_worlds_evaluated=0, may_claim_architecture_advantage=False,
    )


def main() -> None:
    print("ORIGINAL SI HYBRID VS TRANSFORMER — TEN-ARM PUBLIC DATA SMOKE")
    print("CPU DEVELOPER ONLY; 4 UPDATES; NO POLICY RETURN / NO SCIENTIFIC VERDICT")
    report = development_neural_forecast_screen()
    print(json.dumps(asdict(report), sort_keys=True, indent=2,
                     allow_nan=False))


if __name__ == "__main__":
    main()
