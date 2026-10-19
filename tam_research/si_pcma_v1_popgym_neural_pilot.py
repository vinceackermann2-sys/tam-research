"""Bounded POPGym neural MEMORY pilot for original SI hybrid vs Transformer.

IMPORTANT: *development-only supervised imitation* of a transparent, cheap
public-history expert on unmodified, published POPGym environments.
NOT a general PCMA/agent scientific experiment, heldout run, seed reservation,
causal probe experiment, or an SI breakthrough. Original model code unchanged.

Hybrid and Transformer use exactly SIConfig()/original implementation, plus
identical inputs and a shared-shape 3->27 readout. Both train with the same
AdamW schedule, same action-independent public trajectories, same expert
history-derived targets, the same per-task update budget, and CPU-only.

The 3-D original world-head bottleneck is preserved and explicitly acknowledged
as an engineering limitation; this pilot does NOT match actual active FLOPs
or confirm architectural advantage. Exact classical public-memory policies
remain stronger, cheaper, and reach the external task reward ceiling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from tam_research.si_original_hybrid_v0 import (
    OriginalSIHybrid, SIConfig, TransformerControl, count_parameters,
)
from tam_research.si_pcma_v1_published_memory import (
    TASKS, PublicMemoryPolicy, _make_published_task, _task_observation,
)

TRAIN_DEVELOPMENT_EPISODES = tuple(range(101, 113))
VALIDATION_DEVELOPMENT_EPISODES = tuple(range(201, 209))
PUBLIC_EPISODE_STEPS = 51
ACTION_DIM = 27  # count recall has 0..26; suit recall has 0..3
TRAINING_UPDATES = 48
INITIALIZATION_FIXTURE = 70631
CPU_THREADS = 2


@dataclass(frozen=True)
class PublicSequence:
    task: str
    episode_id: int
    inputs: tuple[int | tuple[int, int], ...]
    expert_actions: tuple[int, ...]
    valid_steps: tuple[bool, ...]
    action_count: int


@dataclass(frozen=True)
class ModelDevelopmentResult:
    backbone: str
    parameters: int
    optimizer_updates: int
    development_training_episodes: int
    development_validation_episodes: int
    final_training_cross_entropy: float
    task_returns: tuple[tuple[str, float], ...]
    task_expert_action_accuracy: tuple[tuple[str, float], ...]
    active_training_flops_matched: bool
    scientific_result: bool


@dataclass(frozen=True)
class PilotReadout:
    status: str
    original_v26: str
    pcma_successor: str
    source_task_family: str
    train_episodes_per_task: int
    validation_episodes_per_task: int
    model_results: tuple[ModelDevelopmentResult, ...]
    strong_classical_public_memory_return: float
    heldout_episodes: int
    scientific_seeds_consumed: int
    paid_accelerator_used: bool
    may_claim_hybrid_advantage: bool


def collect_public_sequences(task: str, ids: tuple[int, ...]) -> tuple[PublicSequence, ...]:
    """Use only public observations, not the simulator state or `info`.

    The env action 0 is a fixed, legal logging action. These three published
    memory benchmarks deal cards independently of the chosen action; this
    passive-trajectory invariance is separately tested against other actions.
    All expert targets arise solely from the earlier public observation
    sequence, never from the hidden state or the environment's reward.
    """
    if task not in TASKS or (type(ids) is not tuple or not ids
            or any(type(x) is not int for x in ids)
            or len(set(ids)) != len(ids)
            or not set(ids) <= (set(TRAIN_DEVELOPMENT_EPISODES)
                               | set(VALIDATION_DEVELOPMENT_EPISODES))):
        raise ValueError("only predeclared development episodes permitted")
    rows: list[PublicSequence] = []
    for episode_id in ids:
        expert = PublicMemoryPolicy(task, memory_enabled=True)
        env = _make_published_task(task)
        try:
            obs, _ = env.reset(seed=episode_id)
            inputs: list[int | tuple[int, int]] = []
            actions: list[int] = []
            valid: list[bool] = []
            action_count = int(env.action_space.n)
            for t in range(PUBLIC_EPISODE_STEPS):
                public = _task_observation(task, obs)
                action = expert.choose(public)
                if not env.action_space.contains(action):
                    raise ValueError("public-memory expert gave invalid action")
                inputs.append(public)
                actions.append(action)
                valid.append(task != "RepeatPreviousEasy" or t >= 3)
                obs, _reward, terminated, truncated, _info = env.step(0)
                if bool(terminated or truncated) != (t == PUBLIC_EPISODE_STEPS - 1):
                    raise RuntimeError("published task length/termination changed")
            rows.append(PublicSequence(
                task, episode_id, tuple(inputs), tuple(actions),
                tuple(valid), action_count,
            ))
        finally:
            env.close()
    return tuple(rows)


def _sequence_tensors(rows: tuple[PublicSequence, ...]):
    if not rows or len({row.task for row in rows}) != len(rows):
        raise ValueError("expected one developer sample per task")
    inputs = torch.zeros(len(rows), PUBLIC_EPISODE_STEPS, 3)
    actions = torch.zeros(len(rows), PUBLIC_EPISODE_STEPS, 4)
    modes = torch.zeros(len(rows), PUBLIC_EPISODE_STEPS, dtype=torch.long)
    labels = torch.zeros(len(rows), PUBLIC_EPISODE_STEPS, dtype=torch.long)
    valid = torch.zeros(len(rows), PUBLIC_EPISODE_STEPS, dtype=torch.bool)
    class_valid = torch.zeros(len(rows), ACTION_DIM, dtype=torch.bool)
    for index, row in enumerate(rows):
        if (len(row.inputs) != PUBLIC_EPISODE_STEPS
                or len(row.expert_actions) != PUBLIC_EPISODE_STEPS
                or len(row.valid_steps) != PUBLIC_EPISODE_STEPS
                or row.action_count not in (4, ACTION_DIM)):
            raise ValueError("malformed fixed public sequence")
        mode = TASKS.index(row.task)
        class_valid[index, :row.action_count] = True
        modes[index, :] = mode
        for t, public in enumerate(row.inputs):
            if row.task == "CountRecallEasy":
                dealt, queried = public
                inputs[index, t, 0] = float(dealt)
                inputs[index, t, 1] = float(queried)
            else:
                inputs[index, t, 0] = float(public) / 3.0
            inputs[index, t, 2] = float(mode) / 2.0
            target = row.expert_actions[t]
            if type(target) is not int or not 0 <= target < row.action_count:
                raise ValueError("expert target not valid for public action space")
            labels[index, t] = target
            valid[index, t] = row.valid_steps[t]
    # The observation stream is action independent on these tasks. Zeroed
    # action embedding is identical in both architectures and prevents a
    # teacher-forced expert action becoming the next step's leaked label.
    return inputs, actions, modes, labels, valid, class_valid


class OriginalWorldHeadMemoryReadout(nn.Module):
    """Exact original backbone and same 3-dim world-head -> action readout."""

    def __init__(self, kind: Literal["hybrid", "transformer"]):
        super().__init__()
        if kind not in ("hybrid", "transformer"):
            raise ValueError("hybrid or Transformer only")
        self.kind = kind
        config = SIConfig()
        self.backbone = (OriginalSIHybrid(config)
                         if kind == "hybrid" else TransformerControl(config))
        self.readout = nn.Linear(config.obs_dim, ACTION_DIM)

    def forward(
        self, observations: torch.Tensor,
        actions: torch.Tensor, modalities: torch.Tensor,
    ) -> torch.Tensor:
        return self.readout(
            self.backbone(observations, actions, modality=modalities)["prediction"])


def _mask_class_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if (logits.ndim != 3 or logits.size(-1) != ACTION_DIM
            or allowed.shape != (logits.size(0), ACTION_DIM)):
        raise ValueError("invalid common neural prediction shape")
    return logits.masked_fill(~allowed[:, None, :], -1.0e6)


def _training_loss(model: nn.Module, rows: tuple[PublicSequence, ...]) -> torch.Tensor:
    inp, action, mode, target, valid, classes = _sequence_tensors(rows)
    logits = _mask_class_logits(model(inp, action, mode), classes)
    nll = F.cross_entropy(
        logits.flatten(0, 1), target.flatten(),
        reduction="none").reshape(len(rows), PUBLIC_EPISODE_STEPS)
    # Each of the three external tasks gets the SAME weight regardless of
    # early unrewarded previous-card actions.
    return torch.stack([
        nll[i][valid[i]].mean() for i in range(len(rows))
    ]).mean()


def _same_public_trajectory_real_reward(
    sequence: PublicSequence, decisions: tuple[int, ...],
) -> float:
    """Execute predicted actions to measure REAL POPGym reward.

    We verify at every step that the actually received observation equals
    the passive public trajectory used to form a causal, full-sequence forward
    prediction. If actions change world observations, abort instead of
    silently substituting offline labels for actual environment returns.
    """
    if len(decisions) != PUBLIC_EPISODE_STEPS:
        raise ValueError("incorrect action count")
    env = _make_published_task(sequence.task)
    reward_total = 0.0
    try:
        observation, _ = env.reset(seed=sequence.episode_id)
        for t, action in enumerate(decisions):
            if (_task_observation(sequence.task, observation) != sequence.inputs[t]
                    or not env.action_space.contains(action)):
                raise RuntimeError("action-dependent observation invalidates offline policy replay")
            observation, reward, terminated, truncated, _ = env.step(action)
            reward_total += float(reward)
            if bool(terminated or truncated) != (t == PUBLIC_EPISODE_STEPS - 1):
                raise RuntimeError("published task termination mismatch")
        if not math.isfinite(reward_total):
            raise ValueError("nonfinite unmodified POPGym return")
        return reward_total
    finally:
        env.close()


def run_bounded_development_training(
    *, updates: int = TRAINING_UPDATES,
    training_ids: tuple[int, ...] = TRAIN_DEVELOPMENT_EPISODES,
    validation_ids: tuple[int, ...] = VALIDATION_DEVELOPMENT_EPISODES,
) -> PilotReadout:
    if (type(updates) is not int or updates != TRAINING_UPDATES
            or training_ids != TRAIN_DEVELOPMENT_EPISODES
            or validation_ids != VALIDATION_DEVELOPMENT_EPISODES
            or set(training_ids) & set(validation_ids)):
        raise ValueError("frozen development CPU pilot configuration")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    try:
        training = {task: collect_public_sequences(task, training_ids)
                    for task in TASKS}
        validation = {task: collect_public_sequences(task, validation_ids)
                      for task in TASKS}
        outputs: list[ModelDevelopmentResult] = []
        for kind in ("hybrid", "transformer"):
            torch.manual_seed(INITIALIZATION_FIXTURE)
            model = OriginalWorldHeadMemoryReadout(kind)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.002,
                                          weight_decay=0.0)
            model.train()
            loss = 0.0
            for step in range(updates):
                # Exactly 48 updates * THREE equal-weight task trajectories.
                # Epoch-wise modulo rotation covers all 12 development episode
                # IDs per task with four passes; no validation sample is fitted.
                rows = tuple(training[task][step % len(training_ids)]
                             for task in TASKS)
                optimizer.zero_grad(set_to_none=True)
                current = _training_loss(model, rows)
                if not torch.isfinite(current):
                    raise RuntimeError("nonfinite development training objective")
                current.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss = float(current.detach())
            model.eval()
            returns = []
            accuracies = []
            with torch.no_grad():
                for task in TASKS:
                    results = []
                    matched = []
                    for row in validation[task]:
                        inputs, actions, modalities, labels, valid, classes = (
                            _sequence_tensors((row,)))
                        predicted = _mask_class_logits(
                            model(inputs, actions, modalities), classes
                        ).argmax(dim=-1)[0]
                        chosen = tuple(int(x) for x in predicted.tolist())
                        results.append(_same_public_trajectory_real_reward(row, chosen))
                        matched.append(
                            float(((predicted == labels[0]) & valid[0]).sum().item()) /
                            float(valid[0].sum().item()))
                    returns.append((task, sum(results) / len(results)))
                    accuracies.append((task, sum(matched) / len(matched)))
            outputs.append(ModelDevelopmentResult(
                backbone=kind, parameters=count_parameters(model),
                optimizer_updates=updates,
                development_training_episodes=len(training_ids) * len(TASKS),
                development_validation_episodes=len(validation_ids) * len(TASKS),
                final_training_cross_entropy=loss,
                task_returns=tuple(returns),
                task_expert_action_accuracy=tuple(accuracies),
                active_training_flops_matched=False,
                scientific_result=False,
            ))
        return PilotReadout(
            status="NONCONFIRMATORY_POPGYM_PUBLIC_MEMORY_IMITATION",
            original_v26="NO_GO", pcma_successor="UNKNOWN",
            source_task_family="independent-published-POPGym-1.0.7",
            train_episodes_per_task=len(training_ids),
            validation_episodes_per_task=len(validation_ids),
            model_results=tuple(outputs),
            strong_classical_public_memory_return=1.0,
            heldout_episodes=0, scientific_seeds_consumed=0,
            paid_accelerator_used=False, may_claim_hybrid_advantage=False,
        )
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    print("NONCONFIRMATORY PUBLISHED MEMORY NEURAL CPU DEVELOPMENT PILOT", flush=True)
    print("NO PCMA SUPERIORITY CLAIM, NO SCIENCE SEED, NO PAID GPU", flush=True)
    result = run_bounded_development_training()
    print(json.dumps(asdict(result), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
