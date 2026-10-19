"""Public-factual online memory/adapter ablation on published POPGym.

EXPLORATORY CPU DEVELOPMENT ONLY. The original SI hybrid and Transformer
backbones remain byte-identical and are first fitted with the EXACT bounded
48-step public-history imitation recipe of PR #1517. Each frozen model is
then deployed with the same optional online memory and rank-2 reward adapter.

This is a PORT of the PCMA principle, NOT the original 3-symbol/4-action
TwoStepPCMAState: POPGym has up to 27 actions and does not offer paid probes.
It cannot establish original-PCMA persistent-cross-restart, causal probe or
SI architecture superiority. Exact classical public memory earns +1.0.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math

import torch

from tam_research.si_pcma_v1_published_memory import (
    TASKS, _make_published_task, _task_observation,
)
from tam_research.si_pcma_v1_popgym_neural_pilot import (
    ACTION_DIM, CPU_THREADS, INITIALIZATION_FIXTURE, PUBLIC_EPISODE_STEPS,
    TRAIN_DEVELOPMENT_EPISODES, TRAINING_UPDATES,
    VALIDATION_DEVELOPMENT_EPISODES, OriginalWorldHeadMemoryReadout,
    PublicSequence, _mask_class_logits, _sequence_tensors, _training_loss,
    collect_public_sequences,
)

ABLATIONS = ("none", "memory_only", "adapter_only", "both")
MAX_FACTUAL_MEMORY = 128
MAX_ADAPTER_UPDATES = 8
PCMA_BLEND = 0.5
FEATURE_WIDTH = 9
# External POPGym 1.0.7 reward normalization:
# RepeatPrevious(k=4) rewards 52-4=48 steps, others reward 51.
# This is published simulator accounting, not a fitted hyperparameter.
ACTUAL_REWARD_DENOMINATOR = {
    "RepeatFirstEasy": 51,
    "RepeatPreviousEasy": 48,
    "CountRecallEasy": 51,
}


@dataclass(frozen=True)
class ExecutedPublicFeedback:
    task: str
    public: int | tuple[int, int]
    chosen_action: int
    observed_reward: float
    step: int


class PublicFactualRank2State:
    """One episode's actually observed public reward record, not hidden state.

    Updates are bounded; unused-arm values are never looked up from the
    simulator. Records clear at independent POPGym episode boundaries; there
    is NO pretense of cross-restart persistent world identity in POPGym.
    """

    def __init__(self, *, memory_enabled: bool, adaptation_enabled: bool):
        if type(memory_enabled) is not bool or type(adaptation_enabled) is not bool:
            raise ValueError("explicit state ablation required")
        self.memory_enabled = memory_enabled
        self.adaptation_enabled = adaptation_enabled
        self.memory: deque[ExecutedPublicFeedback] = deque(maxlen=MAX_FACTUAL_MEMORY)
        self.u = tuple(tuple(0.05 if (j+i) % 2 == 0 else -0.025
                             for j in range(FEATURE_WIDTH)) for i in range(2))
        self.v = [0.0, 0.0]
        self.updates = 0
        self.observations = 0

    @staticmethod
    def features(
        task: str, public: int | tuple[int, int],
        action: int, step: int,
    ) -> tuple[float, ...]:
        if task not in TASKS or type(action) is not int or not 0 <= action < ACTION_DIM:
            raise ValueError("invalid public task/action query")
        if type(step) is not int or not 0 <= step < PUBLIC_EPISODE_STEPS:
            raise ValueError("future/out-of-bounds public time")
        if task == "CountRecallEasy":
            if (type(public) is not tuple or len(public) != 2
                    or any(type(i) is not int or i not in range(2)
                           for i in public)):
                raise ValueError("malformed public color observation")
            a, b = public
        else:
            if type(public) is not int or public not in range(4):
                raise ValueError("malformed public suit observation")
            a, b = public / 3.0, 0.0
        return tuple(float(task == t) for t in TASKS) + (
            float(a), float(b), action / 26.0,
            step / float(PUBLIC_EPISODE_STEPS),
            float(action < 4), 1.0,
        )

    def _adapter_value(self, features: tuple[float, ...]) -> float:
        return sum(weight * sum(a*x for a, x in zip(row, features))
                   for row, weight in zip(self.u, self.v))

    def correction(
        self, task: str, public: int | tuple[int, int],
        action: int, step: int,
    ) -> float:
        x = self.features(task, public, action, step)
        matches = [e.observed_reward * ACTUAL_REWARD_DENOMINATOR[task]
                   for e in reversed(self.memory)
                   if e.task == task and e.public == public and
                   e.chosen_action == action and e.step < step]
        observed = sum(matches[:8]) / len(matches[:8]) if matches else 0.0
        adapted = self._adapter_value(x) if self.adaptation_enabled else 0.0
        return PCMA_BLEND * (
            (observed if self.memory_enabled else 0.0) + adapted)

    def record(
        self, task: str, public: int | tuple[int, int], action: int,
        reward: float, step: int,
    ) -> None:
        features = self.features(task, public, action, step)
        if (type(reward) not in (float, int) or not math.isfinite(reward)
                or abs(reward * ACTUAL_REWARD_DENOMINATOR[task]) > 1.01
                or step != self.observations):
            raise ValueError("only chronological bounded executed reward allowed")
        if self.memory_enabled:
            self.memory.append(ExecutedPublicFeedback(
                task, public, action, float(reward), step))
        if self.adaptation_enabled and self.updates < MAX_ADAPTER_UPDATES:
            z = [sum(a*x for a, x in zip(row, features)) for row in self.u]
            prediction = sum(w*zi for w, zi in zip(self.v, z))
            delta = max(-1.0, min(1.0,
                       reward * ACTUAL_REWARD_DENOMINATOR[task] - prediction))
            old_v = self.v[:]
            old_u = self.u
            self.v = [max(-1., min(1., vi+0.05*delta*zi))
                      for vi, zi in zip(self.v, z)]
            self.u = tuple(tuple(max(-1., min(1.,
                           old_u[i][j] + 0.05*delta*old_v[i]*features[j]))
                           for j in range(FEATURE_WIDTH)) for i in range(2))
            self.updates += 1
        self.observations += 1


@dataclass(frozen=True)
class AblationTaskReadout:
    task: str
    backbone: str
    ablation: str
    development_episodes: int
    mean_executed_return: float
    mean_expert_action_accuracy: float
    observed_actions: int
    memory_records: int
    adapter_updates: int


@dataclass(frozen=True)
class PCMAPortAblationReadout:
    status: str
    original_v26: str
    pcma_v1_architecture: str
    trained_original_backbones: tuple[str, ...]
    training_updates_each: int
    training_episodes_each: int
    development_validation_episodes_each: int
    classical_public_memory_ceiling: float
    results: tuple[AblationTaskReadout, ...]
    scientific_seeds_consumed: int
    heldout_episodes: int
    active_flops_matched: bool
    paid_accelerator_used: bool
    permitted_architectural_claim: bool


def _train_original_frozen(kind: str) -> OriginalWorldHeadMemoryReadout:
    """EXACT same 48-update CPU supervised method as earlier parent PR."""
    if kind not in ("hybrid", "transformer"):
        raise ValueError("original source backbone required")
    rows = {task: collect_public_sequences(task, TRAIN_DEVELOPMENT_EPISODES)
            for task in TASKS}
    torch.manual_seed(INITIALIZATION_FIXTURE)
    model = OriginalWorldHeadMemoryReadout(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0.)
    model.train()
    for i in range(TRAINING_UPDATES):
        batch = tuple(rows[task][i % len(TRAIN_DEVELOPMENT_EPISODES)]
                      for task in TASKS)
        optimizer.zero_grad(set_to_none=True)
        loss = _training_loss(model, batch)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite public-history imitation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _run_one(
    model: OriginalWorldHeadMemoryReadout, seq: PublicSequence,
    *, memory_enabled: bool, adaptation_enabled: bool,
) -> tuple[float, float, int, int, int]:
    state = PublicFactualRank2State(
        memory_enabled=memory_enabled, adaptation_enabled=adaptation_enabled)
    input_obs, actions, modes, targets, _valid, classes = (
        _sequence_tensors((seq,)))
    with torch.no_grad():
        logits = _mask_class_logits(model(input_obs, actions, modes), classes)[0]
    environment = _make_published_task(seq.task)
    reward_total = 0.
    matched, eligible = 0, 0
    try:
        obs, _ = environment.reset(seed=seq.episode_id)
        for step in range(PUBLIC_EPISODE_STEPS):
            public = _task_observation(seq.task, obs)
            if public != seq.inputs[step]:
                raise RuntimeError("untrusted action-dependent POPGym trajectory")
            action_count = seq.action_count
            base = logits[step]
            if memory_enabled or adaptation_enabled:
                candidates = [
                    float(base[action]) + state.correction(
                        seq.task, public, action, step)
                    for action in range(action_count)
                ]
                chosen = max(range(action_count), key=lambda a: candidates[a])
            else:
                chosen = int(base[:action_count].argmax())
            if not environment.action_space.contains(chosen):
                raise RuntimeError("illegal neural action")
            if seq.valid_steps[step]:
                matched += int(chosen == seq.expert_actions[step])
                eligible += 1
            obs, reward, done, truncated, _ = environment.step(chosen)
            state.record(seq.task, public, chosen, float(reward), step)
            reward_total += float(reward)
            if bool(done or truncated) != (step == PUBLIC_EPISODE_STEPS-1):
                raise RuntimeError("unexpected external world termination")
        if not eligible or not math.isfinite(reward_total):
            raise ValueError("invalid factual reward summary")
        return (
            reward_total, matched/eligible, state.observations,
            len(state.memory), state.updates,
        )
    finally:
        environment.close()


def run_frozen_public_factual_ablation() -> PCMAPortAblationReadout:
    prior = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    try:
        validation = {
            task: collect_public_sequences(task, VALIDATION_DEVELOPMENT_EPISODES)
            for task in TASKS
        }
        records = []
        for kind in ("hybrid", "transformer"):
            model = _train_original_frozen(kind)
            for ablation in ABLATIONS:
                memory = ablation in ("memory_only", "both")
                adapter = ablation in ("adapter_only", "both")
                for task in TASKS:
                    outcomes = [_run_one(model, seq,
                                       memory_enabled=memory,
                                       adaptation_enabled=adapter)
                                for seq in validation[task]]
                    records.append(AblationTaskReadout(
                        task=task, backbone=kind, ablation=ablation,
                        development_episodes=len(outcomes),
                        mean_executed_return=sum(x[0] for x in outcomes) / len(outcomes),
                        mean_expert_action_accuracy=sum(x[1] for x in outcomes) / len(outcomes),
                        observed_actions=sum(x[2] for x in outcomes),
                        memory_records=sum(x[3] for x in outcomes),
                        adapter_updates=sum(x[4] for x in outcomes),
                    ))
        return PCMAPortAblationReadout(
            status="EXPLORATORY_POPGYM_PUBLIC_FACTUAL_PCMA_PORT",
            original_v26="NO_GO", pcma_v1_architecture="UNKNOWN",
            trained_original_backbones=("hybrid", "transformer"),
            training_updates_each=TRAINING_UPDATES,
            training_episodes_each=len(TASKS)*len(TRAIN_DEVELOPMENT_EPISODES),
            development_validation_episodes_each=len(TASKS)*len(VALIDATION_DEVELOPMENT_EPISODES),
            classical_public_memory_ceiling=1.,
            results=tuple(records), scientific_seeds_consumed=0,
            heldout_episodes=0, active_flops_matched=False,
            paid_accelerator_used=False, permitted_architectural_claim=False,
        )
    finally:
        torch.set_num_threads(prior)


def main() -> None:
    print("SI PCMA PORT — FROZEN BOUNDED PUBLIC-FACTUAL ABLATIONS (DEV ONLY)", flush=True)
    print("NO PROBES; NO CROSS-EPISODE WORLD ID; NO SCIENTIFIC OR SI CLAIM", flush=True)
    report = run_frozen_public_factual_ablation()
    print(json.dumps(asdict(report), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
