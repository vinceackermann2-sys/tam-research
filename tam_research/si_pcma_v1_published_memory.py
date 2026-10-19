"""External POPGym memory-necessity controls (development only, not PCMA science).

Use the unmodified, independently published POPGym 1.0.7 environments:
RepeatFirstEasy, RepeatPreviousEasy(k=4), CountRecallEasy.
No hidden Markov state, `get_state`, simulator rules, oracle, extra Markovian
wrappers or per-action counterfactual rewards are given to a policy.

A trivially correct, task-specific *classical* memory policy is the positive
control; a current-observation-only policy is the negative control. These
controls show that the PUBLIC observation sequence has usable memory signal,
NOT that PCMA generalizes, learns or beats classical baselines. Do not regard
their scores as scientific confirmation or as a fair neural comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Literal


TASKS = ("RepeatFirstEasy", "RepeatPreviousEasy", "CountRecallEasy")
DEVELOPER_EPISODE_IDS = tuple(range(21, 29))
POPGYM_VERSION = "1.0.7"


def _task_observation(task: str, raw: Any) -> int | tuple[int, int]:
    """Copy out only public observation; do not pass Gym info or world."""
    if task == "CountRecallEasy":
        if len(raw) != 2:
            raise ValueError("invalid public count/query")
        dealt, queried = int(raw[0]), int(raw[1])
        if not (0 <= dealt < 2 and 0 <= queried < 2):
            raise ValueError("invalid public colors")
        return (dealt, queried)
    value = int(raw)
    if value not in range(4):
        raise ValueError("invalid public suit")
    return value


class PublicMemoryPolicy:
    """Deterministic exact public-memory reference. NOT a learned agent."""

    def __init__(self, task: str, *, memory_enabled: bool):
        if task not in TASKS or type(memory_enabled) is not bool:
            raise ValueError("unknown published benchmark control")
        self.task = task
        self.memory_enabled = memory_enabled
        self.first: int | None = None
        self.recent: list[int] = []
        self.counts = [0, 0]

    def choose(self, observation: int | tuple[int, int]) -> int:
        if self.task == "CountRecallEasy":
            if (type(observation) is not tuple or len(observation) != 2
                    or any(type(x) is not int or x not in (0, 1)
                           for x in observation)):
                raise ValueError("invalid count public input")
            dealt, queried = observation
            if self.memory_enabled:
                self.counts[dealt] += 1
                return self.counts[queried]
            return int(dealt == queried)  # Public current-only heuristic.
        if type(observation) is not int or observation not in range(4):
            raise ValueError("invalid suit public input")
        if not self.memory_enabled:
            return observation  # Current observation only.
        if self.task == "RepeatFirstEasy":
            if self.first is None:
                self.first = observation
            return self.first
        # Reward starts only when k=4 observations have arrived. Earlier
        # actions receive zero reward regardless of the choice.
        self.recent.append(observation)
        return self.recent[-4] if len(self.recent) >= 4 else 0


@dataclass(frozen=True)
class EpisodeRecord:
    task: str
    episode_id: int
    memory_enabled: bool
    decisions: int
    total_return: float
    return_finite: bool


@dataclass(frozen=True)
class TaskComparison:
    task: str
    episodes: int
    decisions_per_episode: int
    no_memory_mean_return: float
    exact_public_memory_mean_return: float
    memory_minus_no_memory: float
    memory_positive_control_pass: bool


@dataclass(frozen=True)
class PublishedMemoryDiagnostic:
    status: str
    package_version: str
    trainable_neural_models_evaluated: int
    scientific_seeds_reserved: int
    heldout_worlds_evaluated: int
    tasks: tuple[TaskComparison, ...]
    original_si_v26: str
    pcma_architecture_verdict: str
    may_claim_pcma_advantage: bool


def _make_published_task(task: str) -> Any:
    import popgym
    if popgym.__version__ != POPGYM_VERSION:
        raise RuntimeError("external task release changed")
    if task == "RepeatFirstEasy":
        return popgym.envs.RepeatFirstEasy()
    if task == "RepeatPreviousEasy":
        return popgym.envs.RepeatPreviousEasy(k=4)
    if task == "CountRecallEasy":
        return popgym.envs.CountRecallEasy()
    raise ValueError("only three predeclared memory-necessity tasks")


def run_public_episode(
    task: str, episode_id: int, *, memory_enabled: bool,
) -> EpisodeRecord:
    if (task not in TASKS or type(episode_id) is not int
            or episode_id not in DEVELOPER_EPISODE_IDS):
        raise ValueError("only predeclared development episode IDs allowed")
    policy = PublicMemoryPolicy(task, memory_enabled=memory_enabled)
    env = _make_published_task(task)
    try:
        observation, _info = env.reset(seed=episode_id)
        decisions = 0
        score = 0.0
        for _ in range(int(env.max_episode_length) + 1):
            public = _task_observation(task, observation)
            chosen_action = policy.choose(public)
            if not env.action_space.contains(chosen_action):
                raise RuntimeError("policy returned invalid public action")
            observation, reward, terminated, truncated, _info = env.step(chosen_action)
            decisions += 1
            score += float(reward)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("published environment violated bounded episode")
        if decisions != int(env.max_episode_length):
            raise RuntimeError("unexpected published task episode length")
        if not math.isfinite(score):
            raise RuntimeError("nonfinite published score")
        return EpisodeRecord(
            task, episode_id, memory_enabled, decisions, score, True)
    finally:
        env.close()


def published_memory_development_diagnostic() -> PublishedMemoryDiagnostic:
    """Fixed development episodes only, no model fitting and no heldout."""
    comparisons: list[TaskComparison] = []
    for task in TASKS:
        pairs = [
            (run_public_episode(task, episode_id, memory_enabled=False),
             run_public_episode(task, episode_id, memory_enabled=True))
            for episode_id in DEVELOPER_EPISODE_IDS
        ]
        without = sum(a.total_return for a, _ in pairs) / len(pairs)
        with_memory = sum(b.total_return for _, b in pairs) / len(pairs)
        if len({a.decisions for a, _ in pairs}) != 1:
            raise RuntimeError("different public action budgets")
        comparisons.append(TaskComparison(
            task=task, episodes=len(pairs),
            decisions_per_episode=pairs[0][0].decisions,
            no_memory_mean_return=without,
            exact_public_memory_mean_return=with_memory,
            memory_minus_no_memory=with_memory - without,
            memory_positive_control_pass=(
                with_memory >= 0.98 and with_memory - without >= 0.05),
        ))
    return PublishedMemoryDiagnostic(
        status="DEVELOPMENT_POSITIVE_CONTROL_ONLY",
        package_version=POPGYM_VERSION,
        trainable_neural_models_evaluated=0,
        scientific_seeds_reserved=0,
        heldout_worlds_evaluated=0,
        tasks=tuple(comparisons),
        original_si_v26="NO_GO",
        pcma_architecture_verdict="UNKNOWN",
        may_claim_pcma_advantage=False,
    )


def main() -> None:
    print("EXTERNALLY AUTHORED POPGYM MEMORY POSITIVE CONTROLS")
    print("NOT NEURAL PCMA / NOT A SCIENTIFIC RUN / NO ARCHITECTURE CLAIMS")
    result = published_memory_development_diagnostic()
    print(json.dumps(asdict(result), sort_keys=True, indent=2, allow_nan=False))
    if not all(row.memory_positive_control_pass for row in result.tasks):
        raise SystemExit("External task failed memory-discrimination preflight")


if __name__ == "__main__":
    main()
