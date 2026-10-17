"""PCMA-v1 factual two-step neural acquisition (developer-only, nonconfirmatory).

A neural first-action value is trained on ACTUALLY OBSERVED two-action
returns under a fixed public-information continuation; the 2nd action is
always chosen by the SAME frozen continuation. First-action labels from
randomized factual rollouts, never hidden rules or unchosen-arm outcomes.
The SI hybrid, Transformer and compact GRU receive identical causal tensors.
No scientific seeds, heldout training/evaluation, promotions or paid compute.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_factual_learning import (
    CausalRewardSample, FactualRewardPredictor, causal_tokens,
    fit_factual_predictor,
)
from tam_research.si_pcma_v1_fair_comparison import PCMAGRURewardPredictor
from tam_research.si_pcma_v1_probe_randomization import (
    FrozenPublicContinuation, RandomizedFactualPair,
    randomized_development_first_actions,
)
from tam_research.si_pcma_v1_ood_world import OODWorldConfig


def randomized_pairs_to_causal_samples(
    pairs: Sequence[RandomizedFactualPair],
) -> tuple[CausalRewardSample, ...]:
    """Current target = REALIZED two-step net; prior = older factual events.

    Reconstructs in-session history from previous paired actions. Future first
    and second rewards never enter the query. World/session ID is only an
    evaluator grouping key, NEVER fed to the predictor. Logical time is local
    to each world and is validated against its own prior.
    """
    if not pairs:
        raise ValueError("no development randomized pairs")
    samples: list[CausalRewardSample] = []
    prev_group: tuple[int, int] | None = None
    prev_pair_index = -1
    previous_second_time = 0
    prior: list[FactualTransition] = []
    seen: set[tuple[int, int, int]] = set()
    for pair in pairs:
        group = (pair.world_index, pair.session_index)
        key = (*group, pair.pair_index)
        if any(type(i) is not int or i < 0 for i in key) or key in seen:
            raise ValueError("invalid or duplicate randomized grouping")
        if prev_group != group:
            if prev_group is not None and group <= prev_group:
                raise ValueError("development groups not ordered")
            if pair.pair_index != 0:
                raise ValueError("new session must start at pair zero")
            prior.clear()
            previous_second_time = 0
            prev_pair_index = -1
        if pair.pair_index != prev_pair_index + 1:
            raise ValueError("missing or reordered factual pair")
        first, second = pair.first_event, pair.second_event
        if (first.before != pair.first_observation or
                first.action != pair.assigned_first_action or
                pair.assignment_probability != 0.25 or
                second.time != first.time + 1 or
                first.done or second.action == 3 or
                second.before != first.after or
                (previous_second_time and first.time != previous_second_time + 1)):
            raise ValueError("not an admissible randomized factual pair")
        if pair.continuation_actions[second.before.symbol] != second.action:
            raise ValueError("not the fixed factual continuation")
        samples.append(CausalRewardSample(
            observation=first.before,
            action=first.action,
            prior=tuple(prior),
            observed_net_reward=pair.observed_net,
            logical_time=first.time,
        ))
        prior.extend((first, second))
        previous_second_time = second.time
        seen.add(key)
        prev_group, prev_pair_index = group, pair.pair_index
    return tuple(samples)


@dataclass(frozen=True)
class ActualTwoStepRecord:
    """One completed first+second factual result, NO counterfactual fields."""
    before: PublicObservation
    first_action: int
    first_time: int
    observed_net: float
    first_provenance: str
    second_provenance: str


class TwoStepPCMAState:
    """Bounded factual pair memory plus rank-2 residual, probe-inclusive.

    This separate adapter is necessary because the original 1-step reward
    adapter purposely skips probes; it would never learn probe value.
    Identical mechanism/budget is shared by all neural architectures.
    """

    def __init__(
        self, *, memory_enabled: bool = True,
        adaptation_enabled: bool = True, capacity: int = 128,
        rank: int = 2, update_limit: int = 8, lr: float = 0.05,
    ):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("invalid memory capacity")
        if type(rank) is not int or rank < 1 or type(update_limit) is not int or update_limit < 0:
            raise ValueError("invalid adapter size/budget")
        if not (0 < lr <= 0.5):
            raise ValueError("invalid adapter learning rate")
        self.memory_enabled, self.adaptation_enabled = memory_enabled, adaptation_enabled
        self.capacity, self.rank, self.update_limit, self.lr = capacity, rank, update_limit, lr
        self.memory: deque[ActualTwoStepRecord] = deque(maxlen=capacity)
        self._seen_provenance: set[str] = set()
        self._last_time = 0
        self.u = [[0.1 if (j + i) % rank == 0 else -0.05 for j in range(8)]
                  for i in range(rank)]
        self.v = [0.0] * rank
        self.session_updates = 0
        self.total_updates = 0

    @staticmethod
    def features(obs: PublicObservation, action: int) -> list[float]:
        if type(obs) is not PublicObservation or obs.symbol not in range(3) or action not in range(4):
            raise ValueError("invalid public query")
        x = [0.0] * 8
        x[obs.symbol] = 1.0
        x[4 + action] = 1.0
        return x

    def _predict_adapter(self, obs: PublicObservation, action: int) -> float:
        x = self.features(obs, action)
        return sum(v * sum(a*b for a, b in zip(row, x))
                   for row, v in zip(self.u, self.v))

    def value(self, obs: PublicObservation, action: int) -> float:
        self.features(obs, action)
        rows = [r.observed_net for r in reversed(self.memory)
                if r.before.symbol == obs.symbol and r.first_action == action] \
               if self.memory_enabled else []
        pair_value = sum(rows[:8]) / min(8, len(rows)) if rows else 0.0
        adapted = self._predict_adapter(obs, action) if self.adaptation_enabled else 0.0
        denom = int(bool(rows)) + int(self.adaptation_enabled)
        return (pair_value + adapted) / denom if denom else 0.0

    def record(self, first: FactualTransition, second: FactualTransition) -> None:
        """Updates ONLY after actual second chosen-action feedback arrives."""
        if (type(first) is not FactualTransition or type(second) is not FactualTransition
                or first.done or second.time != first.time + 1
                or first.after != second.before or
                first.provenance == second.provenance):
            raise ValueError("invalid two-step factual feedback")
        if (first.time <= self._last_time or
                first.provenance in self._seen_provenance or
                second.provenance in self._seen_provenance):
            raise ValueError("duplicate/out-of-order factual pair")
        if (first.action not in range(4) or second.action not in range(3)
                or first.reward not in (0.0, 1.0) or second.reward not in (0.0, 1.0)
                or first.fee != (0.1 if first.action == 3 else 0.0)
                or second.fee != 0.0 or (first.action == 3 and first.reward != 0.0)):
            raise ValueError("invalid chosen-action outcome or actual probe cost")
        result = first.reward - first.fee + second.reward - second.fee
        if not math.isfinite(result):
            raise ValueError("nonfinite actual pair net")
        if self.memory_enabled:
            self.memory.append(ActualTwoStepRecord(
                first.before, first.action, first.time, result,
                first.provenance, second.provenance,
            ))
        # Do NOT truncate seen-provenance with bounded memory, since eviction
        # must never permit replay of an old factual pair.
        self._seen_provenance.update((first.provenance, second.provenance))
        self._last_time = second.time
        if self.adaptation_enabled and self.session_updates < self.update_limit:
            x = self.features(first.before, first.action)
            z = [sum(a*b for a, b in zip(row, x)) for row in self.u]
            pred = sum(v*h for v, h in zip(self.v, z))
            delta = max(-0.5, min(0.5, result - pred))
            prior_v = self.v[:]
            for i in range(self.rank):
                self.v[i] += self.lr * max(-0.5, min(0.5, delta*z[i]))
                for j in range(8):
                    self.u[i][j] += self.lr * max(
                        -0.5, min(0.5, delta*prior_v[i]*x[j]))
            self.session_updates += 1
            self.total_updates += 1

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        self.session_updates = 0
        if erase_persistent:
            self.memory.clear()
            self._seen_provenance.clear()
            self._last_time = 0
            self.u = [[0.1 if (j+i) % self.rank == 0 else -0.05 for j in range(8)]
                      for i in range(self.rank)]
            self.v = [0.0] * self.rank
            self.total_updates = 0


class FrozenNeuralTwoStepPolicy:
    """All 3 neural forecasters share same factual memory and continuation.

    In first stage choose argmax of predicted two-step net (optional PCMA
    residual). In second stage use the same fixed public-information guess
    for every architecture; learn factual PCMA only after the completed pair.
    """

    def __init__(
        self, predictor: nn.Module, *,
        continuation: FrozenPublicContinuation = FrozenPublicContinuation(),
        pcma_weight: float = 0.5,
        memory_enabled: bool = True, adaptation_enabled: bool = True,
        allow_probe: bool = True,
    ):
        if not 0 <= pcma_weight <= 1:
            raise ValueError("invalid PCMA fusion")
        if type(continuation) is not FrozenPublicContinuation:
            raise ValueError("fixed continuation required")
        if not isinstance(predictor, (FactualRewardPredictor, PCMAGRURewardPredictor)):
            raise TypeError("only explicitly supported public-token models")
        self.predictor = predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.continuation = continuation
        self.pcma_weight, self.allow_probe = pcma_weight, allow_probe
        self.state = TwoStepPCMAState(memory_enabled=memory_enabled,
                                      adaptation_enabled=adaptation_enabled)
        self.history: list[FactualTransition] = []
        self._pending: tuple[PublicObservation, int] | None = None
        self._first_event: FactualTransition | None = None
        self._last_time = 0

    @torch.no_grad()
    def choose(self, observation: PublicObservation) -> int:
        if self._pending is not None:
            raise RuntimeError("previous chosen action has no factual feedback")
        if self._first_event is not None:
            action = self.continuation.choose_guess(observation)
        else:
            choices = range(4) if self.allow_probe and observation.steps_remaining > 1 else range(3)
            scores: list[float] = []
            for action in choices:
                obs, act, mode = causal_tokens(observation, action, self.history)
                neural = float(self.predictor(obs, act, mode)[0])
                pcma = self.state.value(observation, action)
                scores.append((1.0 - self.pcma_weight)*neural + self.pcma_weight*pcma)
            action = max(range(len(scores)), key=lambda i: scores[i])
        self._pending = (observation, action)
        return action

    def observe(self, event: FactualTransition) -> None:
        if self._pending is None:
            raise RuntimeError("no chosen action awaiting factual feedback")
        before, action = self._pending
        if (type(event) is not FactualTransition or
                event.before != before or event.action != action):
            raise ValueError("mismatched executed action")
        if event.time <= self._last_time or event.fee != (0.1 if action == 3 else 0.0):
            raise ValueError("invalid logical time or paid fee")
        if event.reward not in (0.0, 1.0) or (action == 3 and event.reward != 0.0):
            raise ValueError("invalid factual reward")
        if self._first_event is None:
            if event.done:
                raise ValueError("a two-step first action cannot end session")
            self._first_event = event
        else:
            self.state.record(self._first_event, event)
            self._first_event = None
        self.history.append(event)
        self._last_time = event.time
        self._pending = None

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if self._pending is not None or self._first_event is not None:
            raise RuntimeError("cannot restart with unresolved two-step outcome")
        self.history.clear()
        self.state.restart_session(erase_persistent=erase_persistent)


def fit_development_two_step_predictors(
    configs: Sequence[OODWorldConfig], *,
    fixture_randomization_seed: int, fixture_initialization_seed: int,
    updates: int = 2, sessions: int = 2,
) -> tuple[dict[str, nn.Module], tuple[CausalRewardSample, ...]]:
    """Identical train rows and update budget for all; NO heldout worlds.

    Deliberately capped to 32 updates; engineering-only. Training a proposed
    scientific comparison requires a fresh, frozen protocol and explicit new
    scientific compute authorization. Does not save or register checkpoints.
    """
    if (not configs or any(c.split != "development" for c in configs)
            or type(fixture_initialization_seed) is not int
            or fixture_initialization_seed < 0):
        raise ValueError("heldout and invalid initialization forbidden")
    if type(updates) is not int or not 1 <= updates <= 32:
        raise ValueError("developer update budget must be between 1 and 32")
    pairs = randomized_development_first_actions(
        configs, fixture_randomization_seed=fixture_randomization_seed,
        sessions=sessions,
    )
    samples = randomized_pairs_to_causal_samples(pairs)
    models: dict[str, nn.Module] = {}
    for kind in ("hybrid", "transformer", "gru"):
        torch.manual_seed(fixture_initialization_seed)
        model = (PCMAGRURewardPredictor() if kind == "gru"
                 else FactualRewardPredictor(kind))
        fit_factual_predictor(model, samples, updates=updates)
        models[kind] = model.eval()
    return models, samples
