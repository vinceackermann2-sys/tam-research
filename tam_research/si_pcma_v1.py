"""PCMA-v1 non-confirmatory engineering scaffold for the original SI successor.

No benchmark result, trained controller, causal identification or breakthrough is implied.
Every policy-visible value is from public observations or factual executed actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

import torch
from tam_research.si_original_hybrid_v0 import (
    OriginalSIHybrid, SIConfig, TransformerControl,
)


@dataclass(frozen=True)
class WorldConfig:
    seed: int
    family: Literal["stable", "shift", "noisy", "long"] = "stable"
    max_steps: int = 6
    shift_after: int = 9

    def __post_init__(self):
        if self.family not in ("stable", "shift", "noisy", "long"):
            raise ValueError("unknown world family")
        if self.max_steps < 4 or self.max_steps > 16:
            raise ValueError("unsupported action horizon")
        if self.shift_after < 1:
            raise ValueError("shift_after must be positive")


@dataclass(frozen=True)
class PublicObservation:
    symbol: int
    steps_remaining: int


@dataclass(frozen=True)
class FactualTransition:
    before: PublicObservation
    action: int
    after: PublicObservation
    reward: float
    fee: float
    time: int
    provenance: str
    done: bool


class PersistentWorld:
    """Hidden 16-state toy dynamics; public symbols only, never latent regime.

    Sessions reset the public action window; the hidden state/world persists.
    No counterfactual API is exposed to the agent.
    """
    def __init__(self, config: WorldConfig):
        self.config = config
        self._rng = random.Random(config.seed)
        self._state = self._rng.randrange(16)
        self._regime = self._rng.randrange(4)
        self._time = 0
        self._session_actions = 0
        self._did_shift = False
        self._observation = self._observe(probe=False)

    def _observe(self, *, probe: bool) -> int:
        rate = 0.0 if probe else (0.25 if self.config.family == "noisy" else 0.06)
        value = self._state % 4
        if self._rng.random() < rate:
            value = (value + self._rng.randrange(1, 4)) % 4
        return value

    def public_observation(self) -> PublicObservation:
        return PublicObservation(self._observation, self.config.max_steps - self._session_actions)

    def restart_session(self) -> PublicObservation:
        if self._session_actions < self.config.max_steps:
            raise RuntimeError("cannot restart a session before termination")
        self._session_actions = 0
        self._observation = self._observe(probe=False)
        return self.public_observation()

    def step(self, action: int) -> FactualTransition:
        if not isinstance(action, int) or action not in range(4):
            raise ValueError("actions are 0,1,2 (effectful) or 3 (probe)")
        if self._session_actions >= self.config.max_steps:
            raise RuntimeError("session ended; restart_session required")
        before = self.public_observation()
        if (self.config.family == "shift" and not self._did_shift
                and self._time >= self.config.shift_after):
            self._regime = (self._regime + 1) % 4
            self._did_shift = True
        fee = 0.10 if action == 3 else 0.0
        if action != 3:
            self._state = (self._state + (action + 1) * (self._regime + 1)) % 16
            reward = float(self._state % 4 == 0)
        else:
            reward = 0.0
        self._time += 1
        self._session_actions += 1
        self._observation = self._observe(probe=(action == 3))
        return FactualTransition(before, action, self.public_observation(),
                                 reward, fee, self._time, f"event-{self._time}",
                                 self._session_actions == self.config.max_steps)


class FactualMemory:
    """Bounded in-process memory: factual rows only, no source-authentication claim."""
    def __init__(self, capacity: int = 128):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[FactualTransition] = []
        self._seen: set[str] = set()
        self._last_time = 0

    @property
    def items(self) -> tuple[FactualTransition, ...]:
        return tuple(self._items)

    def record(self, event: FactualTransition) -> None:
        if not isinstance(event, FactualTransition):
            raise TypeError("memory accepts only factual transition records")
        if event.provenance in self._seen or event.time <= self._last_time:
            raise ValueError("duplicate provenance or nonmonotonic logical time")
        if not (0 <= event.before.symbol < 4 and 0 <= event.after.symbol < 4):
            raise ValueError("invalid public observation")
        if event.action not in range(4) or event.fee < 0:
            raise ValueError("invalid action or fee")
        self._items.append(event)
        self._seen.add(event.provenance)
        self._last_time = event.time
        if len(self._items) > self.capacity:
            old = self._items.pop(0)
            self._seen.remove(old.provenance)

    def retrieve(self, observation: PublicObservation, action: int, k: int = 8
                 ) -> tuple[FactualTransition, ...]:
        if action not in range(4) or k < 1:
            raise ValueError("invalid retrieval request")
        matches = [event for event in reversed(self._items)
                   if event.before.symbol == observation.symbol and event.action == action]
        return tuple(matches[:k])


class ReversibleLowRankRewardAdapter:
    """Rank-r online reward residual; clipped updates use chosen factual outcomes.

    This is a small explicit adapter, not pretrained fast-weights nor a learned
    neural policy. It receives no unchosen-arm reward and keeps snapshots.
    """
    def __init__(self, rank: int = 2, lr: float = 0.05,
                 max_updates_per_session: int = 8, clip: float = 0.5):
        if rank < 1 or not (0 < lr <= 1) or max_updates_per_session < 0 or clip <= 0:
            raise ValueError("invalid adapter configuration")
        self.rank, self.lr, self.limit, self.clip = rank, lr, max_updates_per_session, clip
        self.u = [[(0.1 if (j + i) % rank == 0 else -0.05)
                   for j in range(8)] for i in range(rank)]
        self.v = [0.0] * rank
        self.updates = 0
        self.total_updates = 0

    @staticmethod
    def _features(observation: PublicObservation, action: int) -> list[float]:
        if observation.symbol not in range(4) or action not in range(4):
            raise ValueError("invalid public feature")
        x = [0.0] * 8
        x[observation.symbol] = 1.0
        x[4 + action] = 1.0
        return x

    def predict(self, observation: PublicObservation, action: int) -> float:
        x = self._features(observation, action)
        return sum(v * sum(u_j * x_j for u_j, x_j in zip(u, x))
                   for v, u in zip(self.v, self.u))

    def update(self, event: FactualTransition) -> bool:
        if self.updates >= self.limit or event.action == 3:
            return False
        x = self._features(event.before, event.action)
        hidden = [sum(u_j * x_j for u_j, x_j in zip(u, x)) for u in self.u]
        pred = sum(v * h for v, h in zip(self.v, hidden))
        delta = max(-self.clip, min(self.clip, event.reward - pred))
        old_v = self.v[:]
        for i in range(self.rank):
            self.v[i] += self.lr * max(-self.clip, min(self.clip, delta * hidden[i]))
            for j in range(8):
                self.u[i][j] += self.lr * max(-self.clip,
                    min(self.clip, delta * old_v[i] * x[j]))
        self.updates += 1
        self.total_updates += 1
        return True

    def restart_session(self) -> None:
        self.updates = 0

    def snapshot(self) -> tuple:
        return (tuple(tuple(row) for row in self.u), tuple(self.v),
                self.updates, self.total_updates)

    def restore(self, snapshot: tuple) -> None:
        rows, values, count, total = snapshot
        if len(rows) != self.rank or len(values) != self.rank or any(len(row) != 8 for row in rows):
            raise ValueError("incompatible adapter snapshot")
        self.u = [list(row) for row in rows]
        self.v = list(values)
        self.updates = count
        self.total_updates = total


class PCMAState:
    """Same factual memory and reversible adapter API for every backbone."""
    def __init__(self, *, memory_enabled: bool = True, adaptation_enabled: bool = True,
                 capacity: int = 128, rank: int = 2):
        self.memory_enabled, self.adaptation_enabled = memory_enabled, adaptation_enabled
        self.memory = FactualMemory(capacity)
        self.adapter = ReversibleLowRankRewardAdapter(rank)
        self.window: list[FactualTransition] = []

    def observe(self, event: FactualTransition) -> None:
        # No policy can query the world for an unexecuted action through this API.
        if self.memory_enabled:
            self.memory.record(event)
        if self.adaptation_enabled:
            self.adapter.update(event)
        self.window.append(event)

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        self.window.clear()
        self.adapter.restart_session()
        if erase_persistent:
            self.memory = FactualMemory(self.memory.capacity)
            self.adapter = ReversibleLowRankRewardAdapter(self.adapter.rank,
                                                           self.adapter.lr,
                                                           self.adapter.limit,
                                                           self.adapter.clip)

    def value(self, observation: PublicObservation, action: int) -> float:
        rows = self.memory.retrieve(observation, action) if self.memory_enabled else ()
        # Identical public-information, nonprivileged exact-match baseline.
        memory_value = (sum(x.reward - x.fee for x in rows) / len(rows)) if rows else 0.0
        adapted = self.adapter.predict(observation, action) if self.adaptation_enabled else 0.0
        return (memory_value + adapted) / (int(bool(rows)) + int(self.adaptation_enabled)) \
            if rows or self.adaptation_enabled else 0.0


class SharedBackboneInterface:
    """Same public inputs + same PCMA state for canonical hybrid/Transformer.

    The heads are UNTRAINED; this is a shape/leakage interface, not an evaluated
    or matched-compute full controller. Training/learned gating comes later.
    """
    def __init__(self, kind: Literal["hybrid", "transformer"], state: PCMAState):
        if kind not in ("hybrid", "transformer"):
            raise ValueError("unsupported backbone")
        self.kind, self.state = kind, state
        cfg = SIConfig()
        self.model = (OriginalSIHybrid(cfg) if kind == "hybrid" else TransformerControl(cfg))
        self.model.eval()

    @torch.no_grad()
    def forecast(self, observation: PublicObservation, action: int) -> torch.Tensor:
        if observation.symbol not in range(4) or action not in range(4):
            raise ValueError("invalid public input")
        obs = torch.tensor([[[observation.symbol / 3., observation.steps_remaining / 16., 0.]]])
        a = torch.zeros((1, 1, 4))
        a[0, 0, action] = 1
        return self.model(obs, a)["prediction"][0, -1].detach().clone()
