"""Strong inexpensive factual-policy controls for PCMA-v1 engineering only.

These are evaluable on the SAME public observation / executed-transition API
as neural-PCMA. No privileged hidden world rule or hypothetical action outcome
is accessible. Hyperparameters below are placeholders, NOT frozen thresholds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation


class UniformActionControl:
    """Deterministic reproducible random policy (including paid probe)."""
    def __init__(self, fixture_seed: int = 0):
        self._rng = random.Random(fixture_seed)
        self.feedback: list[FactualTransition] = []

    def choose(self, observation: PublicObservation) -> int:
        return self._rng.randrange(4)

    def observe(self, event: FactualTransition) -> None:
        self.feedback.append(event)

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if erase_persistent:
            self.feedback.clear()


class FixedGuessControl:
    """Completely static no-learning, no-probe lower-cost comparator."""
    def __init__(self, guess: int = 0):
        if type(guess) is not int or guess not in range(3):
            raise ValueError("guess must be 0, 1 or 2")
        self.guess = guess
        self.observed_count = 0

    def choose(self, observation: PublicObservation) -> int:
        return self.guess

    def observe(self, event: FactualTransition) -> None:
        self.observed_count += 1

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if erase_persistent:
            self.observed_count = 0


class SlidingWindowFactualControl:
    """Recency-limited Beta-smoothed empirical reward per symbol/action.

    Epsilon is fixed *action exploration* rather than privileged optimism.
    A deterministic first-action or periodic probe schedule may be enabled
    (not optimized), and its fees come from actual executed transitions.
    """
    def __init__(self, *, window: int = 24, exploration: float = 0.10,
                 probe_period: int = 0, fixture_seed: int = 0):
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        if not (0 <= exploration <= 1):
            raise ValueError("exploration must lie in [0,1]")
        if type(probe_period) is not int or probe_period < 0:
            raise ValueError("probe_period must be nonnegative")
        self.window = window
        self.exploration = exploration
        self.probe_period = probe_period
        self._rng = random.Random(fixture_seed)
        self._events: deque[FactualTransition] = deque(maxlen=window)
        self._steps = 0

    @property
    def events(self) -> tuple[FactualTransition, ...]:
        return tuple(self._events)

    def choose(self, observation: PublicObservation) -> int:
        if self.probe_period and self._steps % self.probe_period == 0 \
                and observation.steps_remaining > 1:
            return 3
        if self._rng.random() < self.exploration:
            return self._rng.randrange(3)
        scores = []
        for action in range(3):
            matches = [e for e in self._events if
                       e.before.symbol == observation.symbol and e.action == action]
            # Prior mean 1/3 instead of the binary-class Beta(1,1) 1/2:
            # random guessing among three actions yields reward probability 1/3.
            scores.append((1.0 + sum(e.reward for e in matches)) /
                          (3.0 + len(matches)))
        return max(range(3), key=lambda action: scores[action])

    def observe(self, event: FactualTransition) -> None:
        self._events.append(event)
        self._steps += 1

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        self._steps = 0
        if erase_persistent:
            self._events.clear()


class SurpriseResetFactualControl:
    """Nonprivileged, action-conditional exponential values and CUSUM reset.

    Only error on chosen guess can reset a value. No access to latent change
    times, heldout family, unchosen action, or true hidden rule. Not tuned.
    """
    def __init__(self, *, rate: float = 0.25, threshold: float = 1.20,
                 probe_period: int = 0):
        if not (0 < rate <= 1) or threshold <= 0:
            raise ValueError("invalid change-control settings")
        if type(probe_period) is not int or probe_period < 0:
            raise ValueError("invalid probe period")
        self.rate, self.threshold, self.probe_period = rate, threshold, probe_period
        self.q: dict[tuple[int, int], float] = {}
        self.cusum: dict[tuple[int, int], float] = {}
        self.count = 0
        self.session_steps = 0
        self.resets = 0

    def choose(self, observation: PublicObservation) -> int:
        if self.probe_period and self.session_steps % self.probe_period == 0 \
                and observation.steps_remaining > 1:
            return 3
        return max(range(3), key=lambda action:
                   self.q.get((observation.symbol, action), 1.0 / 3.0))

    def observe(self, event: FactualTransition) -> None:
        self.count += 1
        self.session_steps += 1
        if event.action == 3:  # paid probe has no label for guessed actions
            return
        key = (event.before.symbol, event.action)
        old = self.q.get(key, 1.0 / 3.0)
        # One-sided negative surprise: regime shift can invalidate optimistic
        # value estimates. Accumulate only actually experienced errors.
        change = max(0.0, old - event.reward)
        self.cusum[key] = max(0.0, self.cusum.get(key, 0.0) +
                              change - self.rate / 2)
        if self.cusum[key] > self.threshold:
            self.q[key] = 1.0 / 3.0
            self.cusum[key] = 0.0
            self.resets += 1
        else:
            self.q[key] = (1.0 - self.rate) * old + self.rate * event.reward

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        self.session_steps = 0
        if erase_persistent:
            self.q.clear()
            self.cusum.clear()
            self.count = 0
            self.resets = 0


@dataclass(frozen=True)
class FactualTraceAudit:
    """Descriptive measurement; NOT a claim of scientific validity."""
    actions: int
    guesses: int
    paid_probes: int
    gross_reward: float
    total_fees: float
    net_reward: float


def audit_factual_trace(
    traces: tuple[tuple[FactualTransition, ...], ...]
) -> FactualTraceAudit:
    if not traces or not all(traces):
        raise ValueError("need at least one nonempty factual session")
    events = [e for session in traces for e in session]
    times = [e.time for e in events]
    if times != sorted(set(times)):
        raise ValueError("nonunique or unordered factual timestamps")
    if any(session[-1].done is False for session in traces):
        raise ValueError("missing session terminal transitions")
    if any(e.action not in range(4) or e.reward not in (0.0, 1.0)
           for e in events):
        raise ValueError("invalid factual action/return")
    probes = sum(e.action == 3 for e in events)
    gross = float(sum(e.reward for e in events))
    fees = float(sum(e.fee for e in events))
    return FactualTraceAudit(len(events), len(events) - probes, probes,
                             gross, fees, gross - fees)
