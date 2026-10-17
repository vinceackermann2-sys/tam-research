"""Factual two-step information-acquisition engineering controls for PCMA-v1.

No simulator, latent rule, privileged teacher, or counterfactual rewards are
imported. The learned values are REALIZED returns under the policy's second
action, not unbiased estimates of the causal effect of probing. This is a
nonconfirmatory comparison tool, not evidence of a novel SI mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation


@dataclass(frozen=True)
class FactualTwoStepOutcome:
    """Actual paired chosen-action reward, including paid probe/opportunity costs."""
    first_observation: PublicObservation
    first_action: int
    second_action: int
    first_time: int
    second_time: int
    observed_total_net: float


def factual_two_step_outcomes(
    sessions: Sequence[Sequence[FactualTransition]],
) -> tuple[FactualTwoStepOutcome, ...]:
    """Form NONOVERLAPPING consecutive pairs strictly within each session.

    Only both actual executed actions enter the pair's net reward. The query
    never contains an alternative action's outcome. Odd terminal steps yield
    no two-step label, rather than crossing into a new session.
    """
    samples: list[FactualTwoStepOutcome] = []
    last_time = 0
    for session in sessions:
        if not session or not session[-1].done:
            raise ValueError("all sessions must be nonempty and terminal")
        if any(e.done for e in session[:-1]):
            raise ValueError("premature terminal transition")
        for event in session:
            if event.time <= last_time:
                raise ValueError("nonmonotonic or replayed factual time")
            if type(event.action) is not int or event.action not in range(4):
                raise ValueError("invalid factual action")
            if event.reward not in (0.0, 1.0) or event.fee != (0.1 if event.action == 3 else 0.0):
                raise ValueError("invalid outcome or known fee")
            last_time = event.time
        for i in range(0, len(session) - 1, 2):
            first, second = session[i], session[i + 1]
            if second.time != first.time + 1 or first.done:
                raise ValueError("nonconsecutive in-session outcomes")
            samples.append(FactualTwoStepOutcome(
                first.before, first.action, second.action,
                first.time, second.time,
                first.reward - first.fee + second.reward - second.fee,
            ))
    return tuple(samples)


class OnlineFactualTwoStepPolicy:
    """Learns first-stage 2-action returns from actual purchases/guesses only.

    The first-stage probe may be selected by epsilon exploration, then by the
    observed context-specific two-step net score if sufficiently supported.
    The second action is always a guess; probing repeatedly cannot silently
    create free information. A 0.1 fee and sacrificed guessing opportunity
    are both counted in the realized two-action return.

    Important: empirical probe-vs-guess averages remain action-selection-
    biased, unless first actions were separately randomized with known
    propensities. This simple online policy makes NO counterfactual claim.
    """
    def __init__(
        self, *, allow_probe: bool = True, exploration: float = 0.15,
        pseudocount: float = 2.0, minimum_probe_samples: int = 2,
        probe_margin: float = 0.05, fixture_seed: int = 0,
    ):
        if not (0.0 <= exploration <= 1.0) or pseudocount <= 0:
            raise ValueError("invalid exploration or smoothing")
        if type(minimum_probe_samples) is not int or minimum_probe_samples < 0:
            raise ValueError("invalid probe evidence requirement")
        if probe_margin < 0:
            raise ValueError("invalid probe decision margin")
        self.allow_probe = allow_probe
        self.exploration = exploration
        self.pseudocount = pseudocount
        self.minimum_probe_samples = minimum_probe_samples
        self.probe_margin = probe_margin
        self._rng = random.Random(fixture_seed)
        self._first_values: dict[tuple[int, int], tuple[int, float]] = {}
        self._guess_values: dict[tuple[int, int], tuple[int, float]] = {}
        self._unresolved: tuple[PublicObservation, int, str] | None = None
        self._first_event: FactualTransition | None = None
        self._last_time = 0
        self.paired_samples = 0
        self.observed_steps = 0

    def _estimate(self, key: tuple[int, int], table: dict,
                  prior: float) -> float:
        n, total = table.get(key, (0, 0.0))
        return (total + self.pseudocount * prior) / (n + self.pseudocount)

    def _guess(self, observation: PublicObservation) -> int:
        if self._rng.random() < self.exploration:
            return self._rng.randrange(3)
        return max(range(3), key=lambda a:
                   self._estimate((observation.symbol, a),
                                  self._guess_values, 1.0 / 3.0))

    def _first_action(self, observation: PublicObservation) -> int:
        guesses = tuple(range(3))
        # Last action can never benefit from a probe, regardless of score.
        if observation.steps_remaining <= 1:
            return self._guess(observation)
        candidates = guesses + ((3,) if self.allow_probe else ())
        if self._rng.random() < self.exploration:
            return self._rng.choice(candidates)
        # Two-step shrinkage prior: two guesses yield ~2/3 expected reward
        # before experience; a probe sacrifices a guess and costs 0.1.
        values = {a: self._estimate(
            (observation.symbol, a), self._first_values,
            (1.0 / 3.0 - 0.1) if a == 3 else 2.0 / 3.0,
        ) for a in candidates}
        best_guess = max(guesses, key=lambda a: values[a])
        if (self.allow_probe and
                self._first_values.get((observation.symbol, 3), (0, 0.0))[0]
                >= self.minimum_probe_samples and
                values[3] > values[best_guess] + self.probe_margin):
            return 3
        return best_guess

    def choose(self, observation: PublicObservation) -> int:
        if type(observation) is not PublicObservation or observation.symbol not in range(3):
            raise ValueError("invalid public observation")
        if self._unresolved is not None:
            raise RuntimeError("previous executed action not yet observed")
        if self._first_event is not None:
            action, phase = self._guess(observation), "second"
        else:
            action, phase = self._first_action(observation), "first"
        self._unresolved = (observation, action, phase)
        return action

    @staticmethod
    def _append_value(table: dict, key: tuple[int, int], target: float) -> None:
        n, total = table.get(key, (0, 0.0))
        table[key] = (n + 1, total + target)

    def observe(self, event: FactualTransition) -> None:
        if self._unresolved is None:
            raise RuntimeError("no preceding chosen action")
        before, action, phase = self._unresolved
        if (type(event) is not FactualTransition or
                event.before != before or event.action != action):
            raise ValueError("mismatched factual feedback")
        if event.time <= self._last_time:
            raise ValueError("nonmonotonic factual event time")
        if event.fee != (0.1 if action == 3 else 0.0):
            raise ValueError("invalid probe fee")
        if event.reward not in (0.0, 1.0) or (action == 3 and event.reward != 0.0):
            raise ValueError("invalid factual reward")

        if action < 3:
            self._append_value(
                self._guess_values, (before.symbol, action), event.reward)

        if phase == "first":
            if not event.done:
                self._first_event = event
            # If only one action remained, there is no fictitious two-step
            # outcome, and no sample is added to first-stage values.
        else:
            first = self._first_event
            if first is None or event.time != first.time + 1:
                raise ValueError("missing adjacent first action")
            target = first.reward - first.fee + event.reward - event.fee
            self._append_value(
                self._first_values, (first.before.symbol, first.action), target)
            self.paired_samples += 1
            self._first_event = None
        self._last_time = event.time
        self.observed_steps += 1
        self._unresolved = None

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if self._unresolved is not None or self._first_event is not None:
            raise RuntimeError("cannot restart with unresolved two-step purchase")
        if erase_persistent:
            self._first_values.clear()
            self._guess_values.clear()
            self.paired_samples = 0
            self.observed_steps = 0

    def observed_first_count(self, symbol: int, action: int) -> int:
        if symbol not in range(3) or action not in range(4):
            raise ValueError("invalid query")
        return self._first_values.get((symbol, action), (0, 0.0))[0]

    def observed_first_mean(self, symbol: int, action: int) -> float | None:
        n, total = self._first_values.get((symbol, action), (0, 0.0))
        return total / n if n else None
