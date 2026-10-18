"""Strong cheap factual two-step controls for original SI/PCMA-v1.

Development engineering ONLY: actual randomized first-action pairs are the
training data. Three classical controls share the neural models' query:
pre-action PUBLIC observation, chosen candidate action, and at most the last
seven actually completed factual transitions. No hidden rule, alternate-arm
reward, probe oracle, future outcome, heldout examples, or teacher labels.

These controls are NOT a scientific study. The cheap controller may win; if it
does, that is a key falsifier, not a reason to weaken the classical baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_factual_learning import CausalRewardSample
from tam_research.si_pcma_v1_neural_probe import (
    TwoStepPCMAState, randomized_pairs_to_causal_samples,
)
from tam_research.si_pcma_v1_probe_randomization import (
    FrozenPublicContinuation, randomized_development_first_actions,
)
from tam_research.si_pcma_v1_ood_world import OODWorldConfig

ControlKind = Literal["constant", "symbol", "ridge"]

# Ridges: a fixed, deterministic low-capacity feature family, not tuned on
# heldout worlds. It includes basic recent factual history, as neural inputs do.
def public_two_step_features(
    observation: PublicObservation, action: int,
    prior: Sequence[FactualTransition],
    *, logical_time: int | None = None,
) -> tuple[float, ...]:
    if (type(observation) is not PublicObservation
            or observation.symbol not in range(3)
            or not 1 <= observation.steps_remaining <= 16
            or type(action) is not int or action not in range(4)):
        raise ValueError("only a valid public pre-action query is allowed")
    if logical_time is not None and (type(logical_time) is not int or logical_time < 1):
        raise ValueError("invalid query time")
    recent = tuple(prior)[-7:]
    for i, event in enumerate(recent):
        if (type(event) is not FactualTransition
                or event.before.symbol not in range(3)
                or event.action not in range(4)
                or event.reward not in (0.0, 1.0)
                or event.fee != (0.1 if event.action == 3 else 0.0)
                or (logical_time is not None and event.time >= logical_time)
                or (i and event.time <= recent[i-1].time)):
            raise ValueError("nonfactual or future history")
    a, s = action, observation.symbol
    features = [1.0]
    features.extend(float(i == a) for i in range(4))
    features.extend(float(i == s) for i in range(3))
    features.extend(float(i == 4*s + a) for i in range(12))
    features.append(observation.steps_remaining / 16.0)
    latest = recent[-1] if recent else None
    features.append(latest.reward - latest.fee if latest else 0.0)
    features.append(float(latest.action == 3) if latest else 0.0)
    features.extend(float(latest is not None and latest.action == i)
                    for i in range(4))
    rows = [e.reward - e.fee for e in recent
            if e.before.symbol == s and e.action == a]
    features.append(sum(rows) / len(rows) if rows else 0.0)
    features.append(min(len(rows), 7) / 7.0)
    return tuple(features)


def _prior_for_first_action(action: int) -> float:
    # Two guesses have 2/3 naïve expected total, whereas a paid probe loses
    # the first guess opportunity and costs 0.1. These are only smoothing
    # priors, NOT knowledge of the hidden rule or an oracle estimate.
    return (1.0 / 3.0 - 0.1) if action == 3 else 2.0 / 3.0


def _verify_sample(s: CausalRewardSample) -> None:
    if type(s) is not CausalRewardSample:
        raise TypeError("expected actual chosen-arm factual sample")
    public_two_step_features(s.observation, s.action, s.prior,
                             logical_time=s.logical_time)
    if (type(s.observed_net_reward) not in (int, float)
            or not math.isfinite(s.observed_net_reward)
            or not -0.1 <= s.observed_net_reward <= 2.0):
        raise ValueError("invalid real two-step target")


@dataclass(frozen=True)
class ClassicalFactualTwoStepPredictor:
    """Fit-once public-data-only first-arm value predictor.

    Constant is one smoothed Q per action; symbol is smoothed Q per
    observation symbol/action; ridge has fixed public factual features and
    l2-regularization. All three estimate REAL observed paired returns,
    including lost first guessing opportunity and actual probe fees.
    """
    kind: ControlKind
    counts: tuple[tuple[int, ...], ...]
    totals: tuple[tuple[float, ...], ...]
    coefficients: tuple[float, ...]
    smoothing: float
    training_rows: int

    def predict(
        self, observation: PublicObservation, action: int,
        prior: Sequence[FactualTransition],
    ) -> float:
        features = public_two_step_features(observation, action, prior)
        if self.kind == "ridge":
            if len(features) != len(self.coefficients):
                raise ValueError("incompatible public ridge width")
            return float(sum(x*y for x, y in zip(features, self.coefficients)))
        group = 0 if self.kind == "constant" else observation.symbol
        n = self.counts[group][action]
        total = self.totals[group][action]
        return (total + self.smoothing * _prior_for_first_action(action)) / (
            n + self.smoothing)


def fit_classical_two_step(
    kind: ControlKind, samples: Sequence[CausalRewardSample], *,
    smoothing: float = 2.0, ridge_penalty: float = 1.0,
) -> ClassicalFactualTwoStepPredictor:
    """Deterministic offline fit on caller-supplied factual developer samples.

    The boundary cannot cryptographically know where caller-created samples
    originated. For a development-only verified source, use
    `fit_development_classical_two_step` below, which refuses heldout worlds.
    """
    if kind not in ("constant", "symbol", "ridge") or not samples:
        raise ValueError("missing samples or unsupported baseline")
    if (type(smoothing) not in (int, float) or not math.isfinite(smoothing)
            or smoothing <= 0 or type(ridge_penalty) not in (int, float)
            or not math.isfinite(ridge_penalty) or ridge_penalty <= 0):
        raise ValueError("invalid frozen classical training settings")
    groups = 1 if kind == "constant" else 3
    counts = [[0]*4 for _ in range(groups)]
    totals = [[0.0]*4 for _ in range(groups)]
    x_rows: list[tuple[float, ...]] = []
    y_rows: list[float] = []
    for sample in samples:
        _verify_sample(sample)
        group = 0 if kind == "constant" else sample.observation.symbol
        counts[group][sample.action] += 1
        totals[group][sample.action] += sample.observed_net_reward
        if kind == "ridge":
            x_rows.append(public_two_step_features(
                sample.observation, sample.action, sample.prior,
                logical_time=sample.logical_time))
            y_rows.append(sample.observed_net_reward)
    coefficients: tuple[float, ...] = ()
    if kind == "ridge":
        # Small fixed closed-form ridge fit. Deterministic no-optimizer
        # solution; therefore total training compute differs from neural.
        design = torch.tensor(x_rows, dtype=torch.float64)
        target = torch.tensor(y_rows, dtype=torch.float64)
        gram = design.T @ design
        gram.diagonal().add_(float(ridge_penalty))
        rhs = design.T @ target
        solved = torch.linalg.solve(gram, rhs)
        if not torch.isfinite(solved).all():
            raise ValueError("nonfinite factual ridge solution")
        coefficients = tuple(float(value) for value in solved.tolist())
    return ClassicalFactualTwoStepPredictor(
        kind=kind, counts=tuple(tuple(row) for row in counts),
        totals=tuple(tuple(row) for row in totals),
        coefficients=coefficients, smoothing=float(smoothing),
        training_rows=len(samples),
    )


class FrozenClassicalTwoStepPolicy:
    """Same frozen second guess and probe-inclusive online PCMA as neural.

    All chosen-action feedback is checked before the external memory/adapter
    sees it, and the backbone remains a locked offline factual predictor.
    """

    def __init__(
        self, predictor: ClassicalFactualTwoStepPredictor, *,
        continuation: FrozenPublicContinuation = FrozenPublicContinuation(),
        pcma_weight: float = 0.5,
        memory_enabled: bool = True,
        adaptation_enabled: bool = True,
        allow_probe: bool = True,
    ):
        if type(predictor) is not ClassicalFactualTwoStepPredictor:
            raise TypeError("frozen classical predictor required")
        if type(continuation) is not FrozenPublicContinuation:
            raise TypeError("frozen public continuation required")
        if type(pcma_weight) not in (int, float) or not 0 <= pcma_weight <= 1:
            raise ValueError("invalid shared PCMA fusion")
        self.predictor = predictor
        self.continuation, self.allow_probe = continuation, allow_probe
        self.pcma_weight = float(pcma_weight)
        self.state = TwoStepPCMAState(
            memory_enabled=memory_enabled,
            adaptation_enabled=adaptation_enabled)
        self.history: list[FactualTransition] = []
        self._pending: tuple[PublicObservation, int] | None = None
        self._first_event: FactualTransition | None = None
        self._last_time = 0

    def choose(self, observation: PublicObservation) -> int:
        if self._pending is not None:
            raise RuntimeError("unresolved prior chosen action")
        if self._first_event is not None:
            action = self.continuation.choose_guess(observation)
        else:
            candidates = (0, 1, 2, 3) if self.allow_probe and observation.steps_remaining > 1 else (0, 1, 2)
            value = {}
            for candidate in candidates:
                pred = self.predictor.predict(observation, candidate, self.history)
                correction = self.state.value(observation, candidate)
                value[candidate] = ((1.0-self.pcma_weight)*pred +
                                    self.pcma_weight*correction)
            action = max(candidates, key=lambda a: value[a])
        self._pending = (observation, action)
        return action

    def observe(self, event: FactualTransition) -> None:
        if self._pending is None:
            raise RuntimeError("no executed action awaiting feedback")
        before, action = self._pending
        if (type(event) is not FactualTransition or event.before != before
                or event.action != action or event.time <= self._last_time
                or event.fee != (0.1 if action == 3 else 0.0)
                or event.reward not in (0.0, 1.0)
                or (action == 3 and event.reward != 0.0)):
            raise ValueError("invalid or mismatched executed factual feedback")
        if self._first_event is None:
            if event.done:
                raise ValueError("two-step first action cannot end episode")
            self._first_event = event
        else:
            self.state.record(self._first_event, event)
            self._first_event = None
        self.history.append(event)
        self._last_time = event.time
        self._pending = None

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if self._pending is not None or self._first_event is not None:
            raise RuntimeError("cannot restart with unresolved factual pair")
        self.history.clear()
        self.state.restart_session(erase_persistent=erase_persistent)


def fit_development_classical_two_step(
    configs: Sequence[OODWorldConfig], *,
    fixture_randomization_seed: int,
    sessions: int = 2,
    continuation: FrozenPublicContinuation = FrozenPublicContinuation(),
) -> tuple[dict[str, ClassicalFactualTwoStepPredictor], tuple[CausalRewardSample, ...]]:
    """A trusted deterministic development collection; no heldout access."""
    if not configs or any(type(c) is not OODWorldConfig or c.split != "development"
                          for c in configs):
        raise ValueError("heldout configs prohibited by classical training")
    pairs = randomized_development_first_actions(
        configs, fixture_randomization_seed=fixture_randomization_seed,
        continuation=continuation, sessions=sessions)
    samples = randomized_pairs_to_causal_samples(pairs)
    return ({kind: fit_classical_two_step(kind, samples) for kind in
             ("constant", "symbol", "ridge")}, samples)
