"""Factual-only reliability gate for PCMA-v1; nonconfirmatory engineering.

Not a learned neural information-acquisition or probe policy. The gate estimates
past squared prediction errors of two permitted information sources using ONLY
the outcome of the action actually executed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_factual_learning import (
    FrozenNeuralFactualPolicy, FactualRewardPredictor, causal_tokens,
)


@dataclass
class FeedbackReliabilityGate:
    decay: float = 0.15
    prior_error: float = 0.25
    minimum_error: float = 0.01
    count: int = 0
    neural_error: float = 0.25
    memory_error: float = 0.25

    def __post_init__(self):
        if not (0 < self.decay <= 1) or self.prior_error <= 0 or self.minimum_error <= 0:
            raise ValueError("invalid reliability gate settings")
        if self.count != 0:
            raise ValueError("cannot inject external calibration counts")
        self.neural_error = self.memory_error = self.prior_error

    def weights(self) -> tuple[float, float]:
        inv_neural = 1 / max(self.minimum_error, self.neural_error)
        inv_memory = 1 / max(self.minimum_error, self.memory_error)
        total = inv_neural + inv_memory
        return inv_neural / total, inv_memory / total

    def observe(self, neural_prediction: float, memory_prediction: float,
                executed_net_reward: float) -> None:
        if not all(math.isfinite(x) for x in
                   (neural_prediction, memory_prediction, executed_net_reward)):
            raise ValueError("nonfinite observed calibration")
        # Predicted reward and its factual feedback lie in comparable units.
        errors = ((neural_prediction - executed_net_reward) ** 2,
                  (memory_prediction - executed_net_reward) ** 2)
        self.neural_error = ((1 - self.decay) * self.neural_error +
                             self.decay * min(4.0, errors[0]))
        self.memory_error = ((1 - self.decay) * self.memory_error +
                             self.decay * min(4.0, errors[1]))
        self.count += 1


class FeedbackCalibratedPCMAPolicy(FrozenNeuralFactualPolicy):
    """Adapt reliability across sessions without updating backbone weights.

    No current outcome is consumed until observe() follows choose(); decisions
    about unexecuted actions never produce calibration observations.
    """
    def __init__(self, predictor: FactualRewardPredictor, *,
                 memory_enabled: bool = True, adaptation_enabled: bool = True,
                 decay: float = 0.15):
        super().__init__(predictor,
                         memory_enabled=memory_enabled,
                         adaptation_enabled=adaptation_enabled)
        self.gate = FeedbackReliabilityGate(decay=decay)
        self._pending: tuple[PublicObservation, int, float, float] | None = None

    @torch.no_grad()
    def choose(self, observation: PublicObservation) -> int:
        if self._pending is not None:
            raise RuntimeError("previous chosen action has no factual feedback")
        neural_weight, memory_weight = self.gate.weights()
        estimates: list[tuple[float, float]] = []
        for action in range(4):
            obs, acts, mode = causal_tokens(observation, action, self.history)
            neural_prediction = float(self.predictor(obs, acts, mode)[0])
            memory_prediction = self.state.value(observation, action)
            estimates.append((neural_prediction, memory_prediction))
        best = max(range(4), key=lambda a:
                   neural_weight * estimates[a][0] +
                   memory_weight * estimates[a][1])
        self._pending = (observation, best, *estimates[best])
        return best

    def observe(self, event: FactualTransition) -> None:
        if self._pending is None:
            raise RuntimeError("no executed action expected")
        observation, action, neural_prediction, memory_prediction = self._pending
        if event.before != observation or event.action != action:
            raise ValueError("feedback does not match chosen public action")
        # Learn reliability BEFORE writing the chosen outcome to memory/adapter.
        self.gate.observe(neural_prediction, memory_prediction, event.reward - event.fee)
        super().observe(event)
        self._pending = None

    def restart_session(self, *, erase_persistent: bool = False) -> None:
        if self._pending is not None:
            raise RuntimeError("session cannot restart with unresolved action")
        super().restart_session(erase_persistent=erase_persistent)
        if erase_persistent:
            self.gate = FeedbackReliabilityGate(decay=self.gate.decay)
