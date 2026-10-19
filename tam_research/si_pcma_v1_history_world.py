"""SI PCMA-v1 development-only history-necessity world; NOT science.

The earlier three-symbol OOD law could be represented by a 3x4 public-symbol
lookup. This *additional* structural fixture instead makes the correct guess
depend on hidden phase and context separately, though the public symbol sees
only their sum. Thus two valid latent states can have the exact same public
observation and different correct guesses; a current-symbol lookup cannot
represent both. This is a test of architectural task discriminativity, NOT a
claim that any model learns or generalizes on this toy.

This module never constructs or runs heldout worlds. All random fixture seeds
are engineering inputs, never scientific reservations.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import OODWorldConfig
from tam_research.si_pcma_v1_probe_randomization import (
    FrozenPublicContinuation, RandomizedFactualPair,
)


class HistoryNecessaryDevelopmentWorld:
    """Partially observed 3x3 phase/context x 3 hidden-mode dynamics.

    Guess reward depends on hidden context *separately* from hidden phase,
    while observation only includes their sum modulo 3. Executed guess
    actions update both latent variables differently. A paid probe consumes
    an action, reveals an exact public symbol, and does not advance either
    hidden variable. The model has only the normal PublicObservation and
    FactualTransition interface. There is no oracle or unchosen-action API.

    A shift changes the hidden mode after nine executed actions globally.
    Hidden context/mode/phase continue across session boundaries.
    """

    def __init__(self, config: OODWorldConfig):
        if type(config) is not OODWorldConfig or config.split != "development":
            raise ValueError("heldout history-world rules are not available")
        self._config = config
        self._rng = random.Random(config.seed + 2**24)
        self._context = self._rng.randrange(3)
        self._phase = self._rng.randrange(3)
        self._mode = self._rng.randrange(3)
        self._offset = self._rng.randrange(3)
        self._time = 0
        self._actions_in_session = 0
        self._shifted = False
        self._symbol = self._sense(trusted=False)

    def _public_truth(self) -> int:
        return (self._context + self._phase) % 3

    def _correct_guess(self) -> int:
        # NOT representable by public truth alone: if context changes by +1
        # and phase by -1, public truth is unchanged, but this changes by +1.
        return (2*self._context + self._phase + self._mode + self._offset) % 3

    def _sense(self, *, trusted: bool) -> int:
        symbol = self._public_truth()
        if trusted:
            return symbol
        rate = 0.50 if self._config.scenario == "noisy" else 0.20
        if self._rng.random() < rate:
            return (symbol + self._rng.choice((1, 2))) % 3
        return symbol

    def observe(self) -> PublicObservation:
        return PublicObservation(
            symbol=self._symbol,
            steps_remaining=self._config.session_horizon - self._actions_in_session,
        )

    def step(self, action: int) -> FactualTransition:
        if type(action) is not int or action not in range(4):
            raise ValueError("invalid actual action")
        if self._actions_in_session == self._config.session_horizon:
            raise RuntimeError("session ended before next action")
        before = self.observe()
        if (self._config.scenario == "shift" and
                not self._shifted and self._time >= 9):
            self._mode = (self._mode + 1) % 3
            self._shifted = True
        fee = 0.1 if action == 3 else 0.0
        if action == 3:
            reward = 0.0
            self._symbol = self._sense(trusted=True)
        else:
            reward = float(action == self._correct_guess())
            self._context = (self._context + 1 + self._mode + (action == 2)) % 3
            self._phase = (self._phase + action + 1) % 3
            self._symbol = self._sense(trusted=False)
        self._time += 1
        self._actions_in_session += 1
        return FactualTransition(
            before=before, action=action, after=self.observe(),
            reward=reward, fee=fee, time=self._time,
            provenance=f"history-factual-{self._time}",
            done=self._actions_in_session == self._config.session_horizon,
        )

    def restart_session(self) -> PublicObservation:
        if self._actions_in_session != self._config.session_horizon:
            raise RuntimeError("cannot restart unfinished history session")
        self._actions_in_session = 0
        self._symbol = self._sense(trusted=False)
        return self.observe()


def randomized_history_development_pairs(
    configs: Sequence[OODWorldConfig], *,
    fixture_randomization_seed: int,
    continuation: FrozenPublicContinuation = FrozenPublicContinuation(),
    sessions: int = 2,
) -> tuple[RandomizedFactualPair, ...]:
    """Identical first-arm randomization/public follow-up as earlier dataset.

    Produces actual chosen pairs only, with known P(first action)=1/4.
    Source world IDs identify independent fixture worlds, not model features.
    No heldout input or world construction, even for one forbidden element.
    """
    if (not configs or type(sessions) is not int or not 1 <= sessions <= 32
            or any(type(c) is not OODWorldConfig or c.split != "development"
                   for c in configs)):
        raise ValueError("history training requires development-only config")
    if len({(c.seed, c.scenario) for c in configs}) != len(configs):
        raise ValueError("duplicate development history-world fixture")
    if type(fixture_randomization_seed) is not int or fixture_randomization_seed < 0:
        raise ValueError("invalid engineering fixture randomization")
    if type(continuation) is not FrozenPublicContinuation:
        raise ValueError("precommitted public continuation required")
    rng = random.Random(fixture_randomization_seed)
    rows: list[RandomizedFactualPair] = []
    for world_index, config in enumerate(configs):
        world = HistoryNecessaryDevelopmentWorld(config)
        for session_index in range(sessions):
            if session_index:
                world.restart_session()
            for pair_index in range(config.session_horizon // 2):
                before = world.observe()
                first_action = rng.randrange(4)
                first = world.step(first_action)
                second_action = continuation.choose_guess(world.observe())
                second = world.step(second_action)
                if first.done or second.time != first.time + 1 or second.before != first.after:
                    raise RuntimeError("broken chosen-action history pair")
                rows.append(RandomizedFactualPair(
                    world_index=world_index, session_index=session_index,
                    pair_index=pair_index, first_observation=before,
                    assigned_first_action=first_action,
                    assignment_probability=0.25,
                    first_event=first, second_event=second,
                    continuation_actions=continuation.action_by_symbol,
                ))
    return tuple(rows)
