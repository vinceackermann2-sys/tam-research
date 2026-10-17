"""PCMA-v1 unseen-law world (engineering fixtures, NOT scientific benchmarks).

The development and held-out rule families are disjoint permutations.
The controller is given ONLY a public symbol and previously executed factual
transitions. Python's private attributes are *not* a sandbox: keep the world
instance confined to the evaluator, not controller code.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal, Protocol

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation


RuleSplit = Literal["development", "heldout"]
Scenario = Literal["stable", "shift", "noisy", "long"]

# Three even/cyclic permutations versus three odd/reflected permutations.
# Full family disjointness is inspectable by the evaluation author, but the
# mapping is NEVER included in a PublicObservation or FactualTransition.
DEVELOPMENT_RULES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
HELDOUT_RULES = ((0, 2, 1), (2, 1, 0), (1, 0, 2))


@dataclass(frozen=True)
class OODWorldConfig:
    seed: int
    split: RuleSplit
    scenario: Scenario = "stable"

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.split not in ("development", "heldout"):
            raise ValueError("invalid rule split")
        if self.scenario not in ("stable", "shift", "noisy", "long"):
            raise ValueError("invalid scenario")

    @property
    def session_horizon(self) -> int:
        return 16 if self.scenario == "long" else 6


class OODRuleWorld:
    """Four actions: guesses 0/1/2, or paid sensor probe 3.

    Hidden context has three values. Guess reward is 1 only when it matches
    an unknown context-to-correct-action permutation; all other guesses
    return reward 0. A probe costs 0.1, yields no immediate reward, does not
    advance hidden context, and reveals a trusted symbol *for the next turn*.
    Guess actions advance context by one. Ordinary symbols are corrupted with
    independent probability (0.12 stable/shift/long, 0.50 noisy).
    Shift worlds rotate their unknown permutation after nine global actions.
    The hidden process persists across session restarts. No oracle/counter-
    factual endpoints are exposed to policies.
    """

    def __init__(self, config: OODWorldConfig):
        self._config = config
        # Different RNG namespaces avoid accidentally sharing a stream across
        # development and heldout even when a fixture uses the same integer.
        offset = 0 if config.split == "development" else 2**32
        self._rng = random.Random(config.seed + offset)
        permitted = DEVELOPMENT_RULES if config.split == "development" else HELDOUT_RULES
        self._rule_index = self._rng.randrange(len(permitted))
        self._permitted = permitted
        self._rule = permitted[self._rule_index]
        self._context = self._rng.randrange(3)
        self._time = 0
        self._actions_in_session = 0
        self._shifted = False
        self._symbol = self._sense(trusted=False)

    def _sense(self, *, trusted: bool) -> int:
        if trusted:
            return self._context
        noise = 0.50 if self._config.scenario == "noisy" else 0.12
        if self._rng.random() < noise:
            return (self._context + self._rng.choice((1, 2))) % 3
        return self._context

    def observe(self) -> PublicObservation:
        return PublicObservation(
            symbol=self._symbol,
            steps_remaining=self._config.session_horizon - self._actions_in_session,
        )

    def restart_session(self) -> PublicObservation:
        if self._actions_in_session != self._config.session_horizon:
            raise RuntimeError("session must terminate before restart")
        self._actions_in_session = 0
        self._symbol = self._sense(trusted=False)
        return self.observe()

    def step(self, action: int) -> FactualTransition:
        if type(action) is not int or action not in range(4):
            raise ValueError("expected action 0,1,2 or probe 3")
        if self._actions_in_session >= self._config.session_horizon:
            raise RuntimeError("session ended")
        before = self.observe()
        if self._config.scenario == "shift" and not self._shifted and self._time >= 9:
            self._rule_index = (self._rule_index + 1) % len(self._permitted)
            self._rule = self._permitted[self._rule_index]
            self._shifted = True
        fee = 0.10 if action == 3 else 0.0
        if action == 3:
            reward = 0.0
            self._symbol = self._sense(trusted=True)
        else:
            reward = float(action == self._rule[self._context])
            self._context = (self._context + 1) % 3
            self._symbol = self._sense(trusted=False)
        self._time += 1
        self._actions_in_session += 1
        return FactualTransition(
            before=before, action=action, after=self.observe(),
            reward=reward, fee=fee, time=self._time,
            provenance=f"ood-factual-{self._time}",
            done=self._actions_in_session == self._config.session_horizon,
        )


class PublicPolicy(Protocol):
    def choose(self, observation: PublicObservation) -> int: ...
    def observe(self, event: FactualTransition) -> None: ...
    def restart_session(self, *, erase_persistent: bool = False) -> None: ...


def run_ood_sessions(
    policy: PublicPolicy, config: OODWorldConfig, *,
    sessions: int = 3, erase_persistent: bool = False,
) -> tuple[tuple[FactualTransition, ...], ...]:
    """Give each policy its own world, passing ONLY public factual records.

    No outcome is assigned to an unchosen action. Identical world seeds do NOT
    ensure identical RNG trajectories once policies select different probes.
    This intentionally returns traces, NOT result scores or promotion gates.
    """
    if type(sessions) is not int or not 1 <= sessions <= 32:
        raise ValueError("invalid session count")
    world = OODRuleWorld(config)  # The policy never receives this reference.
    traces: list[tuple[FactualTransition, ...]] = []
    for session in range(sessions):
        if session:
            world.restart_session()
            policy.restart_session(erase_persistent=erase_persistent)
        trace: list[FactualTransition] = []
        for _ in range(config.session_horizon):
            public_before = world.observe()
            action = policy.choose(public_before)
            # The choice must be made before the factual feedback is seen.
            event = world.step(action)
            policy.observe(event)
            trace.append(event)
        traces.append(tuple(trace))
    return tuple(traces)


def engineering_fixture_configs() -> tuple[OODWorldConfig, ...]:
    """Tiny structural fixtures; NOT preregistered heldout scientific seeds.

    Do not score, tune, or reuse these as scientific replication attempts.
    The values exist exclusively for CI and contract-level unit tests.
    """
    return (
        OODWorldConfig(1, "development", "stable"),
        OODWorldConfig(2, "development", "shift"),
        OODWorldConfig(3, "heldout", "stable"),
        OODWorldConfig(4, "heldout", "noisy"),
        OODWorldConfig(5, "heldout", "long"),
    )
