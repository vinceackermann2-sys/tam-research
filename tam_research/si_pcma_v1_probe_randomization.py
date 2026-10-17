"""Development-only randomized FIRST-action factual logger for PCMA-v1.

This is not a confirmation harness. It refuses heldout rules, uses a fixed
PUBLIC continuation rule, randomizes first actions with known equal probability,
and logs only actually executed rewards. No world/private rule is exposed
to the continuation. Samples must NOT be represented as causal proof without
preregistered independent worlds, strata, uncertainty and stopping rules.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import OODRuleWorld, OODWorldConfig


@dataclass(frozen=True)
class FrozenPublicContinuation:
    """No training, privileged world knowledge, or future observation."""
    action_by_symbol: tuple[int, int, int] = (0, 1, 2)

    def __post_init__(self) -> None:
        if len(self.action_by_symbol) != 3 or set(self.action_by_symbol) != {0, 1, 2}:
            raise ValueError("continuation must be a fixed guess permutation")

    def choose_guess(self, observation: PublicObservation) -> int:
        if type(observation) is not PublicObservation or observation.symbol not in range(3):
            raise ValueError("not a public OOD observation")
        return self.action_by_symbol[observation.symbol]


@dataclass(frozen=True)
class RandomizedFactualPair:
    # Stable grouping units: repeated actions in one world are NOT independent.
    world_index: int
    session_index: int
    pair_index: int
    first_observation: PublicObservation
    assigned_first_action: int
    assignment_probability: float
    first_event: FactualTransition
    second_event: FactualTransition
    continuation_actions: tuple[int, int, int]

    @property
    def observed_net(self) -> float:
        return (self.first_event.reward - self.first_event.fee +
                self.second_event.reward - self.second_event.fee)


def randomized_development_first_actions(
    configs: Sequence[OODWorldConfig],
    *,
    fixture_randomization_seed: int,
    continuation: FrozenPublicContinuation = FrozenPublicContinuation(),
    sessions: int = 2,
) -> tuple[RandomizedFactualPair, ...]:
    """First-stage arm uniformly random; second-stage rule fixed in advance.

    A probe uses a whole action and 0.1 fee. Session horizons 6/16 are even,
    so pairs never cross a session boundary. This function is DEVELOPMENT
    ENGINEERING ONLY; heldout requests raise before creating any world.
    """
    if not configs or type(sessions) is not int or not 1 <= sessions <= 32:
        raise ValueError("invalid development collection plan")
    if any(c.split != "development" for c in configs):
        raise ValueError("heldout rules forbidden in randomized development logger")
    if len({(c.seed, c.scenario) for c in configs}) != len(configs):
        raise ValueError("duplicate development fixture")
    if type(fixture_randomization_seed) is not int or fixture_randomization_seed < 0:
        raise ValueError("invalid fixture randomization seed")
    if type(continuation) is not FrozenPublicContinuation:
        raise ValueError("requires frozen public continuation")

    rng = random.Random(fixture_randomization_seed)
    pairs: list[RandomizedFactualPair] = []
    for world_index, cfg in enumerate(configs):
        world = OODRuleWorld(cfg)
        for session in range(sessions):
            if session:
                world.restart_session()
            for pair_index in range(cfg.session_horizon // 2):
                before = world.observe()
                assigned = rng.randrange(4)  # Known P(A=a | public history) = 1/4
                first = world.step(assigned)
                second_guess = continuation.choose_guess(world.observe())
                second = world.step(second_guess)
                if first.done or second.time != first.time + 1:
                    raise RuntimeError("invalid first-action acquisition pair")
                pairs.append(RandomizedFactualPair(
                    world_index=world_index,
                    session_index=session,
                    pair_index=pair_index,
                    first_observation=before,
                    assigned_first_action=assigned,
                    assignment_probability=0.25,
                    first_event=first,
                    second_event=second,
                    continuation_actions=continuation.action_by_symbol,
                ))
    return tuple(pairs)


@dataclass(frozen=True)
class DescriptiveArmValue:
    symbol: int
    action: int
    count: int
    observed_mean_net: float | None


def descriptive_randomized_arm_values(
    pairs: Sequence[RandomizedFactualPair],
) -> tuple[DescriptiveArmValue, ...]:
    """Descriptive arm means by pre-action observed symbol; no p-values.

    Under registered randomization and fixed continuation, such estimates
    *could* support a causal first-action analysis in a separate study.
    These developer fixtures do not estimate variance across independent
    worlds and MUST NOT be used for a scientific advancement decision.
    """
    if not pairs:
        raise ValueError("no actual randomized acquisitions")
    table: dict[tuple[int, int], list[float]] = {
        (symbol, action): [] for symbol in range(3) for action in range(4)
    }
    seen: set[tuple[int, int, int]] = set()
    fixed_continuations = set()
    for p in pairs:
        if type(p) is not RandomizedFactualPair:
            raise TypeError("expected factual randomized pair")
        first, second = p.first_event, p.second_event
        if (p.assigned_first_action != first.action or
                p.first_observation != first.before or
                second.time != first.time + 1 or
                p.assignment_probability != 0.25):
            raise ValueError("invalid randomized assignment / factual pair")
        if second.action not in range(3) or first.done:
            raise ValueError("invalid continuation or pair")
        if p.continuation_actions[second.before.symbol] != second.action:
            raise ValueError("continuation changed after randomization")
        # Each world has its own clock: provenance alone is not globally unique;
        # do not treat per-world timestamps as independent-world identifiers.
        group = (p.world_index, p.session_index, p.pair_index)
        if group in seen or any(type(x) is not int or x < 0 for x in group):
            raise ValueError("duplicate or invalid independent-world grouping key")
        seen.add(group)
        fixed_continuations.add(p.continuation_actions)
        table[(p.first_observation.symbol, first.action)].append(p.observed_net)
    if len(fixed_continuations) != 1:
        raise ValueError("mixed continuation policies")
    return tuple(
        DescriptiveArmValue(s, a, len(table[(s, a)]),
                            sum(table[(s, a)]) / len(table[(s, a)])
                            if table[(s, a)] else None)
        for s in range(3) for a in range(4)
    )
