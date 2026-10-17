"""Development-only randomized first-action factual acquisition logs.

NO confirmation runs, heldout evaluation, experiment-statistic promotion,
privileged simulator labels or hypothetical unchosen returns.
Only the first action of each adjacent pair is randomized uniformly and
logged with its known propensity; the second action is chosen by a completely
frozen PUBLIC continuation policy applied equally to every first action.

The continuation callback is trusted research code, not a sandbox. Its only
inputs are public observation and the action/outcome actually observed.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import OODRuleWorld, OODWorldConfig


@dataclass(frozen=True)
class RandomizedAcquisitionRecord:
    world_id: int
    session_id: int
    before: PublicObservation
    first_action: int
    first_propensity: float
    first_event: FactualTransition
    second_event: FactualTransition
    realized_two_action_net: float

    def __post_init__(self) -> None:
        a, b = self.first_event, self.second_event
        if (type(self.first_action) is not int or self.first_action not in range(4)
                or self.first_propensity != 0.25):
            raise ValueError("first action must be uniformly randomized")
        if a.before != self.before or a.action != self.first_action:
            raise ValueError("record must match factual first execution")
        if a.done or b.time != a.time + 1 or b.before != a.after:
            raise ValueError("not adjacent same-session factual transitions")
        if b.action not in range(3):
            raise ValueError("continuation may only guess, never buy a second probe")
        if a.fee != (0.1 if a.action == 3 else 0.0) or b.fee != 0.0:
            raise ValueError("invalid executed-action fees")
        if a.reward not in (0.0, 1.0) or b.reward not in (0.0, 1.0):
            raise ValueError("invalid rewards")
        if self.realized_two_action_net != (a.reward - a.fee + b.reward - b.fee):
            raise ValueError("two-action target must equal ONLY observed returns")


class FrozenSymbolContinuation:
    """Explicitly fixed public-context rule, no privileged hidden rule family.

    The preselected mapping is development engineering scaffolding, and MUST
    be replaced by a preregistered/frozen learned continuation if appropriate.
    No training, update, persistent memory, or simulator reference.
    """
    def __init__(self, mapping: tuple[int, int, int] = (0, 1, 2)):
        if len(mapping) != 3 or any(type(a) is not int or a not in range(3) for a in mapping):
            raise ValueError("continuation mapping must contain three guesses")
        self.mapping = mapping

    def __call__(self, observation: PublicObservation,
                 first: FactualTransition) -> int:
        if type(observation) is not PublicObservation or type(first) is not FactualTransition:
            raise TypeError("public factual inputs required")
        if observation != first.after:
            raise ValueError("continuation must use actual next observation")
        return self.mapping[observation.symbol]


def collect_randomized_development_acquisition(
    configs: Sequence[OODWorldConfig],
    *, draw_fixture_seed: int,
    sessions_per_world: int = 2,
    continuation: Callable[[PublicObservation, FactualTransition], int] | None = None,
) -> tuple[RandomizedAcquisitionRecord, ...]:
    """Randomize first action *before* seeing that action's outcome.

    This is a development-only logger; disallow any heldout rule config prior
    to creating worlds, random draws or continuations. Each pair's second
    action is selected by the same frozen continuation callable; never adapt
    it during collection. Every returned label is a realized chosen-action
    two-step return, not an estimate of unchosen-arm counterfactual reward.
    """
    if not configs or type(sessions_per_world) is not int or not 1 <= sessions_per_world <= 32:
        raise ValueError("invalid development collection budget")
    if type(draw_fixture_seed) is not int or draw_fixture_seed < 0:
        raise ValueError("invalid engineering RNG seed")
    if any(c.split != "development" for c in configs):
        raise ValueError("heldout world forbidden from development logs")
    if len({(c.seed, c.scenario) for c in configs}) != len(configs):
        raise ValueError("duplicate development world")
    if any(c.session_horizon % 2 for c in configs):
        raise ValueError("two-step logger requires even session horizon")
    fn = continuation if continuation is not None else FrozenSymbolContinuation()
    if not callable(fn):
        raise TypeError("continuation must be a frozen public callable")
    rng = random.Random(draw_fixture_seed)
    output: list[RandomizedAcquisitionRecord] = []
    for world_i, cfg in enumerate(configs):
        world = OODRuleWorld(cfg)
        for session_i in range(sessions_per_world):
            if session_i:
                world.restart_session()
            for _ in range(cfg.session_horizon // 2):
                before = world.observe()
                # Four equal-probability actions; propensity is known BEFORE
                # executing, not estimated retrospectively from rewards.
                first_action = rng.randrange(4)
                e1 = world.step(first_action)
                second_action = fn(e1.after, e1)
                if type(second_action) is not int or second_action not in range(3):
                    raise ValueError("fixed continuation returned an invalid guess")
                e2 = world.step(second_action)
                # No policy is supplied the world object or hidden rule.
                output.append(RandomizedAcquisitionRecord(
                    world_i, session_i, before, first_action, 0.25, e1, e2,
                    float(e1.reward - e1.fee + e2.reward - e2.fee),
                ))
    return tuple(output)


@dataclass(frozen=True)
class ArmDescription:
    action: int
    factual_count: int
    sample_mean: float | None
    importance_weighted_total: float
    effective_weight_sum: float


def describe_randomized_arms(
    records: Sequence[RandomizedAcquisitionRecord],
    *, symbol: int | None = None,
) -> tuple[ArmDescription, ...]:
    """Purely descriptive arm coverage and realized returns, no significance.

    Uniform propensity makes IPS adjusted totals equal 4*observed totals.
    Never interpret differences as a preregistered effect, let alone an
    independent confirmation across repeated within-world trajectories.
    """
    if not records:
        raise ValueError("no factual randomized records")
    if symbol is not None and symbol not in range(3):
        raise ValueError("invalid public symbol")
    visible = [r for r in records if symbol is None or r.before.symbol == symbol]
    return tuple(
        ArmDescription(
            action=a,
            factual_count=len(rows := [r for r in visible if r.first_action == a]),
            sample_mean=(sum(r.realized_two_action_net for r in rows)/len(rows))
                        if rows else None,
            importance_weighted_total=sum(
                r.realized_two_action_net/r.first_propensity for r in rows),
            effective_weight_sum=sum(1/r.first_propensity for r in rows),
        )
        for a in range(4)
    )


def arm_support_complete(
    records: Sequence[RandomizedAcquisitionRecord],
    *, min_each: int,
) -> bool:
    """Fail-closed diagnostic; no implicit substituted attempts or thresholds."""
    if type(min_each) is not int or min_each < 1:
        raise ValueError("support threshold must be positive")
    if not records:
        return False
    for s in range(3):
        if any(a.factual_count < min_each for a in describe_randomized_arms(records,symbol=s)):
            return False
    return True
