"""Cluster-aware descriptive analysis of randomized factual first actions.

This is a PURE analysis of existing actual randomized logs (no simulator
execution, research seeds or neural training). An entire independently drawn
world is one observational cluster: repeat opportunities within a world are
NEVER treated as independent samples.

Horvitz-Thompson estimates identify the immediate two-action randomized
assignment effect for the common, frozen, public continuation policy and the
sampled on-policy context distribution. They DO NOT establish a general
causal effect of buying information, an optimal probe strategy, or SI.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

from tam_research.si_pcma_v1_probe_randomization import RandomizedFactualPair


Baseline = Literal["uniform_guesses", "guess_0", "guess_1", "guess_2"]


@dataclass(frozen=True)
class WorldContribution:
    world_index: int
    opportunity_count: int
    arm_counts: tuple[int, int, int, int]
    probe_ht_mean: float
    baseline_ht_mean: float
    probe_minus_baseline: float


@dataclass(frozen=True)
class ClusterDescriptiveEffect:
    """No p-value, CI, hypothesis gate or promotion authority."""
    comparison: str
    independent_worlds: int
    factual_two_action_opportunities: int
    complete_worlds_with_probe: int
    complete_worlds_with_all_guesses: int
    pooled_arm_counts: tuple[int, int, int, int]
    mean_world_probe_effect: float
    between_world_sd: float
    world_cluster_standard_error: float
    world_contributions: tuple[WorldContribution, ...]


def _validate(records: Sequence[RandomizedFactualPair]) -> dict[int, list[RandomizedFactualPair]]:
    if not records:
        raise ValueError("cannot analyze an empty randomized factual log")
    grouped: dict[int, list[RandomizedFactualPair]] = {}
    keys = set()
    continuations = set()
    for row in records:
        if type(row) is not RandomizedFactualPair:
            raise TypeError("requires genuine randomized factual pair objects")
        key = (row.world_index, row.session_index, row.pair_index)
        if any(type(x) is not int or x < 0 for x in key) or key in keys:
            raise ValueError("invalid or duplicated world/session/pair")
        keys.add(key)
        a, b = row.first_event, row.second_event
        if (row.assignment_probability != 0.25
                or row.assigned_first_action not in range(4)
                or a.action != row.assigned_first_action
                or a.before != row.first_observation
                or a.done or b.time != a.time + 1
                or b.before != a.after):
            raise ValueError("invalid randomized causal transition")
        if (a.fee != (0.1 if a.action == 3 else 0.0)
                or b.fee != 0.0
                or a.reward not in (0.0, 1.0)
                or b.reward not in (0.0, 1.0)
                or (a.action == 3 and a.reward != 0.0)):
            raise ValueError("forged or mispriced factual reward")
        if (len(row.continuation_actions) != 3
                or set(row.continuation_actions) != {0, 1, 2}
                or b.action != row.continuation_actions[b.before.symbol]):
            raise ValueError("changed, privileged or malformed continuation")
        continuations.add(row.continuation_actions)
        grouped.setdefault(row.world_index, []).append(row)
    if len(continuations) != 1:
        raise ValueError("mixed continuation policies invalidate common estimand")
    return grouped


def randomized_probe_contrast(
    records: Sequence[RandomizedFactualPair],
    *,
    baseline: Baseline = "uniform_guesses",
    min_independent_worlds: int = 2,
) -> ClusterDescriptiveEffect:
    """Report world-clustered Horvitz-Thompson contrasts only.

    World contributions weight each independent world equally, irrespective
    of the number of 6- versus 16-step sessions. Within each world an arm's HT
    score is mean_{opportunities}[1(A=arm) * factual_net / 0.25].
    The uniform-guesses baseline is fixed 1/3 of guesses 0,1,2, NOT a
    retrospective max-guess baseline selected after seeing outcomes.

    Standard errors use the *independent worlds* as units; reporting SE does
    not license inference if world seeds, models or follow-up policies were
    tuned on these engineering fixtures. No scientific test is run here.
    """
    if baseline not in ("uniform_guesses", "guess_0", "guess_1", "guess_2"):
        raise ValueError("baseline must be fixed before viewing outcomes")
    if type(min_independent_worlds) is not int or min_independent_worlds < 2:
        raise ValueError("at least two independent worlds required for cluster SE")
    grouped = _validate(records)
    if len(grouped) < min_independent_worlds:
        raise ValueError("too few independent worlds for cluster analysis")

    parts: list[WorldContribution] = []
    for wid, rows in sorted(grouped.items()):
        n = len(rows)
        counts = tuple(sum(r.assigned_first_action == action for r in rows)
                       for action in range(4))
        scores = tuple(sum(
            r.observed_net / r.assignment_probability
            for r in rows if r.assigned_first_action == action
        ) / n for action in range(4))
        control = (sum(scores[:3]) / 3 if baseline == "uniform_guesses"
                   else scores[int(baseline[-1])])
        parts.append(WorldContribution(
            wid, n, counts, scores[3], control, scores[3] - control,
        ))

    effects = [x.probe_minus_baseline for x in parts]
    worlds = len(parts)
    mean = sum(effects) / worlds
    variance = sum((x - mean) ** 2 for x in effects) / (worlds - 1)
    return ClusterDescriptiveEffect(
        comparison=f"probe_vs_{baseline}",
        independent_worlds=worlds,
        factual_two_action_opportunities=sum(x.opportunity_count for x in parts),
        complete_worlds_with_probe=sum(x.arm_counts[3] > 0 for x in parts),
        complete_worlds_with_all_guesses=sum(all(y > 0 for y in x.arm_counts[:3])
                                             for x in parts),
        pooled_arm_counts=tuple(sum(x.arm_counts[a] for x in parts) for a in range(4)),
        mean_world_probe_effect=mean,
        between_world_sd=math.sqrt(variance),
        world_cluster_standard_error=math.sqrt(variance / worlds),
        world_contributions=tuple(parts),
    )
