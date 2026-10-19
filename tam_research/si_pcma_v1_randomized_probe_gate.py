"""Original SI PCMA-v1 paid-probe FIRST-ACTION assignment audit.

DEVELOPMENT ONLY. The policy action is randomized BEFORE outcomes, then the
same immutable PUBLIC continuation acts once. This audits whether choosing the
paid sensor rather than a UNIFORMLY randomized guess improves realized
*two-step* net reward. It is NOT a learned-PCMA assessment, and NEVER shows
hidden simulator state, alternate-arm outcomes or heldout data.

The randomized treatment is assigned in a stateful sequential world. The
Horvitz-Thompson difference estimates an average immediate first-action
"blip" effect under the RANDOMIZED LOGGING HISTORY; it is NOT the causal
value of a multi-session adaptive probing policy or a stable-world oracle.
Independent world instances, rather than correlated action-pairs, are the
units used for exploratory uncertainty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

from tam_research.si_pcma_v1_probe_randomization import (
    FrozenPublicContinuation, RandomizedFactualPair,
)

WORLD_SEEDS = tuple(range(401, 425))
SCENARIOS = ("stable", "shift", "noisy", "long")
SESSIONS_PER_WORLD = 8
RANDOMIZATION_FIXTURE = 271828
# 24 *disjoint dev world instances*, exactly 6 per task family.
EXPECTED_WORLD_COUNT = len(WORLD_SEEDS)
T_CRITICAL_23 = 2.069  # two-sided 95%, 23 independent development-world units
T_CRITICAL_5 = 2.571   # two-sided 95%, 6 independent development worlds/family


@dataclass(frozen=True)
class WorldProbeContrast:
    world_index: int
    scenario: str
    assigned_probe: int
    assigned_guess: int
    total_factual_pairs: int
    # Horvitz-Thompson effect, includes actual paid probe and missed guess.
    probe_minus_uniform_guess: float
    unweighted_probe_mean: float
    unweighted_guess_mean: float


@dataclass(frozen=True)
class ScenarioProbeContrast:
    scenario: str
    independent_worlds: int
    pair_count: int
    mean_probe_minus_uniform_guess: float
    developer_95_interval: tuple[float, float]


@dataclass(frozen=True)
class ProbeDeveloperReadout:
    status: str
    estimand: str
    worlds: int
    randomized_factual_pairs: int
    sessions_per_world: int
    first_action_probability_probe: float
    first_action_probability_any_guess: float
    fixed_second_action: tuple[int, int, int]
    paid_probe_fee: float
    mean_probe_minus_uniform_guess: float
    developer_95_interval: tuple[float, float]
    descriptive_robust_negative: bool
    per_scenario: tuple[ScenarioProbeContrast, ...]
    per_world: tuple[WorldProbeContrast, ...]
    learned_neural_models_evaluated: int
    scientific_seeds_consumed: int
    heldout_worlds_evaluated: int
    scientific_gate_passed: bool
    may_claim_probe_is_useful: bool
    may_claim_si_breakthrough: bool


def _dev_interval(values: Sequence[float], t_critical: float) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("need independent world replication")
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean)**2 for v in values) / (len(values) - 1))
    halfwidth = t_critical * sd / math.sqrt(len(values))
    return (mean-halfwidth, mean+halfwidth)


def estimate_randomized_probe_blip(
    pairs: Sequence[RandomizedFactualPair], scenarios: Sequence[str],
    *, sessions_per_world: int,
) -> ProbeDeveloperReadout:
    """Public-factual-only direct assignment effect for a FIXED continuation.

    With probe P=1/4, random guess P=3/4 and fixed second-action mapping:
      Y*[4 I(first=probe) - (4/3) I(first in guesses)]
    is a Horvitz-Thompson first-stage assignment contrast conditional on the
    previously randomized action history. It does NOT compare potential
    returns for an identical latent state by seeing unexecuted-arm outcomes.
    As first actions alter later hidden contexts, this is *not* long-run
    value of an adaptive sensor-acquisition policy.
    """
    if (type(sessions_per_world) is not int or not 1 <= sessions_per_world <= 32
            or not scenarios or len(scenarios) < 2 or not pairs
            or any(s not in SCENARIOS for s in scenarios)):
        raise ValueError("invalid bounded development-only world evidence")
    world_count = len(scenarios)
    grouped: list[list[RandomizedFactualPair]] = [
        [] for _ in range(world_count)]
    seen: set[tuple[int, int, int]] = set()
    previous_group: tuple[int, int, int] | None = None
    last_times: dict[int, int] = {}
    for row in pairs:
        if type(row) is not RandomizedFactualPair:
            raise TypeError("real randomized pair required")
        key = (row.world_index, row.session_index, row.pair_index)
        if (any(type(z) is not int or z < 0 for z in key)
                or row.world_index >= world_count
                or row.session_index >= sessions_per_world
                or key in seen
                or (previous_group is not None and key <= previous_group)):
            raise ValueError("duplicate, unordered or unregistered world/group")
        first, second = row.first_event, row.second_event
        action = row.assigned_first_action
        if (type(action) is not int or action not in range(4)
                or row.assignment_probability != .25
                or type(row.continuation_actions) is not tuple
                or row.continuation_actions != (0, 1, 2)
                or action != first.action or first.before != row.first_observation
                or first.done or second.time != first.time+1
                or second.before != first.after
                or second.action != row.continuation_actions[second.before.symbol]
                or second.action not in range(3)
                or first.fee != (.1 if action == 3 else 0.)
                or second.fee != 0.
                or first.reward not in (0.,1.) or second.reward not in (0.,1.)
                or (action == 3 and first.reward != 0.)
                or not math.isfinite(row.observed_net)
                or not -.1 <= row.observed_net <= 2.):
            raise ValueError("unrandomized, posthoc, future or invalid factual pair")
        if (row.world_index in last_times
                and first.time != last_times[row.world_index] + 1):
            raise ValueError("time gap or duplicate within original world")
        last_times[row.world_index] = second.time
        previous_group = key
        seen.add(key)
        grouped[row.world_index].append(row)
    if any(not x for x in grouped):
        raise ValueError("missing independent developer world")
    world_rows = []
    for i, (scenario, rows) in enumerate(zip(scenarios, grouped)):
        if len(rows) % sessions_per_world != 0:
            raise ValueError("incomplete within-world session evidence")
        if len({x.session_index for x in rows}) != sessions_per_world:
            raise ValueError("missing complete world session")
        size = len(rows)//sessions_per_world
        if size not in (3, 8) or (scenario=="long") != (size == 8):
            raise ValueError("incorrect 6-/16-step world factual-pair budget")
        if any(len([r for r in rows if r.session_index==session])!=size
               for session in range(sessions_per_world)):
            raise ValueError("incomplete session action pairing")
        count_probe = sum(r.assigned_first_action == 3 for r in rows)
        count_guess = len(rows)-count_probe
        if not count_probe or not count_guess:
            raise ValueError("randomized arms missing in a developer world")
        # HT uses all pairs, rather than conditioning on realized arm counts,
        # and is valid for adaptive sequential randomized logging blip estimand.
        contrast = sum(
            r.observed_net * (4. if r.assigned_first_action == 3 else -4./3.)
            for r in rows
        ) / len(rows)
        probe_mean = sum(r.observed_net for r in rows
                         if r.assigned_first_action == 3) / count_probe
        guess_mean = sum(r.observed_net for r in rows
                         if r.assigned_first_action != 3) / count_guess
        world_rows.append(WorldProbeContrast(
            i, scenario, count_probe, count_guess, len(rows),
            contrast, probe_mean, guess_mean))
    scenario_rows = []
    for name in SCENARIOS:
        current = [r for r in world_rows if r.scenario==name]
        if len(current) != 6:
            raise ValueError("expected six independent worlds per family")
        effects = [r.probe_minus_uniform_guess for r in current]
        scenario_rows.append(ScenarioProbeContrast(
            name, len(current), sum(r.total_factual_pairs for r in current),
            sum(effects)/len(effects),
            _dev_interval(effects, T_CRITICAL_5)))
    effects = [r.probe_minus_uniform_guess for r in world_rows]
    mean = sum(effects)/len(effects)
    interval = _dev_interval(effects, T_CRITICAL_23)
    # Explicit EXPLORATORY developer-only no-scale triage, not scientific
    # accept/reject test. CI above zero would still NOT certify probe success.
    clearly_negative = (interval[1] < -.05 and
                        all(s.developer_95_interval[1] < 0.
                            for s in scenario_rows))
    return ProbeDeveloperReadout(
        status="DESCRIPTIVE_RANDOMIZED_DEVELOPMENT_ONLY",
        estimand="two-step probe-vs-uniform-guess assignment blip under sequential randomized logging and fixed public continuation",
        worlds=len(world_rows),
        randomized_factual_pairs=len(pairs),
        sessions_per_world=sessions_per_world,
        first_action_probability_probe=.25,
        first_action_probability_any_guess=.75,
        fixed_second_action=(0, 1, 2),
        paid_probe_fee=.1,
        mean_probe_minus_uniform_guess=mean,
        developer_95_interval=interval,
        descriptive_robust_negative=clearly_negative,
        per_scenario=tuple(scenario_rows),
        per_world=tuple(world_rows),
        learned_neural_models_evaluated=0,
        scientific_seeds_consumed=0,
        heldout_worlds_evaluated=0,
        scientific_gate_passed=False,
        may_claim_probe_is_useful=False,
        may_claim_si_breakthrough=False,
    )


def frozen_development_probe_screen() -> ProbeDeveloperReadout:
    """The only fixed bounded set of developer seeds this PR runs.

    None is available for scientific reuse, and 401..424 are NOT external
    heldout seeds. Does not run an original or adapted neural backbone.
    """
    from tam_research.si_pcma_v1_ood_world import OODWorldConfig
    from tam_research.si_pcma_v1_probe_randomization import (
        randomized_development_first_actions,
    )
    configs = tuple(
        OODWorldConfig(seed, "development", SCENARIOS[(seed - 401) % 4])
        for seed in WORLD_SEEDS
    )
    randomized = randomized_development_first_actions(
        configs,
        fixture_randomization_seed=RANDOMIZATION_FIXTURE,
        continuation=FrozenPublicContinuation((0, 1, 2)),
        sessions=SESSIONS_PER_WORLD,
    )
    return estimate_randomized_probe_blip(
        randomized, tuple(c.scenario for c in configs),
        sessions_per_world=SESSIONS_PER_WORLD,
    )


def main() -> None:
    print("SI PCMA ORIGINAL PAID PROBE — RANDOMIZED DEV BLIP ONLY", flush=True)
    print("NOT long-run adaptive probe value; no neural model; no scientific run",flush=True)
    out = frozen_development_probe_screen()
    print(json.dumps(asdict(out), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
