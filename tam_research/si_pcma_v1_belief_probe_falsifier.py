"""Known-FAMILY Bayesian cheap control for original PCMA development world.

This classical policy has an explicit SIMULATOR-SHAPED prior: three cyclic
candidate rule permutations, known context-transition physics, possible
one-time shift at clock 9, and two candidate observation-noise levels.
That structural prior is NOT available by default to the neural contenders.
It never receives the selected rule, true latent context, task scenario, or
unexecuted rewards, and never accesses the world instance. Thus it is a
*model-informed cheap reference*, NOT proof of neural inferiority.

For each nonoverlapping two-action decision it computes exact conditional
belief-space expected immediate reward+second-step optimal GUESS:
  Q(first) = -fee + E[real first guess reward if any
                       + best second GUESS reward after observing feedback].
A probe consumes first action and 0.1; a guess also provides factual reward.
Only the first step may probe. No synthetic alternative outcome is ever
passed to the actually deployed controller or scored as reward.

All evaluation IDs and outputs are NONCONFIRMATORY developer fixtures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_ood_world import (
    OODWorldConfig, run_ood_sessions,
)
from tam_research.si_pcma_v1_ood_controls import (
    SlidingWindowFactualControl, audit_factual_trace,
)

Hypothesis = tuple[int, int, int, int]  # rule index, hidden context, noise flag, shift flag
CYCLIC_FAMILY = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
NOISE_LEVELS = (0.12, 0.50)
HYPOTHESES: tuple[Hypothesis, ...] = tuple(
    (rule, context, noise_flag, shift_flag)
    for rule in range(3) for context in range(3)
    for noise_flag in range(2) for shift_flag in range(2)
)
DEV_SEEDS = tuple(range(501, 525))
SCENARIOS = ("stable", "shift", "noisy", "long")
SESSIONS_PER_WORLD = 4
FEE = 0.1


def _uniform_belief() -> dict[Hypothesis, float]:
    return {hyp: 1.0/len(HYPOTHESES) for hyp in HYPOTHESES}


def _sensor_probability(hyp: Hypothesis, symbol: int, *, trusted: bool) -> float:
    if type(symbol) is not int or symbol not in range(3):
        raise ValueError("public symbol out of range")
    context, noise_flag = hyp[1], hyp[2]
    if trusted:
        return float(symbol == context)
    error = NOISE_LEVELS[noise_flag]
    return (1.0-error) if symbol == context else (error/2.0)


def _next(hyp: Hypothesis, action: int, time_before_action: int) -> tuple[Hypothesis, int]:
    if (type(action) is not int or action not in range(4)
            or type(time_before_action) is not int or time_before_action < 0):
        raise ValueError("unknown public action or global time")
    rule, context, noise, shift = hyp
    # The *original* simulator changes rule on the 10th global action if
    # the shift scenario is active. Hypothesis shift is unknown to agent.
    if shift and time_before_action == 9:
        rule = (rule + 1)%3
    if action == 3:
        return ((rule, context, noise, shift), 0)
    reward = int(CYCLIC_FAMILY[rule][context] == action)
    return ((rule, (context + 1)%3, noise, shift), reward)


def _normalize(unnormalized: dict[Hypothesis, float]) -> dict[Hypothesis, float]:
    total = sum(unnormalized.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("observed feedback impossible under known-family prior")
    return {hyp: value/total for hyp, value in unnormalized.items()
            if value > 0.0}


def _branches(belief: dict[Hypothesis, float], action: int, time: int
             ) -> tuple[tuple[float, int, int, dict[Hypothesis, float]], ...]:
    """Predict possible future PUBLIC observations, not privileged outcomes.

    This is classical belief planning under a task-specific known-family
    generative model; the actual policy sees ONLY the eventual chosen branch.
    """
    data: dict[tuple[int,int], dict[Hypothesis,float]] = {}
    for hyp, probability in belief.items():
        next_hyp, reward = _next(hyp,action,time)
        for symbol in range(3):
            probability_after = probability * _sensor_probability(
                next_hyp, symbol, trusted=(action==3))
            if probability_after > 0:
                group=data.setdefault((reward,symbol), {})
                group[next_hyp]=group.get(next_hyp,0.0)+probability_after
    branches=[]
    for (reward,symbol),posterior in sorted(data.items()):
        prob=sum(posterior.values())
        branches.append((prob,reward,symbol,_normalize(posterior)))
    if abs(sum(p for p,*_ in branches)-1.0)>1e-9:
        raise ValueError("missing posterior branch probability")
    return tuple(branches)


def _best_guess(belief: dict[Hypothesis,float], time: int) -> tuple[int,float]:
    values=tuple(sum(weight*_next(hyp,guess,time)[1]
                     for hyp,weight in belief.items()) for guess in range(3))
    guess=max(range(3),key=lambda a:values[a])
    return guess,values[guess]


class KnownFamilyBeliefPolicy:
    """Explicit model-informed classical bound; NO hidden-state label access."""

    def __init__(self, *, allow_probe: bool):
        if type(allow_probe) is not bool:
            raise ValueError("probe choice ablation must be explicit")
        self.allow_probe=allow_probe
        self.belief=_uniform_belief()
        self.time=0
        self._fresh_sensor=True
        self._pending: tuple[PublicObservation,int]|None=None
        self.probes=0
        self.observed_events=0
        self.belief_updates=0

    def choose(self, observation: PublicObservation) -> int:
        if self._pending is not None:
            raise RuntimeError("must observe actual chosen reward first")
        if (type(observation) is not PublicObservation or
                observation.symbol not in range(3) or
                not 1 <= observation.steps_remaining <= 16):
            raise ValueError("only pre-action public observation")
        if self._fresh_sensor:
            self.belief=_normalize({
                h:p*_sensor_probability(h,observation.symbol,trusted=False)
                for h,p in self.belief.items()})
            self._fresh_sensor=False
        if observation.steps_remaining%2:
            action,_=_best_guess(self.belief,self.time)
        else:
            candidates=(0,1,2,3) if self.allow_probe else (0,1,2)
            values={}
            for first in candidates:
                q=-FEE if first==3 else 0.
                for prob,reward,_,updated in _branches(self.belief,first,self.time):
                    _,next_guess=_best_guess(updated,self.time+1)
                    q+=prob*(reward+next_guess)
                values[first]=q
            action=max(candidates,key=lambda a: values[a])
        if action==3 and observation.steps_remaining<2:
            raise RuntimeError("probe cannot consume terminal action")
        self._pending=(observation,action)
        return action

    def observe(self, event: FactualTransition) -> None:
        if self._pending is None:
            raise RuntimeError("cannot consume unchosen factual response")
        prior,action=self._pending
        if (type(event) is not FactualTransition or event.before != prior
                or event.action != action or event.time != self.time+1
                or event.after.steps_remaining != prior.steps_remaining-1
                or event.reward not in (0.,1.)
                or event.fee != (FEE if action==3 else 0.)
                or (action==3 and event.reward!=0.)):
            raise ValueError("mismatched, future or forged actual feedback")
        weights:dict[Hypothesis,float]={}
        for prob,reward,symbol,posterior in _branches(self.belief,action,self.time):
            if reward==int(event.reward) and symbol==event.after.symbol:
                for hyp,conditional in posterior.items():
                    weights[hyp]=weights.get(hyp,0.)+prob*conditional
        self.belief=_normalize(weights)
        self.probes+=int(action==3)
        self.time=event.time
        self.observed_events+=1
        self.belief_updates+=1
        self._pending=None

    def restart_session(self, *, erase_persistent: bool=False) -> None:
        if self._pending is not None:
            raise RuntimeError("unresolved executed action")
        if type(erase_persistent) is not bool:
            raise ValueError("explicit reset flag")
        if erase_persistent:
            self.belief=_uniform_belief()
        # The original world emits one EXTRA nonprivileged noisy observation
        # at restart with the same latent context; assimilate it exactly once.
        self._fresh_sensor=True


@dataclass(frozen=True)
class BeliefWorldComparison:
    world_index: int
    scenario: str
    actions: int
    belief_allow_probe_net: float
    belief_no_probe_net: float
    sliding_window_no_probe_net: float
    belief_allow_probe_count: int
    belief_allow_probe_minus_no_probe: float


@dataclass(frozen=True)
class BeliefDeveloperReport:
    status: str
    original_v26: str
    original_pcma_status: str
    independent_dev_worlds: int
    scenarios: tuple[str,...]
    sessions_per_world: int
    total_action_opportunities_per_policy: int
    total_belief_probe_actions: int
    mean_belief_allow_probe_minus_no_probe: float
    mean_belief_allow_probe_net: float
    mean_belief_no_probe_net: float
    mean_simple_factual_no_probe_net: float
    per_world: tuple[BeliefWorldComparison,...]
    model_informed_world_family_prior: bool
    neural_models_evaluated: int
    scientific_seeds_consumed: int
    scientific_gate_passed: bool
    learned_architecture_superiority: bool


def frozen_belief_probe_dev_screen() -> BeliefDeveloperReport:
    """Fixed development inputs, one pass; NEVER a sealed science trial."""
    records=[]
    for index,seed in enumerate(DEV_SEEDS):
        scenario=SCENARIOS[index%4]
        cfg=OODWorldConfig(seed,"development",scenario)
        belief_probe=KnownFamilyBeliefPolicy(allow_probe=True)
        belief_guess=KnownFamilyBeliefPolicy(allow_probe=False)
        # Strong simple *model-free* factual control that gets no rule prior.
        factual=SlidingWindowFactualControl(
            window=24,exploration=.10,probe_period=0,fixture_seed=13)
        def run(policy):
            result=run_ood_sessions(policy,cfg,sessions=SESSIONS_PER_WORLD)
            return audit_factual_trace(result)
        with_probe=run(belief_probe)
        no_probe=run(belief_guess)
        simple=run(factual)
        if not with_probe.actions==no_probe.actions==simple.actions:
            raise RuntimeError("unequal actual action opportunity")
        records.append(BeliefWorldComparison(
            world_index=index,scenario=scenario,actions=with_probe.actions,
            belief_allow_probe_net=with_probe.net_reward,
            belief_no_probe_net=no_probe.net_reward,
            sliding_window_no_probe_net=simple.net_reward,
            belief_allow_probe_count=with_probe.paid_probes,
            belief_allow_probe_minus_no_probe=with_probe.net_reward-no_probe.net_reward,
        ))
    if len(records)!=24 or any(x.scenario!=SCENARIOS[i%4]
                                for i,x in enumerate(records)):
        raise RuntimeError("developer fixture family count changed")
    def mean(field:str)->float:
        return sum(getattr(row,field) for row in records)/len(records)
    return BeliefDeveloperReport(
        status="EXPLORATORY_KNOWN_FAMILY_DEVELOPMENT_ONLY",
        original_v26="NO_GO", original_pcma_status="UNKNOWN",
        independent_dev_worlds=len(records),scenarios=SCENARIOS,
        sessions_per_world=SESSIONS_PER_WORLD,
        total_action_opportunities_per_policy=sum(x.actions for x in records),
        total_belief_probe_actions=sum(x.belief_allow_probe_count for x in records),
        mean_belief_allow_probe_minus_no_probe=mean("belief_allow_probe_minus_no_probe"),
        mean_belief_allow_probe_net=mean("belief_allow_probe_net"),
        mean_belief_no_probe_net=mean("belief_no_probe_net"),
        mean_simple_factual_no_probe_net=mean("sliding_window_no_probe_net"),
        per_world=tuple(records),
        model_informed_world_family_prior=True,
        neural_models_evaluated=0,
        scientific_seeds_consumed=0,
        scientific_gate_passed=False,
        learned_architecture_superiority=False,
    )


def main()->None:
    print("PCMA KNOWN-FAMILY BAYES BELIEF: PAID PROBE VS NO-PROBE (DEVELOPMENT)")
    print("Public factual transitions only; model-informed prior; NOT scientific")
    print(json.dumps(asdict(frozen_belief_probe_dev_screen()),
                     sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
