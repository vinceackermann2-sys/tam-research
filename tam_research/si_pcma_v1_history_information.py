"""History-information FALSIFICATION gate for the original SI PCMA successor.

Only DEVELOPMENT-world factual randomized chosen-action first/second pairs.
Never accesses hidden rules, counterfactual outcomes, heldout generator,
scientific seed reservations, training accelerators or approval flags.

Primary use: ask whether a *cheap history-conditioned* factual forecaster
predicts actually observed two-step returns any better than a current-symbol
lookup or the existing public-history ridge. This is a descriptive developer
diagnostic; it is NOT an independent scientific effect estimate or proof that
history is useful for policy decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from tam_research.si_pcma_v1 import FactualTransition, PublicObservation
from tam_research.si_pcma_v1_factual_learning import CausalRewardSample
from tam_research.si_pcma_v1_history_world import randomized_history_development_pairs
from tam_research.si_pcma_v1_neural_probe import randomized_pairs_to_causal_samples
from tam_research.si_pcma_v1_ood_world import OODWorldConfig
from tam_research.si_pcma_v1_probe_randomization import FrozenPublicContinuation
from tam_research.si_pcma_v1_strong_controls import (
    ClassicalFactualTwoStepPredictor, FrozenClassicalTwoStepPolicy,
    fit_classical_two_step, public_two_step_features,
)


def _last_feedback_key(
    obs: PublicObservation, action: int,
    prior: Sequence[FactualTransition], *, logical_time: int | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Current factual public state/action + preceding factual response.

    The last event's observed reward and action are legitimate *past*
    feedback. No latent state and no current/future training target.
    """
    public_two_step_features(obs, action, prior, logical_time=logical_time)
    last = prior[-1] if prior else None
    if last is None:
        return (obs.symbol, action, -1, -1, -1, -1)
    if last.reward not in (0., 1.) or last.action not in range(4):
        raise ValueError("last event is not valid factual feedback")
    return (
        obs.symbol, action, last.before.symbol, last.action,
        int(last.reward), last.after.symbol,
    )


@dataclass(frozen=True)
class LastFeedbackBackoff:
    """Cheap public-history count model, smoothed toward public-symbol Q.

    A sufficiently distinct recent observation/action/outcome can shift an
    estimate, while sparse cells back off to the SAME learned symbol-table
    baseline. No recurrent neural weights or privileged model.
    """
    symbol: ClassicalFactualTwoStepPredictor
    keys: tuple[tuple[int, int, int, int, int, int], ...]
    counts: tuple[int, ...]
    totals: tuple[float, ...]
    pseudocount: float
    training_rows: int

    def predict(
        self, obs: PublicObservation, action: int,
        prior: Sequence[FactualTransition],
    ) -> float:
        key = _last_feedback_key(obs, action, prior)
        backoff = self.symbol.predict(obs, action, prior)
        try:
            idx = self.keys.index(key)
        except ValueError:
            return backoff
        return (self.totals[idx] + self.pseudocount * backoff) / (
            self.counts[idx] + self.pseudocount
        )


def fit_last_feedback_backoff(
    samples: Sequence[CausalRewardSample], *,
    pseudocount: float = 2.0,
) -> LastFeedbackBackoff:
    if (type(pseudocount) not in (float, int)
            or not math.isfinite(pseudocount) or pseudocount <= 0):
        raise ValueError("invalid history smoothing")
    # The symbol fit validates actual-armed public samples and target bounds.
    symbol = fit_classical_two_step("symbol", samples)
    table: dict[tuple[int, int, int, int, int, int], list[float]] = {}
    for s in samples:
        if type(s) is not CausalRewardSample:
            raise TypeError("factual samples required")
        key = _last_feedback_key(s.observation, s.action, s.prior,
                                 logical_time=s.logical_time)
        observed = float(s.observed_net_reward)
        if not math.isfinite(observed) or not -.1 <= observed <= 2.0:
            raise ValueError("nonfactual two-step return")
        table.setdefault(key, []).append(observed)
    keys = tuple(sorted(table))
    return LastFeedbackBackoff(
        symbol=symbol, keys=keys,
        counts=tuple(len(table[k]) for k in keys),
        totals=tuple(sum(table[k]) for k in keys),
        pseudocount=float(pseudocount),
        training_rows=len(samples),
    )


class FrozenLastFeedbackTwoStepPolicy(FrozenClassicalTwoStepPolicy):
    """Exactly the existing public-feedback, continuation and PCMA budget.

    Inherits factual first/second phase enforcement, online memory and rank-2
    adaptation, probe fee validation, persistence ablations and reset logic.
    Only the frozen cheap first-action forecasting function differs.
    """

    def __init__(self, predictor: LastFeedbackBackoff, **kwargs: object):
        if type(predictor) is not LastFeedbackBackoff:
            raise TypeError("cheap factual last-feedback predictor required")
        super().__init__(predictor.symbol, **kwargs)
        self.history_predictor = predictor

    def choose(self, observation: PublicObservation) -> int:
        if self._pending is not None:
            raise RuntimeError("unresolved prior chosen action")
        if self._first_event is not None:
            action = self.continuation.choose_guess(observation)
        else:
            choices = (0, 1, 2, 3) if (
                self.allow_probe and observation.steps_remaining > 1
            ) else (0, 1, 2)
            scores = {}
            for candidate in choices:
                base = self.history_predictor.predict(
                    observation, candidate, self.history)
                pcma = self.state.value(observation, candidate)
                scores[candidate] = (
                    (1.0 - self.pcma_weight) * base + self.pcma_weight * pcma
                )
            action = max(choices, key=lambda a: scores[a])
        self._pending = (observation, action)
        return action


@dataclass(frozen=True)
class DevelopmentWorldRisk:
    """One independent validation *world*, never one independent action."""
    world_index: int
    scenario: str
    factual_pairs: int
    constant_mse: float
    symbol_mse: float
    public_ridge_mse: float
    last_feedback_mse: float


@dataclass(frozen=True)
class DevelopmentInformationAudit:
    status: str
    train_worlds: int
    validation_worlds: int
    train_factual_pairs: int
    validation_factual_pairs: int
    per_world: tuple[DevelopmentWorldRisk, ...]
    mean_constant_mse: float
    mean_symbol_mse: float
    mean_public_ridge_mse: float
    mean_last_feedback_mse: float
    last_feedback_minus_symbol_mse: float
    passed_scientific_gate: bool
    may_evaluate_heldout: bool
    may_claim_architecture_gain: bool


def development_history_information_audit(
    train: Sequence[OODWorldConfig],
    validation: Sequence[OODWorldConfig],
    *, training_randomization_fixture: int = 23,
    validation_randomization_fixture: int = 24,
    sessions: int = 2,
) -> DevelopmentInformationAudit:
    """Report *descriptive* disjoint-development-world prediction errors.

    Same logged chosen arms and immutable second-guess policy are used for
    all four forecasting controls. Only the TRAIN worlds build predictors.
    Validation world errors are grouped by independent world and averaged
    equally, not by action (which would falsely increase sample size).
    No p-values, confidence intervals, optimal oracle, model ranking or
    claims about active compute / scientific generalization.
    """
    configs = tuple(train) + tuple(validation)
    if (not train or not validation
            or any(type(c) is not OODWorldConfig or c.split != "development"
                   for c in configs)):
        raise ValueError("both partitions must be development worlds")
    train_ids = {c.seed for c in train}
    validation_ids = {c.seed for c in validation}
    if (len(train_ids) != len(train) or len(validation_ids) != len(validation)
            or train_ids.intersection(validation_ids)):
        raise ValueError("overlapping or duplicated independent world IDs")
    if (type(sessions) is not int or not 1 <= sessions <= 8
            or type(training_randomization_fixture) is not int
            or type(validation_randomization_fixture) is not int
            or min(training_randomization_fixture,
                   validation_randomization_fixture) < 0):
        raise ValueError("invalid bounded developer data collection plan")
    # Always validate *all* split membership and IDs BEFORE constructing any
    # training/evaluation world, so forbidden heldout cannot run even once.
    fixed = FrozenPublicContinuation()
    train_pairs = randomized_history_development_pairs(
        train, fixture_randomization_seed=training_randomization_fixture,
        continuation=fixed, sessions=sessions)
    validation_pairs = randomized_history_development_pairs(
        validation, fixture_randomization_seed=validation_randomization_fixture,
        continuation=fixed, sessions=sessions)
    train_samples = randomized_pairs_to_causal_samples(train_pairs)
    test_samples = randomized_pairs_to_causal_samples(validation_pairs)

    baselines = {
        kind: fit_classical_two_step(kind, train_samples)
        for kind in ("constant", "symbol", "ridge")
    }
    hist = fit_last_feedback_backoff(train_samples)

    groups: list[list[CausalRewardSample]] = [[] for _ in validation]
    for pair, sample in zip(validation_pairs, test_samples):
        groups[pair.world_index].append(sample)
    if any(not g for g in groups):
        raise ValueError("missing factual validation world")
    records: list[DevelopmentWorldRisk] = []
    for i, (cfg, samples) in enumerate(zip(validation, groups)):
        def mse(predictor: object) -> float:
            squared = [
                (float(predictor.predict(s.observation, s.action, s.prior))
                 - s.observed_net_reward) ** 2
                for s in samples
            ]
            if not squared or not all(math.isfinite(x) for x in squared):
                raise ValueError("nonfinite development prediction audit")
            return sum(squared) / len(squared)
        records.append(DevelopmentWorldRisk(
            world_index=i, scenario=cfg.scenario,
            factual_pairs=len(samples),
            constant_mse=mse(baselines["constant"]),
            symbol_mse=mse(baselines["symbol"]),
            public_ridge_mse=mse(baselines["ridge"]),
            last_feedback_mse=mse(hist),
        ))
    def average(field: str) -> float:
        return sum(getattr(row, field) for row in records) / len(records)
    symbol_mse = average("symbol_mse")
    recent_mse = average("last_feedback_mse")
    return DevelopmentInformationAudit(
        status="DESCRIPTIVE_DEVELOPMENT_ONLY",
        train_worlds=len(train), validation_worlds=len(validation),
        train_factual_pairs=len(train_pairs),
        validation_factual_pairs=len(validation_pairs),
        per_world=tuple(records),
        mean_constant_mse=average("constant_mse"),
        mean_symbol_mse=symbol_mse,
        mean_public_ridge_mse=average("public_ridge_mse"),
        mean_last_feedback_mse=recent_mse,
        last_feedback_minus_symbol_mse=recent_mse-symbol_mse,
        passed_scientific_gate=False,
        may_evaluate_heldout=False,
        may_claim_architecture_gain=False,
    )


def print_fixed_development_screen() -> None:
    """Emit one developer-only descriptive report; NEVER an approval.

    The configuration is hardcoded for reviewability. This is NOT a unique
    scientific attempt or independent unseen-law confirmation. Re-running it
    does not generate an independent replication. No heldout rule is created.
    """
    from dataclasses import asdict
    import json

    train = tuple(OODWorldConfig(seed, "development", scenario)
                  for seed, scenario in (
                      (11, "stable"), (12, "shift"), (13, "noisy"), (14, "long"),
                      (15, "stable"), (16, "shift"), (17, "noisy"), (18, "long")))
    validation = tuple(OODWorldConfig(seed, "development", scenario)
                       for seed, scenario in (
                           (31, "stable"), (32, "shift"), (33, "noisy"),
                           (34, "long"), (35, "stable"), (36, "shift"),
                           (37, "noisy"), (38, "long")))
    report = development_history_information_audit(
        train, validation,
        training_randomization_fixture=23,
        validation_randomization_fixture=24, sessions=2,
    )
    print("NONCONFIRMATORY SI PCMA HISTORY DEVELOPMENT DIAGNOSTIC")
    print("NOT scientific validation; NOT a novel seed reservation; NOT a PCMA verdict")
    print(json.dumps(asdict(report), sort_keys=True, indent=2,
                     allow_nan=False))


if __name__ == "__main__":
    print_fixed_development_screen()
