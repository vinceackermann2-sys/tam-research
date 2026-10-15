from __future__ import annotations

"""#1450 prospective grouped statistical readiness, zero data construction.

Pure stdlib. This module NEVER imports CHM generators or neural models,
opens a reserved development/test panel, trains, or issues a scientific verdict.
Scores, if ever provided, require an independently source-locked authorized
future evaluation; current tests use invented synthetic observations only.
"""

from dataclasses import dataclass
import math
from typing import Literal

CLASSIFICATION = "CHM_V4_1450_GROUPED_FAIR_BASELINE_PROSPECTIVE_GATE"
GLOBAL_ONE_SIDED_ALPHA = 0.05
FAMILIES = ("direct", "overwrite", "two_hop")
EXPECTED_TEST_ENTITY_CLUSTERS_PER_FAMILY = 32
EXPECTED_NEGATIVE_ENTITY_CLUSTERS = 32
VARIANTS_PER_CLUSTER = 8
CODES = tuple(range(18001, 18009))
BLIND_ACCURACY = 1.0 / 8.0
# 3 family chance comparisons, 3 family paired comparisons and one FPR.
SIMULTANEOUS_CLAIMS = 7
ONE_SIDED_ALPHA_PER_CLAIM = GLOBAL_ONE_SIDED_ALPHA / SIMULTANEOUS_CLAIMS
SCIENTIFIC_ATTEMPT_AUTHORIZED = False
GPU_AUTHORIZED = False
OLD_100M_STATUS = "CHM_V3_100M_DAEC_STAGE_C_STOP"
OLD_SCIENTIFIC_SEED_CONSUMED = 2013161


@dataclass(frozen=True)
class FairnessAttestation:
    """Unverified assertions requiring independent provenance before use.

    Marking every field true is NOT itself evidence of parity; this is a
    fail-closed protocol completeness check for a separately verified report.
    """
    same_model_visible_encoder: bool
    same_train_examples_and_order: bool
    comparable_total_train_and_inference_compute: bool
    matched_gold_source_address_supervision: bool
    precommitted_unchanged_optimizer: bool
    exact_source_and_first_attempt_evidence: bool


@dataclass(frozen=True)
class PositiveEntityCluster:
    """Eight paired answers from one entity. These are correlated, NOT n=8."""
    family: Literal["direct", "overwrite", "two_hop"]
    entity_id: str
    gold_codes: tuple[int, ...]
    experimental_correct: tuple[bool, ...]
    fair_transformer_correct: tuple[bool, ...]


@dataclass(frozen=True)
class NegativeEntityCluster:
    """One independent absent-query entity. Eight repeats count as ONE."""
    entity_id: str
    false_positive_by_variant: tuple[bool, ...]


def _flag(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("all outcomes and attestation fields must be actual booleans")
    return value


def _attestation_gate(attestation: FairnessAttestation) -> None:
    if not isinstance(attestation, FairnessAttestation):
        raise TypeError("fairness attestation required")
    if not all(_flag(v) for v in vars(attestation).values()):
        raise ValueError("unmatched supervision/compute/source invalidates superiority claim")


def hoeffding_one_sided_lower(
    mean: float, independent_cluster_count: int, width: float, alpha: float,
    *, floor: float,
) -> float:
    """Finite-sample one-sided independent-cluster lower confidence bound.

    For bounded independent cluster means X_i in [floor,floor+width], using
    E[X] >= mean - width*sqrt(log(1/alpha)/(2*n)). Very conservative.
    """
    if (type(independent_cluster_count) is not int or independent_cluster_count < 1
        or not all(math.isfinite(v) for v in (mean,width,alpha,floor))
        or width <= 0 or not 0 < alpha < 1
        or not floor <= mean <= floor+width):
        raise ValueError("invalid bounded cluster mean/alpha")
    penalty = width * math.sqrt(math.log(1.0 / alpha)/(2*independent_cluster_count))
    return max(floor, mean-penalty)


def exact_binomial_upper(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson U: Pr[X <= k | p=U] = alpha.

    Numerical bisection uses only small binomial sums (n=32 here).
    Assumes n independent negative *entities*, never 8*n variants.
    """
    if (type(k) is not int or type(n) is not int
        or n < 1 or not 0 <= k <= n
        or not isinstance(alpha,(float,int)) or not 0 < alpha < 1):
        raise ValueError("invalid independent negative count/alpha")
    if k == n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(85):
        p = (low+high)/2
        # sum_{j=0}^k choose(n,j) p^j (1-p)^(n-j).
        cdf = sum(math.comb(n,j) * p**j * (1-p)**(n-j)
                  for j in range(k+1))
        if cdf > alpha:
            low = p
        else:
            high = p
    return (low+high)/2


def prospective_grouped_screen(
    positives: tuple[PositiveEntityCluster, ...],
    negatives: tuple[NegativeEntityCluster, ...],
    attestation: FairnessAttestation,
) -> dict[str, object]:
    """Hypothetical report arithmetic, NOT permission to construct real data.

    Caller must independently verify provenance and consumption metadata.
    Exact 32 clusters per family, eight counterfactuals within each.
    """
    _attestation_gate(attestation)
    positive_stats = {}
    for family in FAMILIES:
        entries = [r for r in positives if r.family == family]
        if len(entries) != EXPECTED_TEST_ENTITY_CLUSTERS_PER_FAMILY:
            raise ValueError("requires 32 independent positive entity groups per family")
        if len({e.entity_id for e in entries}) != len(entries):
            raise ValueError("reused within-family independent entity cluster")
        scores = []
        paired = []
        for e in entries:
            if (type(e.entity_id) is not str or not e.entity_id
                or len(e.gold_codes) != VARIANTS_PER_CLUSTER
                or len(e.experimental_correct) != VARIANTS_PER_CLUSTER
                or len(e.fair_transformer_correct) != VARIANTS_PER_CLUSTER
                or any(type(v) is not int for v in e.gold_codes)
                or set(e.gold_codes) != set(CODES)):
                raise ValueError("incomplete or unbalanced eight-way group")
            experimental = sum(_flag(v) for v in e.experimental_correct)/8
            baseline = sum(_flag(v) for v in e.fair_transformer_correct)/8
            scores.append(experimental)
            paired.append(experimental-baseline)
        accuracy = sum(scores)/len(scores)
        difference = sum(paired)/len(paired)
        chance_lower = hoeffding_one_sided_lower(
            accuracy,len(scores),1.0,ONE_SIDED_ALPHA_PER_CLAIM,floor=0.0,
        )
        paired_lower = hoeffding_one_sided_lower(
            difference,len(paired),2.0,ONE_SIDED_ALPHA_PER_CLAIM,floor=-1.0,
        )
        positive_stats[family] = {
            "independent_entities":len(scores),
            "variants_per_entity":VARIANTS_PER_CLUSTER,
            "mean_experimental_accuracy":accuracy,
            "mean_accuracy_advantage_over_paired_fair_transformer":difference,
            "simultaneous_lower_accuracy":chance_lower,
            "simultaneous_lower_paired_advantage":paired_lower,
            "above_1_in_8_chance_gate":chance_lower>BLIND_ACCURACY,
            "above_fair_transformer_gate":paired_lower>0.0,
        }
    if len(positives) != len(FAMILIES)*EXPECTED_TEST_ENTITY_CLUSTERS_PER_FAMILY:
        raise ValueError("unknown positive family")
    if len(negatives) != EXPECTED_NEGATIVE_ENTITY_CLUSTERS:
        raise ValueError("requires exactly 32 independent no-match entities")
    if len({r.entity_id for r in negatives}) != len(negatives):
        raise ValueError("duplicate absent-query entity")
    false_positives = 0
    for r in negatives:
        if type(r.entity_id) is not str or not r.entity_id:
            raise ValueError("invalid negative identity")
        outcomes = tuple(_flag(x) for x in r.false_positive_by_variant)
        if len(outcomes) != VARIANTS_PER_CLUSTER or len(set(outcomes))!=1:
            raise ValueError("no-match variants are same visible example; inconsistent outputs")
        false_positives += int(outcomes[0])
    upper = exact_binomial_upper(
        false_positives,len(negatives),ONE_SIDED_ALPHA_PER_CLAIM,
    )
    screen_pass = (all(
        x["above_1_in_8_chance_gate"] and x["above_fair_transformer_gate"]
        for x in positive_stats.values()
    ) and upper < 0.05)
    return {
        "classification":CLASSIFICATION,
        "global_alpha":GLOBAL_ONE_SIDED_ALPHA,
        "simultaneous_claim_count":SIMULTANEOUS_CLAIMS,
        "per_claim_alpha":ONE_SIDED_ALPHA_PER_CLAIM,
        "positive":positive_stats,
        "no_match_independent_entities":len(negatives),
        "no_match_false_positive_entities":false_positives,
        "no_match_exact_upper_false_positive_rate":upper,
        "below_five_percent_false_positive_gate":upper<0.05,
        "all_screen_gates_passed":screen_pass,
        "screen_is_not_breakthrough_or_replication":True,
        "requires_future_separate_replication_and_compute_utility":True,
        "no_data_construction_or_training":True,
        "historical_100m_status":OLD_100M_STATUS,
        "historical_science_seed_consumed":OLD_SCIENTIFIC_SEED_CONSUMED,
    }


def prospective_capacity_limits() -> dict[str, float | int | bool]:
    """Data-free mathematical limit for the currently reserved test set."""
    no_error_upper = exact_binomial_upper(
        0, EXPECTED_NEGATIVE_ENTITY_CLUSTERS, ONE_SIDED_ALPHA_PER_CLAIM,
    )
    independent_n_for_uncorrected_5_percent = math.ceil(
        math.log(GLOBAL_ONE_SIDED_ALPHA)/math.log(1-0.05)
    )
    simultaneous_n = math.ceil(
        math.log(ONE_SIDED_ALPHA_PER_CLAIM)/math.log(1-0.05)
    )
    return {
        "negative_entities_available":EXPECTED_NEGATIVE_ENTITY_CLUSTERS,
        "one_sided_95pct_uncorrected_zero_error_upper":exact_binomial_upper(
            0,EXPECTED_NEGATIVE_ENTITY_CLUSTERS,GLOBAL_ONE_SIDED_ALPHA,
        ),
        "simultaneous_zero_error_upper":no_error_upper,
        "can_claim_fpr_below_5_percent_with_32_and_zero_errors":no_error_upper<0.05,
        "minimum_independent_negative_entities_zero_errors_uncorrected":
            independent_n_for_uncorrected_5_percent,
        "minimum_independent_negative_entities_zero_errors_simultaneous":
            simultaneous_n,
    }
