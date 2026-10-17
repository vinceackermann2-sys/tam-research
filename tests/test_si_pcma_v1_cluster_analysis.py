"""No simulated scientific trials; validates WORLD-cluster attribution only."""
from dataclasses import replace
import math

import pytest

from tam_research.si_pcma_v1_ood_world import OODWorldConfig
from tam_research.si_pcma_v1_probe_randomization import (
    randomized_development_first_actions, FrozenPublicContinuation,
)
from tam_research.si_pcma_v1_cluster_analysis import randomized_probe_contrast


def _fixtures():
    # Pure unit-fixture integers; NOT preregistered scientific attempts.
    return randomized_development_first_actions(
        tuple(OODWorldConfig(i + 100, "development", "stable") for i in range(4)),
        fixture_randomization_seed=17, sessions=2,
    )


def test_counts_worlds_not_actions_as_independent_units():
    rows = _fixtures()
    effect = randomized_probe_contrast(rows)
    assert effect.independent_worlds == 4
    assert effect.factual_two_action_opportunities == len(rows) == 24
    assert len(effect.world_contributions) == 4
    assert sum(effect.pooled_arm_counts) == len(rows)
    assert effect.comparison == "probe_vs_uniform_guesses"
    assert effect.world_cluster_standard_error >= 0
    assert math.isfinite(effect.mean_world_probe_effect)


def test_per_world_ht_terms_equal_only_actual_assigned_returns():
    rows = _fixtures()
    stats = randomized_probe_contrast(rows)
    for world in stats.world_contributions:
        selected = [r for r in rows if r.world_index == world.world_index]
        n = len(selected)
        probe = sum(4*r.observed_net for r in selected
                    if r.assigned_first_action == 3)/n
        guesses = sum(4*r.observed_net for r in selected
                      if r.assigned_first_action in range(3))/(3*n)
        assert world.probe_ht_mean == pytest.approx(probe)
        assert world.baseline_ht_mean == pytest.approx(guesses)
        assert world.probe_minus_baseline == pytest.approx(probe-guesses)


def test_zero_arms_remain_zero_weighted_not_imputed():
    rows = _fixtures()
    effect = randomized_probe_contrast(rows)
    assert effect.complete_worlds_with_probe == sum(
        x.arm_counts[3] > 0 for x in effect.world_contributions
    )
    assert effect.complete_worlds_with_all_guesses == sum(
        all(x.arm_counts[a] > 0 for a in range(3))
        for x in effect.world_contributions
    )
    assert len(effect.pooled_arm_counts) == 4


def test_fixed_guess_contrasts_work_without_posthoc_max():
    rows = _fixtures()
    effects = [randomized_probe_contrast(rows, baseline=f"guess_{i}") for i in range(3)]
    average = sum(x.mean_world_probe_effect for x in effects)/3
    mixture = randomized_probe_contrast(rows, baseline="uniform_guesses")
    assert average == pytest.approx(mixture.mean_world_probe_effect)


def test_rejects_replayed_world_pair_instead_of_pseudoreplicating():
    rows = _fixtures()
    with pytest.raises(ValueError, match="duplicated"):
        randomized_probe_contrast(rows + (rows[0],))


def test_rejects_forged_propensity_fee_and_factual_continuation():
    rows = _fixtures()
    orig = rows[0]
    for modified in (
        replace(orig, assignment_probability=0.5),
        replace(orig, first_event=replace(orig.first_event, fee=9.0)),
        replace(orig, second_event=replace(orig.second_event, action=3)),
        replace(orig, second_event=replace(orig.second_event, time=1000)),
    ):
        with pytest.raises(ValueError):
            randomized_probe_contrast((modified,) + rows[1:])


def test_mixed_frozen_continuation_is_not_a_common_estimand():
    rows = _fixtures()
    other = randomized_development_first_actions(
        (OODWorldConfig(777, "development", "stable"),),
        fixture_randomization_seed=8,
        continuation=FrozenPublicContinuation((1,2,0)),
    )
    # Give the additional fixture an independent world id.
    additional = tuple(replace(x, world_index=99) for x in other)
    with pytest.raises(ValueError, match="mixed continuation"):
        randomized_probe_contrast(rows + additional)


def test_statistics_fail_closed_without_multiple_independent_worlds():
    rows = _fixtures()
    single_world = tuple(x for x in rows if x.world_index == 0)
    with pytest.raises(ValueError, match="few independent"):
        randomized_probe_contrast(single_world)
    with pytest.raises(ValueError):
        randomized_probe_contrast((), min_independent_worlds=2)
    with pytest.raises(ValueError):
        randomized_probe_contrast(rows, min_independent_worlds=1)
    with pytest.raises(ValueError):
        randomized_probe_contrast(rows, baseline="best_posthoc")


def test_per_world_standard_error_uses_world_variance_not_pair_variance():
    result = randomized_probe_contrast(_fixtures())
    vals = [v.probe_minus_baseline for v in result.world_contributions]
    mean = sum(vals)/len(vals)
    expected = math.sqrt(sum((v-mean)**2 for v in vals) /
                         ((len(vals)-1)*len(vals)))
    assert result.world_cluster_standard_error == pytest.approx(expected)


def test_pure_estimator_does_not_import_simulator_or_model_training():
    import tam_research.si_pcma_v1_cluster_analysis as module
    assert not hasattr(module, "OODRuleWorld")
    assert not hasattr(module, "OriginalSIHybrid")
    assert not hasattr(module, "TransformerControl")
