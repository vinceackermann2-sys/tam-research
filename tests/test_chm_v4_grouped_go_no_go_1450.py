from __future__ import annotations

"""#1450 prospective bounds tested ONLY on fabricated numeric outcomes."""

from dataclasses import replace
import ast
import inspect
import math

import pytest

import tam_research.chm_v4_grouped_go_no_go_1450 as gates


def fair() -> gates.FairnessAttestation:
    return gates.FairnessAttestation(True,True,True,True,True,True)


def made_up(accuracy: int, baseline: int = 1):
    """Synthetic calibration, never a CHM benchmark episode."""
    positives = tuple(
        gates.PositiveEntityCluster(
            family=family,entity_id=f"UNREAL-PLACEHOLDER-{i:03}",
            gold_codes=gates.CODES,
            experimental_correct=tuple(j < accuracy for j in range(8)),
            fair_transformer_correct=tuple(j < baseline for j in range(8)),
        )
        for family in gates.FAMILIES for i in range(32)
    )
    negatives=tuple(
        gates.NegativeEntityCluster(
            entity_id=f"UNREAL-ABSENT-{i:03}",
            false_positive_by_variant=(False,)*8,
        ) for i in range(32)
    )
    return positives,negatives


def test_exact_hypothetical_8_way_balancing_counts():
    positives,negatives=made_up(8)
    report=gates.prospective_grouped_screen(positives,negatives,fair())
    assert report["classification"]==gates.CLASSIFICATION
    assert report["simultaneous_claim_count"]==7
    assert report["per_claim_alpha"]==pytest.approx(0.05/7)
    assert report["no_data_construction_or_training"] is True
    assert report["requires_future_separate_replication_and_compute_utility"] is True
    for f in gates.FAMILIES:
        x=report["positive"][f]
        assert x["independent_entities"]==32
        assert x["variants_per_entity"]==8
        assert x["mean_experimental_accuracy"]==1.0
        assert x["mean_accuracy_advantage_over_paired_fair_transformer"]==.875
        assert x["above_1_in_8_chance_gate"]
        assert x["above_fair_transformer_gate"]
    assert report["below_five_percent_false_positive_gate"] is False
    assert report["all_screen_gates_passed"] is False
    assert report["screen_is_not_breakthrough_or_replication"] is True
    assert report["historical_100m_status"]=="CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert report["historical_science_seed_consumed"]==2013161


def test_32_zero_error_negative_entities_cannot_prove_5pct_fpr():
    limits=gates.prospective_capacity_limits()
    assert limits["negative_entities_available"]==32
    assert limits["can_claim_fpr_below_5_percent_with_32_and_zero_errors"] is False
    assert limits["one_sided_95pct_uncorrected_zero_error_upper"]==pytest.approx(
        1-0.05**(1/32),abs=1e-12,
    )
    assert limits["simultaneous_zero_error_upper"]==pytest.approx(
        1-(0.05/7)**(1/32),abs=1e-12,
    )
    assert limits["minimum_independent_negative_entities_zero_errors_uncorrected"]==59
    assert limits["minimum_independent_negative_entities_zero_errors_simultaneous"] >= 90


@pytest.mark.parametrize("k,n",((0,32),(1,32),(4,32),(31,32),(32,32)))
def test_exact_binomial_one_sided_upper_tail(k,n):
    u=gates.exact_binomial_upper(k,n,.05)
    assert 0 <= u <= 1
    if k==0:
        assert u==pytest.approx(1-.05**(1/n),abs=1e-12)
    elif k==n:
        assert u==1
    else:
        cdf=sum(math.comb(n,i)*u**i*(1-u)**(n-i) for i in range(k+1))
        assert cdf==pytest.approx(.05,abs=1e-12)


def test_only_independent_entities_are_n_not_8_times_larger():
    p,n=made_up(2,0)
    r=gates.prospective_grouped_screen(p,n,fair())
    assert r["positive"]["direct"]["independent_entities"]==32
    assert r["positive"]["direct"]["mean_experimental_accuracy"]==.25
    assert r["positive"]["direct"]["above_1_in_8_chance_gate"] is False
    assert r["positive"]["direct"]["above_fair_transformer_gate"] is False
    with pytest.raises(ValueError,match="32 independent"):
        gates.prospective_grouped_screen(p[:-1],n,fair())
    with pytest.raises(ValueError,match="32 independent no-match"):
        gates.prospective_grouped_screen(p,n[:-1],fair())


def test_refuses_duplicate_and_incomplete_entity_clusters():
    p,n=made_up(8,0)
    with pytest.raises(ValueError,match="reused within-family"):
        gates.prospective_grouped_screen(
            (replace(p[0],entity_id=p[1].entity_id),)+p[1:],n,fair(),
        )
    with pytest.raises(ValueError,match="duplicate absent-query"):
        gates.prospective_grouped_screen(
            p,(replace(n[0],entity_id=n[1].entity_id),)+n[1:],fair(),
        )
    with pytest.raises(ValueError,match="unbalanced"):
        gates.prospective_grouped_screen(
            (replace(p[0],gold_codes=(18001,)*8),)+p[1:],n,fair(),
        )
    with pytest.raises(ValueError,match="unbalanced"):
        gates.prospective_grouped_screen(
            (replace(p[0],experimental_correct=(True,)),)+p[1:],n,fair(),
        )


def test_denies_unmatched_address_ce_and_unverified_provenance():
    p,n=made_up(8)
    with pytest.raises(ValueError,match="unmatched"):
        gates.prospective_grouped_screen(
            p,n,replace(fair(),matched_gold_source_address_supervision=False),
        )
    with pytest.raises(ValueError,match="unmatched"):
        gates.prospective_grouped_screen(
            p,n,replace(fair(),exact_source_and_first_attempt_evidence=False),
        )
    with pytest.raises(ValueError,match="actual booleans"):
        gates.prospective_grouped_screen(
            p,n,replace(fair(),comparable_total_train_and_inference_compute=1),
        )


def test_negative_same_visible_example_repeats_must_have_same_answer():
    p,n=made_up(8)
    with pytest.raises(ValueError,match="inconsistent outputs"):
        gates.prospective_grouped_screen(
            p,(replace(n[0],false_positive_by_variant=(True,False)*4),)+n[1:],
            fair(),
        )
    with pytest.raises(ValueError,match="actual booleans"):
        gates.prospective_grouped_screen(
            p,(replace(n[0],false_positive_by_variant=(0,)*8),)+n[1:],
            fair(),
        )


@pytest.mark.parametrize("k,n,alpha",((-1,32,.05),(33,32,.05),(1,0,.05),
                                    (0,32,0),(0,32,1),(False,32,.05)))
def test_rejects_invalid_binomial_inputs(k,n,alpha):
    with pytest.raises(ValueError):
        gates.exact_binomial_upper(k,n,alpha)


@pytest.mark.parametrize("mean,n,width,alpha,floor",(
    (0.5,0,1,.05,0), (0.5,32,0,.05,0), (0.5,32,1,0,0),
    (0.5,32,1,1,0), (1.5,32,1,.05,0), (float("nan"),32,1,.05,0),
))
def test_refuses_invalid_hoeffding(mean,n,width,alpha,floor):
    with pytest.raises(ValueError):
        gates.hoeffding_one_sided_lower(mean,n,width,alpha,floor=floor)


def test_no_historical_data_generation_in_source_or_tests():
    assert gates.GPU_AUTHORIZED is False
    assert gates.SCIENTIFIC_ATTEMPT_AUTHORIZED is False
    src=inspect.getsource(gates)
    tree=ast.parse(src)
    imp=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            imp.append(node.module)
        if isinstance(node,ast.Import):
            imp.extend(a.name for a in node.names)
    assert imp==["dataclasses","math","typing"]
    prohibited={"torch", "modal", "tensorflow", "chm_v4_independent_binding_benchmark_1421"}
    assert not prohibited.intersection(imp)
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    for forbidden in ("generate_episode","train_counterfactual_group","model.forward",
                      "torch.cuda","optimizer.step","torch.load"):
        assert forbidden not in calls
