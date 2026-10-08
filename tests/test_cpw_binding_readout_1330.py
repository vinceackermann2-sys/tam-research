"""#1330: synthetic decision fixtures, NOT observations from trained CPW.

These tests mutate invented metrics to catch invalid or cherry-picked results.
No training, CUDA, Modal or scientific seeds are allocated.
"""
from __future__ import annotations

from copy import deepcopy
import math

import pytest

from tam_research.cpw_binding_readout.classifier import (
    ARMS, CANDIDATES, PARAMETERS, PROVISIONAL, screen_replications,
)


def _eval(*, acc: float, nll: float, group: float, local: bool = False):
    delays = {
        str(d): {"examples": 512, "accuracy": acc, "nll": nll}
        for d in (32, 64, 128, 256)
    }
    return {
        "by_distance": delays,
        "long_mean_accuracy": acc,
        "long_mean_nll": nll,
        "counterfactual_groups": 512,
        "counterfactual_all_four_accuracy": group,
        "counterfactual_individual_accuracy": max(acc, group),
        "keyblind_fixed_query_accuracy": 0.25,
    }


def _fixture():
    data = []
    for seed in (11, 12, 13):
        arms = {}
        for arm in ARMS:
            if arm == "transformer":
                acc, nll, group = .95, .15, .80
            elif arm == "sequence_only":
                acc, nll, group = .02, 7.0, .0
            else:
                acc, nll, group = .91, .4, .75
            arms[arm] = {
                "arm": arm,
                "seed": seed,
                "steps": 1500,
                "examples_seen": 96000,
                "parameters": PARAMETERS[arm],
                "training_seconds": 100.0,
                "total_compute_seconds": 123.0,
                "examples_per_second": 960.0,
                "peak_vram_gib": 15.0,
                "last_train_query_loss": .3,
                "evaluation": _eval(acc=acc, nll=nll, group=group),
            }
        data.append({
            "seed": seed, "phase": "replication", "source_sha": "a" * 40,
            "arms": arms,
        })
    return data


def _check_invalid(data, status="INVALID_INCOMPLETE"):
    decision = screen_replications(data)
    assert decision["status"] == status, decision
    assert decision["candidates"] == {}
    assert decision["evidence_level"] == PROVISIONAL
    assert decision["verified_scientific_run"] is False
    assert decision["breakthrough_claim_allowed"] is False
    return decision


def test_all_synthetic_quality_thresholds_pass_but_no_science_is_certified():
    result = screen_replications(_fixture())
    assert result["status"] == "VALID_CONTROLS_PROVISIONAL"
    assert result["source_sha_unverified"] == "a" * 40
    assert result["paired_seed_ids_unverified"] == [11, 12, 13]
    assert not result["scientific_gpu_authorized"]
    assert not result["verified_scientific_run"]
    assert not result["breakthrough_claim_allowed"]
    for arm in CANDIDATES:
        decision = result["candidates"][arm]
        assert decision["passed_seeds"] == 3
        assert decision["metric_classification"] == "PROVISIONAL_STRONG_3_OF_3"
        assert decision["mean_total_compute_seconds"] == 123


@pytest.mark.parametrize("mutation", [
    lambda d: d.pop(),
    lambda d: d.append(deepcopy(d[-1])),
    lambda d: d[0].update(seed=d[1]["seed"]),
    lambda d: d[0].update(seed=True),
    lambda d: d[0].update(phase="smoke"),
    lambda d: d[0].update(source_sha="z" * 40),
    lambda d: d[2].update(source_sha="b" * 40),
    lambda d: d[0]["arms"].pop("r1_final"),
    lambda d: d[0]["arms"]["transformer"].update(steps=250),
    lambda d: d[0]["arms"]["transformer"].update(examples_seen=63999),
    lambda d: d[0]["arms"]["transformer"].update(parameters=1),
    lambda d: d[0]["arms"]["transformer"].update(peak_vram_gib=-1),
    lambda d: d[0]["arms"]["transformer"].update(total_compute_seconds=1),
    lambda d: d[0]["arms"]["transformer"].update(training_seconds=float("nan")),
    lambda d: d[0]["arms"]["transformer"].update(examples_per_second=0),
    lambda d: d[0]["arms"]["transformer"]["evaluation"]["by_distance"]["64"].update(examples=511),
    lambda d: d[0]["arms"]["transformer"]["evaluation"]["by_distance"]["64"].update(nll=float("nan")),
    lambda d: d[0]["arms"]["transformer"]["evaluation"]["by_distance"]["64"].update(accuracy=1.1),
    lambda d: d[0]["arms"]["transformer"]["evaluation"].update(long_mean_accuracy=.1),
    lambda d: d[0]["arms"]["transformer"]["evaluation"].update(long_mean_nll=1),
    lambda d: d[0]["arms"]["transformer"]["evaluation"].update(counterfactual_groups=511),
    lambda d: d[0]["arms"]["transformer"]["evaluation"].update(keyblind_fixed_query_accuracy=.50),
    lambda d: d[0]["arms"]["transformer"]["evaluation"].update(counterfactual_individual_accuracy=0.1),
    lambda d: d[0]["arms"]["transformer"]["evaluation"]["by_distance"].update({"17": {"examples":512,"accuracy":1.0,"nll":0}}),
])
def test_adversarial_input_is_incomplete_not_positive(mutation):
    data = _fixture()
    mutation(data)
    _check_invalid(data)


def test_transfomer_weak_long_control_blocks_all_candidates():
    data = _fixture()
    for d in ("64", "128", "256"):
        data[0]["arms"]["transformer"]["evaluation"]["by_distance"][d]["accuracy"] = .7
    data[0]["arms"]["transformer"]["evaluation"]["long_mean_accuracy"] = .7
    decision = _check_invalid(data, "INVALID_TRANSFORMER_CONTROL")
    assert decision["transformer_invalid_seeds"] == [11]


def test_transformer_256_failure_blocks_even_when_long_mean_strong():
    data = _fixture()
    e = data[0]["arms"]["transformer"]["evaluation"]
    e["by_distance"]["256"]["accuracy"] = .69
    e["long_mean_accuracy"] = (.95 + .95 + .69) / 3
    assert e["long_mean_accuracy"] > .8
    _check_invalid(data, "INVALID_TRANSFORMER_CONTROL")


def test_leaky_negative_control_blocks_claims():
    data = _fixture()
    e = data[1]["arms"]["sequence_only"]["evaluation"]
    for d in ("64", "128", "256"):
        e["by_distance"][d]["accuracy"] = .06
    e["long_mean_accuracy"] = .06
    decision = _check_invalid(data, "INVALID_NEGATIVE_CONTROL")
    assert decision["negative_control_invalid_seeds"] == [12]


def test_two_of_three_is_supported_but_not_strong():
    data = _fixture()
    data[2]["arms"]["afm_first1"]["evaluation"]["counterfactual_all_four_accuracy"] = .68
    got = screen_replications(data)
    assert got["candidates"]["afm_first1"]["passed_seeds"] == 2
    assert got["candidates"]["afm_first1"]["metric_classification"] == (
        "PROVISIONAL_SUPPORT_2_OF_3"
    )
    assert not got["candidates"]["afm_first1"]["seed_decisions"]["13"]["passed"]


def test_one_of_three_is_no_support():
    data = _fixture()
    for i in (1, 2):
        data[i]["arms"]["afm_first1"]["evaluation"]["counterfactual_all_four_accuracy"] = .69
    result = screen_replications(data)
    assert result["candidates"]["afm_first1"]["passed_seeds"] == 1
    assert result["candidates"]["afm_first1"]["metric_classification"] == "PROVISIONAL_NO_SUPPORT"


@pytest.mark.parametrize("fail", ["long_mean", "distance256", "margin_each_long_distance",
                                  "lower_nll_each_long_distance", "counterfactual_all_four"])
def test_every_independent_binding_gate_is_required(fail):
    data = _fixture()
    e = data[0]["arms"]["afm_first1"]["evaluation"]
    if fail == "long_mean":
        for d in ("64", "128", "256"):
            e["by_distance"][d]["accuracy"] = .78
        e["long_mean_accuracy"] = .78
    elif fail == "distance256":
        e["by_distance"]["256"]["accuracy"] = .69
        e["long_mean_accuracy"] = (.91 + .91 + .69) / 3
    elif fail == "margin_each_long_distance":
        # local is 2%: >40%-advantage fails at 41%, despite high other distances
        e["by_distance"]["64"]["accuracy"] = .41
        e["long_mean_accuracy"] = (.41 + .91 + .91) / 3
    elif fail == "lower_nll_each_long_distance":
        e["by_distance"]["64"]["nll"] = 7.01
        e["long_mean_nll"] = (7.01 + .4 + .4) / 3
    elif fail == "counterfactual_all_four":
        e["counterfactual_all_four_accuracy"] = .69
    decision = screen_replications(data)
    checks = decision["candidates"]["afm_first1"]["seed_decisions"]["11"]["tests"]
    assert checks[fail] is False


def test_three_quality_passes_not_strong_when_one_seed_behind_transformer():
    data = _fixture()
    e = data[0]["arms"]["afm_first1"]["evaluation"]
    for d in ("64", "128", "256"):
        e["by_distance"][d]["accuracy"] = .88
    e["long_mean_accuracy"] = .88
    # With Transformer at .95, gap .07, outside 0.05 strong margin
    got = screen_replications(data)
    assert got["candidates"]["afm_first1"]["passed_seeds"] == 3
    assert got["candidates"]["afm_first1"]["metric_classification"] == (
        "PROVISIONAL_SUPPORT_2_OF_3"
    )


def test_mocked_metrics_cannot_verify_provenance_or_breakthrough():
    data = _fixture()
    for row in data:
        row["claims"] = {"scientific_gpu_authorized": True, "breakthrough": True}
    result = screen_replications(data)
    assert result["status"] == "VALID_CONTROLS_PROVISIONAL"
    assert not result["verified_scientific_run"]
    assert not result["breakthrough_claim_allowed"]
