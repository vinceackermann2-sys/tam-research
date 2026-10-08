from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "research/reduced_attention/attention8_pair1_successor_frozen_metrics_v1.json"
DOC = ROOT / "research/reduced_attention/attention8_pair1_successor_postmortem_v1.md"


def test_attention8_frozen_primary_provenance_and_fail_closed():
    obj = json.loads(METRICS.read_text(encoding="utf-8"))
    source, exp, final = obj["source"], obj["experiment"], obj["final"]
    assert source["commit"] == "27a3a15cb8456523798163f4cfa5a829844477ed"
    assert source["tree"] == "027e0c1bbc845783d547f935f378889d298c9dab"
    assert source["harness"] == "5d9e1a170d9bbb93e4a02de6a47467049df10d55"
    assert source["terminal_result_sha256"] == "872daab1193742942ef01fd8dee77080a07272eee4b55aa0b9236cffb584d88d"
    assert (source["scientific_issue"], source["observer_issue"]) == (1300, 1331)
    assert exp["seed"] == 60232 and exp["seed_consumed"] is True
    assert exp["classification"] == "SCIENTIFIC_ATTENTION8_V2_PAIR1_SUCCESSOR_SCREEN_FAIL"
    assert exp["model_exposures"] == 2000027648 and exp["optimizer_steps"] == 30518
    assert exp["candidate_attention_layers"] == [3, 6, 9, 12, 15, 18, 21, 24]
    assert final["transformer"]["status"] == "COMPLETE"
    assert final["attention8"]["status"] == "COMPLETE"
    assert final["quality_gate_pass"] is False
    assert final["systems_gate_pass"] is True
    assert final["combined_progression_gate_pass"] is False
    assert final["quality_limit"] == 0.015 and final["throughput_minimum"] == 1.03
    assert math.isclose(final["attention8"]["nll"] - final["transformer"]["nll"], final["delta_nll"], abs_tol=1e-12)
    assert math.isclose(final["attention8"]["tps"] / final["transformer"]["tps"], final["throughput_ratio"], abs_tol=1e-12)
    assert final["delta_nll"] > final["quality_limit"]
    assert final["throughput_ratio"] >= final["throughput_minimum"]


def test_attention8_periodic_curve_is_matched_and_descriptive():
    obj = json.loads(METRICS.read_text(encoding="utf-8"))
    curve = obj["periodic_curve"]
    assert len(curve) == 10
    exposures = [x["exposures"] for x in curve]
    assert exposures == sorted(exposures) and len(set(exposures)) == 10
    assert exposures[-1] == obj["experiment"]["model_exposures"]
    deltas = [x["attention8_nll"] - x["transformer_nll"] for x in curve]
    assert all(x > 0 for x in deltas)
    assert math.isclose(deltas[0], 0.006631374359130859, abs_tol=1e-12)
    assert math.isclose(deltas[-1], 0.04330445528030369, abs_tol=1e-12)
    assert sum(delta > 0.015 for delta in deltas) == 9
    assert obj["engineering"]["not_scientific_evidence"] is True
    text = DOC.read_text(encoding="utf-8")
    assert "not apples-to-apples" in text
    assert "No experiment code or Modal volume files are modified" in text
    assert "Do not" in text
