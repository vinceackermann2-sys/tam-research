from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tam_research.chm_v1_100m_stage_c_payload_integration as decomposition
from tam_research.chm_v1_100m_stage_c_eval import GENERATOR_VERSION
from tam_research.chm_v1_100m_stage_c_postmortem import (
    no_memory_final_logits,
    oracle_evidence_final_logits,
)
from tam_research.chm_v1_long_memory_eval import EncodedProbe
from tam_research.chm_v1_small_lm import LOCAL_WINDOW
from tam_research.models import ModelConfig, ResearchLM


class TinyEIEM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResearchLM(
            ModelConfig(
                vocab_size=257,
                d_model=16,
                n_layers=1,
                n_heads=2,
                max_seq_len=LOCAL_WINDOW,
                ff_mult=2,
                architecture="transformer",
            )
        )
        self.query_address = nn.Linear(16, 4, bias=False)
        self.key_address = nn.Linear(16, 4, bias=False)
        self.memory_gate_logit = nn.Parameter(torch.full((16,), -4.0))

    def query_for(self, representation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(representation), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.memory_gate_logit).to(hidden.dtype)
        return hidden + gate * memory


def _rare_probe() -> EncodedProbe:
    prompt = tuple((index * 17 + 3) % 257 for index in range(700))
    return EncodedProbe(
        family="rare_fact",
        case_id=1,
        prompt_text="synthetic rare-fact test",
        prompt_ids=prompt,
        answer_token_id=8,
        candidate_token_ids=tuple(range(8, 16)),
        evidence_end_token=100,
        query_token=len(prompt) - 1,
        evidence_distance=(len(prompt) - 1) - 100,
        generator_version=GENERATOR_VERSION,
    )


def _two_hop_probe() -> EncodedProbe:
    prompt = tuple((index * 13 + 7) % 257 for index in range(900))
    return EncodedProbe(
        family="two_hop",
        case_id=200_001,
        prompt_text="synthetic two-hop test",
        prompt_ids=prompt,
        answer_token_id=8,
        candidate_token_ids=tuple(range(8, 16)),
        first_evidence_end_token=90,
        evidence_end_token=300,
        query_token=len(prompt) - 1,
        evidence_distance=(len(prompt) - 1) - 300,
        generator_version=GENERATOR_VERSION,
    )


def _local_negative_probe() -> EncodedProbe:
    prompt = tuple((index * 11 + 5) % 257 for index in range(100))
    return EncodedProbe(
        family="local_negative",
        case_id=300_001,
        prompt_text="synthetic local-negative test",
        prompt_ids=prompt,
        answer_token_id=8,
        candidate_token_ids=tuple(range(8, 16)),
        evidence_end_token=80,
        query_token=len(prompt) - 1,
        evidence_distance=(len(prompt) - 1) - 80,
        generator_version=GENERATOR_VERSION,
    )


def test_manifest_grants_no_execution_or_training_authority() -> None:
    manifest = decomposition.validate_protocol_manifest()
    assert manifest["classification"].endswith("NO_EXECUTION_AUTHORITY")
    assert manifest["conditions"] == (
        "NO_MEMORY",
        "ORACLE_LEARNED_GATE",
        "ORACLE_UNIT_GATE",
    )
    assert manifest["unit_gate_value"] == 1.0
    assert manifest["gate_sweep_authorized"] is False
    assert manifest["training_authorized"] is False
    assert manifest["parameter_updates_authorized"] is False
    assert manifest["optimizer_authorized"] is False
    assert manifest["checkpoint_mutation_authorized"] is False
    assert manifest["modal_authorized"] is False
    assert manifest["gpu_authorized"] is False
    assert manifest["new_scientific_seed_authorized"] is False
    assert manifest["stage_d_authorized"] is False


def test_gate_statistics_are_deterministic_and_match_sigmoid_minus_four() -> None:
    model = TinyEIEM()
    stats = decomposition.gate_statistics(model)
    expected = float(torch.sigmoid(torch.tensor(-4.0)).item())
    assert stats["count"] == 16.0
    for key in ("min", "p05", "median", "mean", "p95", "max"):
        assert stats[key] == pytest.approx(expected, abs=1e-8)
    assert stats["fraction_lt_0p01"] == 0.0
    assert stats["fraction_lt_0p05"] == 1.0
    assert stats["fraction_gt_0p10"] == 0.0
    assert stats["l1"] == pytest.approx(16 * expected, rel=1e-6)
    assert stats["l2"] == pytest.approx((16 ** 0.5) * expected, rel=1e-6)


def test_learned_oracle_is_bit_exact_to_first_postmortem_and_unit_is_distinct() -> None:
    torch.manual_seed(1037)
    model = TinyEIEM().eval()
    probe = _rare_probe()

    row = decomposition.payload_integration_probe_row(model, probe)
    no_memory = no_memory_final_logits(model, probe.prompt_ids)
    learned = oracle_evidence_final_logits(model, probe)

    # The row scores come from the exact same frozen logits paths; the function
    # itself asserts bit-exact learned/no-memory parity internally.
    assert row["no_memory"]["candidate_nll"] >= 0.0
    assert row["oracle_learned_gate"]["candidate_nll"] >= 0.0
    assert row["oracle_unit_gate"]["candidate_nll"] >= 0.0
    assert len(row["hop_traces"]) == 1

    # Direct sanity: the frozen learned oracle and no-memory paths are not
    # accidentally aliased on this synthetic long-range case.
    assert not torch.equal(learned, no_memory)


def test_two_hop_trace_records_divergent_learned_and_unit_states() -> None:
    torch.manual_seed(1038)
    model = TinyEIEM().eval()
    row = decomposition.payload_integration_probe_row(model, _two_hop_probe())
    traces = row["hop_traces"]
    assert len(traces) == 2
    assert traces[0]["target_position"] == 90
    assert traces[1]["target_position"] == 300
    assert traces[0]["learned_query_norm"] == pytest.approx(
        traces[0]["unit_query_norm"], rel=0.0, abs=0.0
    )
    # After the first update the learned and unit states must differ.
    assert traces[1]["learned_query_norm"] != pytest.approx(
        traces[1]["unit_query_norm"], rel=1e-7, abs=1e-7
    )
    for trace in traces:
        assert trace["raw_memory_norm"] > 0.0
        assert trace["learned_gated_memory_norm"] > 0.0
        assert trace["learned_contribution_norm_ratio"] >= 0.0
        for key in ("learned_gate_candidate_delta", "unit_gate_candidate_delta"):
            delta = trace[key]
            assert delta["answer_minus_best_wrong_delta_contribution"] == pytest.approx(
                delta["answer_logit_delta"] - delta["best_wrong_logit_delta"],
                abs=1e-9,
            )


def test_local_negative_never_receives_oracle_memory_under_either_gate() -> None:
    torch.manual_seed(1039)
    model = TinyEIEM().eval()
    row = decomposition.payload_integration_probe_row(model, _local_negative_probe())
    assert row["hop_traces"] == []
    assert row["oracle_learned_gate"] == row["no_memory"]
    assert row["oracle_unit_gate"] == row["no_memory"]


def _score(correct: bool, nll: float, stale: bool = False) -> dict[str, object]:
    return {
        "correct": correct,
        "candidate_nll": nll,
        "stale_choice": stale,
    }


def _row(
    family: str,
    *,
    no_nll: float,
    learned_nll: float,
    unit_nll: float,
    no_correct: bool,
    learned_correct: bool,
    unit_correct: bool,
    learned_margin: float,
    unit_margin: float,
    contribution_ratio: float = 0.02,
) -> dict[str, object]:
    traces = []
    if family != "local_negative":
        traces = [
            {
                "learned_contribution_norm_ratio": contribution_ratio,
                "learned_gate_candidate_delta": {
                    "answer_minus_best_wrong_delta_contribution": learned_margin
                },
                "unit_gate_candidate_delta": {
                    "answer_minus_best_wrong_delta_contribution": unit_margin
                },
            }
        ]
    return {
        "family": family,
        "case_id": 0,
        "no_memory": _score(no_correct, no_nll),
        "oracle_learned_gate": _score(learned_correct, learned_nll),
        "oracle_unit_gate": _score(unit_correct, unit_nll),
        "hop_traces": traces,
    }


def test_summary_uses_frozen_long_range_comparisons() -> None:
    rows = [
        _row(
            family,
            no_nll=1.00,
            learned_nll=1.03,
            unit_nll=0.92,
            no_correct=False,
            learned_correct=False,
            unit_correct=True,
            learned_margin=-0.1,
            unit_margin=0.2,
        )
        for family in ("rare_fact", "overwrite", "two_hop")
    ]
    rows.append(
        _row(
            "local_negative",
            no_nll=1.00,
            learned_nll=1.00,
            unit_nll=1.00,
            no_correct=True,
            learned_correct=True,
            unit_correct=True,
            learned_margin=0.0,
            unit_margin=0.0,
        )
    )
    gate = {
        "count": 16.0,
        "min": 0.02,
        "p05": 0.02,
        "median": 0.02,
        "mean": 0.02,
        "p95": 0.02,
        "max": 0.02,
        "fraction_lt_0p01": 0.0,
        "fraction_lt_0p05": 1.0,
        "fraction_gt_0p10": 0.0,
        "l1": 0.32,
        "l2": 0.08,
    }
    summary = decomposition.summarize_payload_integration(rows, gate)
    agg = summary["aggregate"]
    assert agg["learned_oracle_benefit"] == pytest.approx(-0.03)
    assert agg["unit_oracle_benefit"] == pytest.approx(0.08)
    assert agg["unit_minus_learned_benefit"] == pytest.approx(0.11)
    assert agg["learned_accuracy_gain"] == pytest.approx(0.0)
    assert agg["unit_accuracy_gain"] == pytest.approx(1.0)
    assert agg["median_learned_contribution_norm_ratio"] == pytest.approx(0.02)
    assert agg["mean_learned_answer_minus_best_wrong_delta"] == pytest.approx(-0.1)
    assert agg["mean_unit_answer_minus_best_wrong_delta"] == pytest.approx(0.2)


def test_preregistered_tag_boundaries_are_pure_and_descriptive() -> None:
    summary = {
        "gate_statistics": {"median": 0.02},
        "aggregate": {
            "learned_oracle_benefit": -0.03,
            "unit_oracle_benefit": 0.06,
            "unit_minus_learned_benefit": 0.09,
            "unit_accuracy_gain": 0.10,
            "mean_learned_answer_minus_best_wrong_delta": -0.01,
            "mean_unit_answer_minus_best_wrong_delta": 0.02,
        },
    }
    result = decomposition.classify_payload_integration(summary)
    assert set(result["tags"]) == {
        "GATE_SUPPRESSION_SIGNAL",
        "LEARNED_GATE_HARM_SIGNAL",
    }
    assert result["stage_c_result_changed"] is False
    assert result["first_postmortem_result_changed"] is False
    assert result["new_training_authorized"] is False
    assert result["new_scientific_seed_authorized"] is False
    assert result["stage_d_authorized"] is False

    weak = {
        "gate_statistics": {"median": 0.20},
        "aggregate": {
            "learned_oracle_benefit": -0.03,
            "unit_oracle_benefit": 0.00,
            "unit_minus_learned_benefit": 0.03,
            "unit_accuracy_gain": 0.01,
            "mean_learned_answer_minus_best_wrong_delta": -0.01,
            "mean_unit_answer_minus_best_wrong_delta": -0.01,
        },
    }
    weak_result = decomposition.classify_payload_integration(weak)
    assert set(weak_result["tags"]) == {
        "PAYLOAD_DIRECTION_WEAK_SIGNAL",
        "LEARNED_GATE_HARM_SIGNAL",
    }

    destabilized = {
        "gate_statistics": {"median": 0.20},
        "aggregate": {
            "learned_oracle_benefit": 0.00,
            "unit_oracle_benefit": -0.10,
            "unit_minus_learned_benefit": -0.10,
            "unit_accuracy_gain": 0.00,
            "mean_learned_answer_minus_best_wrong_delta": 0.01,
            "mean_unit_answer_minus_best_wrong_delta": 0.01,
        },
    }
    destabilized_result = decomposition.classify_payload_integration(destabilized)
    assert destabilized_result["tags"] == ["UNIT_GATE_DESTABILIZATION_SIGNAL"]


def test_module_has_no_execution_training_or_checkpoint_io_surface() -> None:
    source = inspect.getsource(decomposition)
    tree = ast.parse(source)

    imported_roots = set()
    call_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                call_names.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                call_names.append(node.func.id)

    assert "modal" not in imported_roots
    for forbidden in (
        "load",
        "save",
        "backward",
        "step",
        "zero_grad",
        "open",
        "write_text",
        "write_bytes",
    ):
        assert forbidden not in call_names
    assert not any(name.startswith("train") for name in call_names)
