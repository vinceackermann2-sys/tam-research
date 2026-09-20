from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tam_research.chm_v1_100m_stage_c_payload_gate_diagnostic as diagnostic
from tam_research.chm_v1_long_memory_eval import EncodedProbe
from tam_research.models import ModelConfig, ResearchLM
from tam_research.chm_v1_small_lm import LOCAL_WINDOW


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


def _probe(*, duplicate_answer: bool = False) -> EncodedProbe:
    prompt = [20] * 700
    prompt[100] = 8
    if duplicate_answer:
        prompt[200] = 8
    return EncodedProbe(
        family="rare_fact",
        case_id=7,
        prompt_text="synthetic payload/gate test",
        prompt_ids=tuple(prompt),
        answer_token_id=8,
        candidate_token_ids=tuple(range(8, 16)),
        evidence_end_token=101,
        query_token=len(prompt) - 1,
        evidence_distance=(len(prompt) - 1) - 101,
        generator_version="chm-v1-100m-heldout-aligned-v4",
    )


def test_manifest_is_checkpoint_only_and_has_no_execution_authority() -> None:
    manifest = diagnostic.validate_protocol_manifest()
    assert manifest == diagnostic.protocol_manifest()
    assert manifest["classification"].endswith("NO_EXECUTION_AUTHORITY")
    assert manifest["source_postmortem_issue"] == 1008
    assert manifest["source_result_comment"] == 5750912351
    assert manifest["source_checkpoint_bytes"] == 407_424_818
    assert len(manifest["source_checkpoint_sha256"]) == 64
    assert manifest["checkpoint_only"] is True
    assert manifest["production_checkpoint_loading_authorized"] is False
    assert manifest["parameter_updates_authorized"] is False
    assert manifest["training_authorized"] is False
    assert manifest["optimizer_authorized"] is False
    assert manifest["backward_authorized"] is False
    assert manifest["gpu_authorized"] is False
    assert manifest["modal_authorized"] is False
    assert manifest["new_scientific_seed_authorized"] is False
    assert manifest["stage_d_authorized"] is False


def test_gate_state_summary_reports_frozen_initial_gate_geometry() -> None:
    model = TinyEIEM()
    summary = diagnostic.gate_state_summary(model)
    expected = torch.sigmoid(torch.tensor(-4.0)).item()
    assert summary["dimensions"] == 16
    assert summary["logit"]["median"] == pytest.approx(-4.0)
    assert summary["sigmoid"]["median"] == pytest.approx(expected)
    assert summary["sigmoid_fraction_ge_0p05"] == 0.0
    assert summary["sigmoid_fraction_ge_0p10"] == 0.0
    assert summary["sigmoid_fraction_ge_0p25"] == 0.0
    assert summary["mean_abs_logit_delta_from_init"] == 0.0
    assert summary["max_abs_logit_delta_from_init"] == 0.0


def test_unique_answer_token_position_is_fail_closed() -> None:
    assert diagnostic.unique_answer_token_position(_probe()) == 100
    with pytest.raises(RuntimeError, match="exactly one answer-token occurrence"):
        diagnostic.unique_answer_token_position(_probe(duplicate_answer=True))

    local = _probe()
    local = EncodedProbe(
        family="local_negative",
        case_id=local.case_id,
        prompt_text=local.prompt_text,
        prompt_ids=local.prompt_ids,
        answer_token_id=local.answer_token_id,
        candidate_token_ids=local.candidate_token_ids,
        evidence_end_token=local.evidence_end_token,
        query_token=local.query_token,
        evidence_distance=local.evidence_distance,
        generator_version=local.generator_version,
    )
    with pytest.raises(RuntimeError, match="long-range only"):
        diagnostic.unique_answer_token_position(local)


def test_payload_gate_probe_row_scores_all_frozen_conditions_without_mutation() -> None:
    torch.manual_seed(1014)
    model = TinyEIEM().eval()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    row = diagnostic.payload_gate_probe_row(model, _probe())

    assert row["answer_token_position"] == 100
    assert row["answer_position_minus_evidence_end"] == -1
    assert set(row["conditions"]) == {
        "no_memory",
        "answer_token_learned_gate",
        "answer_token_full_residual",
        "label_direction_learned_gate",
        "label_direction_full_residual",
    }
    for value in row["conditions"].values():
        assert 0.0 <= float(value["candidate_nll"])
        assert isinstance(value["correct"], bool)
        assert value["predicted_token_id"] in range(8, 16)
        assert value["answer_margin"] == pytest.approx(
            value["answer_logit"] - value["best_wrong_logit"]
        )

    payload = row["payload"]
    assert payload["raw_answer_hidden_norm"] > 0.0
    assert payload["query_state_norm"] > 0.0
    assert -1.0 <= payload["raw_answer_hidden_to_label_direction_cosine"] <= 1.0
    assert payload["full_strength_residual_norm"] == pytest.approx(
        payload["raw_answer_hidden_norm"], rel=1e-5, abs=1e-6
    )
    assert payload["learned_gate_residual_norm"] < payload["full_strength_residual_norm"]

    after = model.state_dict()
    for name, tensor in before.items():
        torch.testing.assert_close(after[name], tensor, rtol=0.0, atol=0.0)


def _condition(nll: float) -> dict[str, float]:
    return {"accuracy": 0.125, "candidate_nll": nll, "answer_margin": -1.0}


def _synthetic_summary(
    *,
    answer_full_nll: float,
    label_learned_nll: float,
    label_full_nll: float,
    gate_median: float,
) -> dict[str, object]:
    return {
        "aggregate": {
            "conditions": {
                "no_memory": _condition(1.0),
                "answer_token_learned_gate": _condition(0.99),
                "answer_token_full_residual": _condition(answer_full_nll),
                "label_direction_learned_gate": _condition(label_learned_nll),
                "label_direction_full_residual": _condition(label_full_nll),
            }
        },
        "gate_state": {"sigmoid": {"median": gate_median}},
    }


def test_preregistered_tags_localize_gate_and_raw_payload_failures() -> None:
    summary = _synthetic_summary(
        answer_full_nll=0.97,
        label_learned_nll=0.98,
        label_full_nll=0.80,
        gate_median=0.02,
    )
    result = diagnostic.classify_payload_gate_signals(summary)
    assert set(result["tags"]) == {
        "GATE_SUPPRESSION_SIGNAL",
        "RAW_PAYLOAD_FORMAT_SIGNAL",
    }
    assert result["nll_benefits"]["label_direction_full_residual"] == pytest.approx(0.20)
    assert result["stage_c_result_changed"] is False
    assert result["stage_d_authorized"] is False
    assert result["new_training_authorized"] is False
    assert result["new_scientific_seed_authorized"] is False


def test_preregistered_tags_can_report_raw_payload_usable() -> None:
    summary = _synthetic_summary(
        answer_full_nll=0.90,
        label_learned_nll=0.90,
        label_full_nll=0.80,
        gate_median=0.10,
    )
    result = diagnostic.classify_payload_gate_signals(summary)
    assert result["tags"] == ["RAW_PAYLOAD_USABLE_SIGNAL"]


def test_preregistered_tags_can_report_residual_channel_weak() -> None:
    summary = _synthetic_summary(
        answer_full_nll=0.98,
        label_learned_nll=0.99,
        label_full_nll=0.95,
        gate_median=0.02,
    )
    result = diagnostic.classify_payload_gate_signals(summary)
    assert result["tags"] == ["RESIDUAL_CHANNEL_WEAK_SIGNAL"]


def test_module_has_no_checkpoint_loader_launcher_or_training_surface() -> None:
    source = inspect.getsource(diagnostic)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    called: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.append(node.func.id)

    assert "modal" not in imported_roots
    assert "load" not in called
    assert "save" not in called
    assert "backward" not in called
    assert "step" not in called
    assert "zero_grad" not in called
    assert not any(name.startswith("train") for name in called)
    assert "torch.optim" not in source
    assert "torch.load(" not in source
    assert "torch.save(" not in source
