from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tam_research.chm_v1_100m_stage_c_postmortem as postmortem
from tam_research.chm_v1_100m_stage_c_eval import GENERATOR_VERSION, eiem_flat_final_logits
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
        self.memory_gate_logit = nn.Parameter(torch.full((16,), -2.0))

    def query_for(self, representation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(representation), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return hidden + torch.sigmoid(self.memory_gate_logit).to(hidden.dtype) * memory


def _rare_probe() -> EncodedProbe:
    prompt = tuple((index * 17 + 3) % 257 for index in range(700))
    return EncodedProbe(
        family="rare_fact",
        case_id=1,
        prompt_text="synthetic test only",
        prompt_ids=prompt,
        answer_token_id=8,
        candidate_token_ids=tuple(range(8, 16)),
        evidence_end_token=100,
        query_token=len(prompt) - 1,
        evidence_distance=(len(prompt) - 1) - 100,
        generator_version=GENERATOR_VERSION,
    )


def _local_negative_probe() -> EncodedProbe:
    prompt = tuple((index * 11 + 5) % 257 for index in range(100))
    return EncodedProbe(
        family="local_negative",
        case_id=300_001,
        prompt_text="synthetic local control",
        prompt_ids=prompt,
        answer_token_id=8,
        candidate_token_ids=tuple(range(8, 16)),
        evidence_end_token=80,
        query_token=len(prompt) - 1,
        evidence_distance=(len(prompt) - 1) - 80,
        generator_version=GENERATOR_VERSION,
    )


def _checkpoint(kind: str = "eiem") -> dict[str, object]:
    return {
        "kind": kind,
        "step": 2048,
        "tokens_seen": 33_554_432,
        "mean_train_nll": 1.0,
        "lr": 0.0,
        "resume_authorized": False,
        "scientific_evaluation_authorized": True,
        "model_state_dict": {"x": torch.tensor([1.0])},
    }


def test_manifest_is_checkpoint_only_and_grants_no_execution_authority() -> None:
    manifest = postmortem.validate_protocol_manifest()
    assert manifest == postmortem.protocol_manifest()
    assert manifest["classification"].endswith("NO_EXECUTION_AUTHORITY")
    assert manifest["soft_temperature"] == 0.10
    assert manifest["final_step"] == 2048
    assert manifest["final_tokens"] == 33_554_432
    assert manifest["checkpoint_only"] is True
    assert manifest["parameter_updates_authorized"] is False
    assert manifest["training_authorized"] is False
    assert manifest["optimizer_authorized"] is False
    assert manifest["gpu_authorized"] is False
    assert manifest["modal_authorized"] is False
    assert manifest["new_scientific_seed_authorized"] is False
    assert manifest["stage_d_authorized"] is False


def test_checkpoint_guard_accepts_only_final_nonresumable_stage_c_artifact() -> None:
    payload = _checkpoint()
    assert postmortem.validate_checkpoint_payload(payload, expected_kind="eiem") is payload

    for field, bad in (
        ("kind", "local"),
        ("step", 1536),
        ("tokens_seen", 33_554_431),
        ("resume_authorized", True),
        ("scientific_evaluation_authorized", False),
    ):
        candidate = dict(payload)
        candidate[field] = bad
        with pytest.raises(RuntimeError):
            postmortem.validate_checkpoint_payload(candidate, expected_kind="eiem")

    empty_state = dict(payload)
    empty_state["model_state_dict"] = {}
    with pytest.raises(RuntimeError, match="model_state_dict"):
        postmortem.validate_checkpoint_payload(empty_state, expected_kind="eiem")


def test_hard_trace_path_is_bit_exact_to_frozen_stage_c_evaluator() -> None:
    torch.manual_seed(1002)
    model = TinyEIEM().eval()
    probe = _rare_probe()
    traced, traces = postmortem.hard_exact_final_logits_with_trace(model, probe)
    frozen = eiem_flat_final_logits(model, probe.prompt_ids)
    torch.testing.assert_close(traced, frozen, rtol=0.0, atol=0.0)
    assert len(traces) == 2
    assert all(trace["target_position"] == probe.evidence_end_token for trace in traces)
    assert all(trace["target_rank"] is not None and trace["target_rank"] >= 1 for trace in traces)
    assert all(isinstance(trace["selected_equals_target"], bool) for trace in traces)


def test_soft_condition_is_frozen_temperature_and_reports_endpoint_mass_entropy() -> None:
    torch.manual_seed(1003)
    model = TinyEIEM().eval()
    probe = _rare_probe()
    logits, traces = postmortem.soft_train_final_logits_with_trace(model, probe)
    assert logits.shape == (257,)
    assert len(traces) == 2
    for trace in traces:
        assert trace["target_position"] == probe.evidence_end_token
        assert trace["target_mass"] is not None
        assert 0.0 <= trace["target_mass"] <= 1.0
        assert trace["entropy"] >= 0.0
        assert 0 <= trace["top1_position"] < 512


def test_local_negative_oracle_never_injects_current_chunk_evidence() -> None:
    torch.manual_seed(1004)
    model = TinyEIEM().eval()
    probe = _local_negative_probe()
    no_memory = postmortem.no_memory_final_logits(model, probe.prompt_ids)
    oracle = postmortem.oracle_evidence_final_logits(model, probe)
    torch.testing.assert_close(oracle, no_memory, rtol=0.0, atol=0.0)

    row = postmortem.diagnostic_probe_row(model, probe)
    assert row["hard_trace"] == []
    assert row["soft_trace"] == []
    assert row["oracle_evidence_endpoint"] == row["no_memory"]


def _condition(*, accuracy: float, nll: float, stale: float) -> dict[str, float]:
    return {"accuracy": accuracy, "candidate_nll": nll, "stale_choice_rate": stale}


def _summary_for_tags() -> dict[str, object]:
    per_family = {}
    for family in ("rare_fact", "overwrite", "two_hop"):
        per_family[family] = {
            "conditions": {
                "no_memory": _condition(accuracy=0.10, nll=1.00, stale=0.20 if family == "overwrite" else 0.0),
                "hard_exact": _condition(accuracy=0.10, nll=1.00, stale=0.30 if family == "overwrite" else 0.0),
                "soft_train_t0p10": _condition(accuracy=0.20, nll=0.90, stale=0.20 if family == "overwrite" else 0.0),
                "oracle_evidence_endpoint": _condition(
                    accuracy=0.30,
                    nll=0.80,
                    stale=0.10 if family == "overwrite" else 0.0,
                ),
            }
        }
    return {
        "aggregate": {
            "soft_minus_hard_nll_benefit": 0.10,
            "oracle_minus_hard_nll_benefit": 0.20,
            "oracle_minus_no_memory_nll_benefit": 0.20,
            "hard_authoritative_endpoint_hit_rate": 0.20,
        },
        "per_family": per_family,
    }


def test_diagnostic_tags_follow_preregistered_descriptive_thresholds() -> None:
    result = postmortem.classify_diagnostic_signals(_summary_for_tags())
    assert result["classification"] == "CHM_V1_100M_STAGE_C_POSTMORTEM_DIAGNOSTIC_ONLY"
    assert set(result["tags"]) == {
        "SOFT_HARD_MISMATCH_SIGNAL",
        "ADDRESS_SELECTION_SIGNAL",
        "OVERWRITE_RECENCY_SIGNAL",
    }
    assert result["stage_c_result_changed"] is False
    assert result["stage_d_authorized"] is False
    assert result["new_training_authorized"] is False

    weak = _summary_for_tags()
    weak["aggregate"]["soft_minus_hard_nll_benefit"] = 0.0
    weak["aggregate"]["oracle_minus_hard_nll_benefit"] = 0.0
    weak["aggregate"]["oracle_minus_no_memory_nll_benefit"] = 0.01
    weak["per_family"]["overwrite"]["conditions"]["oracle_evidence_endpoint"]["stale_choice_rate"] = 0.30
    weak["per_family"]["overwrite"]["conditions"]["oracle_evidence_endpoint"]["candidate_nll"] = 1.00
    weak_result = postmortem.classify_diagnostic_signals(weak)
    assert weak_result["tags"] == ["PAYLOAD_INTEGRATION_WEAK_SIGNAL"]


def test_postmortem_module_has_no_launcher_or_parameter_update_surface() -> None:
    source = inspect.getsource(postmortem)
    tree = ast.parse(source)

    imported_roots = set()
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)

    assert "modal" not in imported_roots
    assert "backward" not in calls
    assert "save" not in calls
    assert "load" not in calls
    assert "step" not in calls
    assert "zero_grad" not in calls
    assert not any(name.startswith("train") for name in calls)
