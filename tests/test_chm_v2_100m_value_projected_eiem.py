from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tam_research.chm_v2_100m_value_projected_eiem as v2
from tam_research.chm_v1_100m_stage_c_eval import eiem_flat_final_logits
from tam_research.chm_v1_100m_stage_c_execution import eiem_exact_flat_two_chunk_logits
from tam_research.chm_v1_small_lm import LOCAL_WINDOW
from tam_research.chm_v1_small_lm_protocol import eiem_flat_training_session_logits
from tam_research.models import ModelConfig, ResearchLM


class TinyIdentityProjectedEIEM(nn.Module):
    """Tiny duck-typed EIEM whose value map starts as exact identity."""

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
        self.value_down = nn.Linear(16, 4, bias=True)
        self.value_up = nn.Linear(4, 16, bias=True)
        nn.init.zeros_(self.value_up.weight)
        nn.init.zeros_(self.value_up.bias)

    def query_for(self, representation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(representation), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)

    def value_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.value_up(self.value_down(hidden))

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.memory_gate_logit).to(hidden.dtype)
        return hidden + gate * memory


def test_manifest_is_no_execution_authority_and_seed_is_reserved_only() -> None:
    manifest = v2.protocol_manifest()
    assert manifest["classification"].endswith("NO_EXECUTION_AUTHORITY")
    assert manifest["issue"] == 1112
    assert manifest["reserved_development_seed"] == 2_011_121
    assert manifest["reserved_seed_authorized"] is False
    assert manifest["consumed_chm_v1_seed"] == 977_001
    assert manifest["value_rank"] == 32
    assert manifest["soft_training_temperature"] == 0.10
    assert manifest["training_tokens_per_model"] == 33_554_432
    assert manifest["tokens_per_optimizer_step"] == 16_384
    assert manifest["optimizer_steps_per_model"] == 2_048
    assert manifest["warmup_steps"] == 40
    assert manifest["gpu_authorized"] is False
    assert manifest["training_execution_authorized"] is False
    assert manifest["scientific_execution_authorized"] is False
    assert manifest["stage_d_authorized"] is False

    with pytest.raises(RuntimeError, match="reserved development seed"):
        v2.validate_engineering_seed(2_011_121)
    with pytest.raises(RuntimeError, match="consumed CHM-v1 seed"):
        v2.validate_engineering_seed(977_001)
    assert v2.validate_engineering_seed(1_112_099) == 1_112_099


def test_exact_parameter_accounting_stays_inside_point_one_percent() -> None:
    expected = {
        "local_trainable_parameters": 101_803_520,
        "raw_eiem_trainable_parameters": 101_836_800,
        "value_projection_parameters": 33_312,
        "vp_eiem_trainable_parameters": 101_870_112,
        "vp_extra_vs_local": 66_592,
    }
    analytical = v2.analytical_parameter_accounting()
    for key, value in expected.items():
        assert analytical[key] == value
    assert analytical["within_point_one_percent"] is True
    assert analytical["vp_delta_fraction_vs_local"] == pytest.approx(
        66_592 / 101_803_520,
        rel=0,
        abs=1e-15,
    )

    contract = v2.validate_architecture_contract()
    instantiated = contract["parameter_accounting"]
    for key, value in expected.items():
        assert instantiated[key] == value
    assert instantiated["within_point_one_percent"] is True


def test_actual_vp_projection_is_identity_initialized_and_trainable() -> None:
    torch.manual_seed(1112)
    model = v2.CHMV2100MValueProjectedEIEMLM()
    assert model.value_rank == 32
    assert torch.count_nonzero(model.value_up.weight).item() == 0
    assert torch.count_nonzero(model.value_up.bias).item() == 0

    hidden = torch.randn(2, 3, 512, requires_grad=True)
    projected = model.value_for(hidden)
    torch.testing.assert_close(projected, hidden, rtol=0.0, atol=0.0)
    ratios = v2.projection_correction_ratio(model, hidden)
    torch.testing.assert_close(ratios, torch.zeros_like(ratios), rtol=0.0, atol=0.0)

    loss = projected.square().mean()
    loss.backward()
    assert model.value_up.weight.grad is not None
    assert torch.isfinite(model.value_up.weight.grad).all()
    assert float(model.value_up.weight.grad.abs().sum()) > 0.0


def test_identity_projected_soft_training_is_exactly_raw_at_initialization() -> None:
    torch.manual_seed(11121)
    model = TinyIdentityProjectedEIEM().eval()
    tokens = torch.randint(0, 257, (1, 2 * LOCAL_WINDOW), dtype=torch.long)

    raw = eiem_flat_training_session_logits(model, tokens)
    projected = v2.vp_eiem_flat_training_session_logits(model, tokens)
    torch.testing.assert_close(projected, raw, rtol=0.0, atol=0.0)

    with pytest.raises(RuntimeError, match="temperature is frozen"):
        v2.vp_eiem_flat_training_session_logits(model, tokens, temperature=0.2)


def test_identity_projected_exact_two_chunk_eval_is_raw_at_initialization() -> None:
    torch.manual_seed(11122)
    model = TinyIdentityProjectedEIEM().eval()
    tokens = torch.randint(0, 257, (1, 2 * LOCAL_WINDOW), dtype=torch.long)
    raw = eiem_exact_flat_two_chunk_logits(model, tokens)
    projected = v2.vp_eiem_exact_flat_two_chunk_logits(model, tokens)
    torch.testing.assert_close(projected, raw, rtol=0.0, atol=0.0)


def test_identity_projected_final_token_eval_is_raw_at_initialization() -> None:
    torch.manual_seed(11123)
    model = TinyIdentityProjectedEIEM().eval()
    prompt = tuple(int(x) for x in torch.randint(0, 257, (700,), dtype=torch.long))
    raw = eiem_flat_final_logits(model, prompt)
    projected = v2.vp_eiem_flat_final_logits(model, prompt)
    torch.testing.assert_close(projected, raw, rtol=0.0, atol=0.0)


def _positive_gate_inputs():
    integrity = {
        "parameter_fairness": True,
        "paired_initialization": True,
        "byte_identical_training_stream": True,
        "exact_training_tokens": True,
        "matched_schedule": True,
        "finite_losses": True,
        "finite_parameters": True,
        "no_future_leakage": True,
        "no_cross_session_aliasing": True,
    }
    aggregate = {
        "vp_vs_local_accuracy_gain": 0.06,
        "vp_vs_local_nll_benefit": 0.06,
        "vp_vs_raw_accuracy_gain": 0.01,
        "vp_vs_raw_nll_benefit": 0.06,
    }
    families = {
        "rare_fact": {
            "vp_vs_local_accuracy_gain": 0.04,
            "vp_vs_local_nll_benefit": 0.03,
            "vp_vs_raw_nll_benefit": 0.01,
        },
        "overwrite": {
            "vp_vs_local_accuracy_gain": 0.04,
            "vp_vs_local_nll_benefit": 0.03,
            "vp_vs_raw_nll_benefit": 0.01,
        },
        "two_hop": {
            "vp_vs_local_accuracy_gain": 0.00,
            "vp_vs_local_nll_benefit": -0.01,
            "vp_vs_raw_nll_benefit": -0.01,
        },
    }
    return integrity, aggregate, families


def test_development_gate_positive_path_never_self_authorizes_successors() -> None:
    integrity, aggregate, families = _positive_gate_inputs()
    decision = v2.classify_development_gate(
        integrity=integrity,
        aggregate=aggregate,
        families=families,
        ordinary_vp_minus_local_nll=0.00,
        local_negative_accuracy_gain=0.00,
        local_negative_nll_regression=0.00,
        local_overwrite_stale_rate=0.20,
        vp_overwrite_stale_rate=0.15,
    )
    assert decision["classification"] == v2.POSITIVE_CLASSIFICATION
    assert decision["passed"] is True
    assert decision["stop_reasons"] == []
    assert decision["stage_d_authorized"] is False
    assert decision["multi_seed_replication_authorized"] is False
    assert decision["scale_up_authorized"] is False


def test_development_gate_stops_on_projection_ablation_or_preservation_failure() -> None:
    integrity, aggregate, families = _positive_gate_inputs()
    aggregate = dict(aggregate)
    aggregate["vp_vs_raw_nll_benefit"] = 0.049
    decision = v2.classify_development_gate(
        integrity=integrity,
        aggregate=aggregate,
        families=families,
        ordinary_vp_minus_local_nll=0.031,
        local_negative_accuracy_gain=-0.021,
        local_negative_nll_regression=0.031,
        local_overwrite_stale_rate=0.20,
        vp_overwrite_stale_rate=0.17,
    )
    assert decision["classification"] == v2.STOP_CLASSIFICATION
    assert decision["passed"] is False
    assert "vp_vs_raw_nll_benefit" in decision["stop_reasons"]
    assert "ordinary_language_nll" in decision["stop_reasons"]
    assert "local_negative_accuracy" in decision["stop_reasons"]
    assert "local_negative_nll" in decision["stop_reasons"]
    assert "overwrite_stale_guard" in decision["stop_reasons"]


def test_module_contains_no_execution_or_training_loop_surface() -> None:
    source = inspect.getsource(v2)
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
    assert "subprocess" not in imported_roots
    assert "torch.optim" not in source
    assert "torch.load(" not in source
    assert "torch.save(" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "TokenBin" not in source
    assert "paid_run_authorized" not in source
    assert "workflow_dispatch" not in source


def test_projected_values_are_written_only_after_current_chunk_logits() -> None:
    source = inspect.getsource(v2.vp_eiem_flat_training_session_logits)
    logits_index = source.index("logits.append")
    write_index = source.index("memory_values =")
    assert logits_index < write_index
    assert "values = model.value_for(hidden)" in source
    assert "keys = model.key_for(hidden)" in source
