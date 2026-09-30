from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tam_research.chm_v2_100m_qva as qva
from tam_research.chm_v1_100m_stage_c_eval import eiem_flat_final_logits
from tam_research.chm_v1_small_lm import LOCAL_WINDOW, _hidden
from tam_research.models import ModelConfig, ResearchLM


class TinyQVA(nn.Module):
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
        self.qva = qva.QueryConditionedValueAdapter(16, 4)

    def query_for(self, representation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(representation), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.qva(hidden, memory)


def test_stage_a_manifest_has_no_gpu_or_scientific_authority() -> None:
    manifest = qva.stage_a_preflight()
    assert manifest["classification"] == "CHM_V2_100M_QVA_STAGE_A_IMPLEMENTATION_PREFLIGHT_PASS"
    assert manifest["issue"] == 1147
    assert manifest["qva_rank"] == 32
    assert manifest["qva_gate_init"] == -4.0
    assert manifest["legacy_memory_gate_retained_but_inert"] is True
    assert manifest["output_projection_zero_initialized"] is True
    assert manifest["gpu_authorized"] is False
    assert manifest["modal_authorized"] is False
    assert manifest["paid_compute_authorized"] is False
    assert manifest["scientific_training_authorized"] is False
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["stage_b_authorized"] is False
    assert manifest["stage_c_authorized"] is False


def test_parameter_accounting_matches_preregistration_and_point_one_percent_gate() -> None:
    accounting = qva.analytical_parameter_accounting()
    assert qva.qva_added_parameter_count() == 49_185
    assert accounting["local_trainable_parameters"] == 101_803_520
    assert accounting["chm_v1_eiem_trainable_parameters"] == 101_836_800
    assert accounting["qva_added_parameters"] == 49_185
    assert accounting["qva_trainable_parameters"] == 101_885_985
    assert accounting["qva_minus_local_parameters"] == 82_465
    assert accounting["delta_fraction_vs_local"] == pytest.approx(82_465 / 101_803_520)
    assert accounting["delta_fraction_vs_local"] < 0.001
    assert accounting["within_point_one_percent"] is True


def test_instantiated_parameter_count_matches_analytical_contract() -> None:
    assert qva.instantiated_parameter_accounting() == qva.analytical_parameter_accounting()


def test_qva_zero_output_projection_is_exact_noop_at_initialization() -> None:
    torch.manual_seed(1147001)
    adapter = qva.QueryConditionedValueAdapter(16, 4)
    hidden = torch.randn(2, 3, 16)
    memory = torch.randn(2, 3, 16)
    out = adapter(hidden, memory)
    assert torch.equal(out, hidden)
    assert torch.count_nonzero(adapter.output_projection.weight).item() == 0
    assert float(adapter.gate_bias.item()) == -4.0


def test_qva_gradient_flows_through_query_memory_and_gate_after_adapter_activates() -> None:
    torch.manual_seed(1147002)
    adapter = qva.QueryConditionedValueAdapter(16, 4)
    with torch.no_grad():
        adapter.output_projection.weight.normal_(mean=0.0, std=0.05)

    hidden = torch.randn(2, 3, 16, requires_grad=True)
    memory = torch.randn(2, 3, 16, requires_grad=True)
    target = torch.randn(2, 3, 16)
    loss = F.mse_loss(adapter(hidden, memory), target)
    loss.backward()

    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0.0
    assert memory.grad is not None and float(memory.grad.abs().sum()) > 0.0
    for parameter in (
        adapter.query_projection.weight,
        adapter.value_projection.weight,
        adapter.output_projection.weight,
        adapter.gate_projection.weight,
        adapter.gate_bias,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0


def test_qva_rank_is_frozen() -> None:
    with pytest.raises(RuntimeError, match="rank is frozen"):
        qva.CHMV2100MEIEMQVA(rank=16)


def test_engineering_seed_guard_refuses_scientific_and_arbitrary_seeds() -> None:
    assert qva.validate_engineering_seed(1_147_099) == 1_147_099
    with pytest.raises(RuntimeError):
        qva.validate_engineering_seed(977_001)
    with pytest.raises(RuntimeError):
        qva.validate_engineering_seed(123)


def test_soft_training_helper_freezes_temperature_and_preserves_future_causality() -> None:
    torch.manual_seed(1147003)
    model = TinyQVA().eval()

    tokens_a = torch.randint(0, 257, (1, 1024), dtype=torch.long)
    tokens_b = tokens_a.clone()
    tokens_b[:, 512:] = torch.randint(0, 257, (1, 512), dtype=torch.long)

    logits_a = qva.qva_flat_training_session_logits(model, tokens_a)
    logits_b = qva.qva_flat_training_session_logits(model, tokens_b)
    assert logits_a.shape == (1, 1024, 257)
    assert torch.equal(logits_a[:, :512], logits_b[:, :512])

    with pytest.raises(RuntimeError, match="temperature is frozen"):
        qva.qva_flat_training_session_logits(model, tokens_a, temperature=0.2)


def test_soft_training_first_chunk_is_exact_local_backbone_logits_at_zero_init() -> None:
    torch.manual_seed(1147004)
    model = TinyQVA().eval()
    tokens = torch.randint(0, 257, (1, 1024), dtype=torch.long)
    logits = qva.qva_flat_training_session_logits(model, tokens)

    first_hidden = _hidden(model.backbone, tokens[:, :512])
    first_local = model.backbone.lm_head(first_hidden)
    torch.testing.assert_close(logits[:, :512], first_local, rtol=0.0, atol=0.0)


def test_hard_flat_wrapper_is_bit_exact_to_frozen_stage_c_evaluator() -> None:
    torch.manual_seed(1147005)
    model = TinyQVA().eval()
    prompt = tuple((index * 17 + 3) % 257 for index in range(700))
    wrapped = qva.qva_hard_flat_final_logits(model, prompt)
    frozen = eiem_flat_final_logits(model, prompt)
    torch.testing.assert_close(wrapped, frozen, rtol=0.0, atol=0.0)


def test_qva_integration_does_not_reference_inherited_legacy_gate() -> None:
    source = inspect.getsource(qva.CHMV2100MEIEMQVA._integrate)
    assert "memory_gate_logit" not in source
    assert "self.qva" in source


def test_module_contains_no_training_loop_optimizer_modal_or_paid_compute_surface() -> None:
    source = inspect.getsource(qva)
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
    assert "step" not in calls
    assert "zero_grad" not in calls
    assert "save" not in calls
    assert "load" not in calls
    assert "AdamW" not in source
    assert "workflow_dispatch" not in source
