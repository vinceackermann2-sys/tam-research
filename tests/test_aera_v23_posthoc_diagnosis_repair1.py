from __future__ import annotations

import torch

from tam_research import aera_v23_posthoc_diagnosis as base
from tam_research import aera_v23_posthoc_diagnosis_repair1 as repair
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v23 import HardwareAwareAERATextLMV23


def _tiny_model() -> HardwareAwareAERATextLMV23:
    cfg = HardwareAERAConfig(
        vocab_size=97,
        d_model=24,
        n_stages=4,
        n_heads=4,
        chunk_size=256,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=2,
        block_size=2,
    )
    return HardwareAwareAERATextLMV23(cfg)


def test_repair1_exports_exact_undecorated_orchestration_body():
    assert repair.REPAIR_ISSUE == 344
    assert repair.SOURCE_FAILED_TRIGGER == 343
    assert repair.SEMANTIC_CHANGE == "construct_and_load_model_outside_inference_mode_only"
    assert hasattr(base.run_posthoc_diagnosis, "__wrapped__")
    assert repair.run_posthoc_diagnosis is base.run_posthoc_diagnosis.__wrapped__
    assert not hasattr(repair.run_posthoc_diagnosis, "__wrapped__")


def test_version_tracked_parameters_survive_inference_only_forward_unchanged():
    torch.manual_seed(9344)
    model = _tiny_model().eval()
    before = repair.parameter_versions(model)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 16))
    with torch.inference_mode():
        out = model(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=False,
        )
        assert isinstance(out["logits"], torch.Tensor)
    after = repair.parameter_versions(model)
    assert before == after
