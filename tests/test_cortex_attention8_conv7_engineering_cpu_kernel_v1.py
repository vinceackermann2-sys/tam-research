"""CPU-only contract tests on real meta model shapes and tiny toy tensors."""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tam_research.cortex_attention8_conv7_engineering_cpu_kernel_v1 import (
    MODEL_ORDER, EXPECTED_PARAMETERS, SYNTHETIC_CLASS,
    builder_for, cpu_synthetic_optimizer_step, verify_meta_model_geometry,
)
from tam_research.train import cosine_lr


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(19, 8)
        self.head = nn.Linear(8, 19, bias=False)
    def forward(self, x):
        return self.head(self.emb(x))


def two_batches():
    x0 = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]])
    x1 = torch.tensor([[2, 3, 4, 5], [4, 5, 6, 7]])
    return ((x0, (x0 + 1) % 19), (x1, (x1 + 1) % 19))


def test_exact_three_real_builder_geometry_meta_only_and_paid_blocked():
    out = verify_meta_model_geometry()
    assert out["classification"] == "CONV7_META_MODEL_CONTRACT_PASS_PAID_EXECUTION_BLOCKED"
    assert out["comparators"] == list(MODEL_ORDER)
    assert out["parameter_counts"] == list(EXPECTED_PARAMETERS)
    assert out["optimizer_steps"] == 3052
    assert out["tokens_per_optimizer_step"] == 65536
    assert out["full_horizon_steps"] == 30518
    for name in MODEL_ORDER:
        assert callable(builder_for(name))
    with pytest.raises(ValueError):
        builder_for("unknown")
    for key in ("engineering_seed_assigned", "strict_spend_cap_assigned",
                "paid_execution_authorized", "gpu_allocated",
                "scientific_evidence", "model_weights_allocated"):
        assert out[key] is False


@pytest.mark.parametrize("step", (0, 609, 610, 3051))
def test_exact_full_horizon_prefix_and_two_microbatch_gradient(step):
    torch.manual_seed(45273)
    model = ToyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    initial = model.emb.weight.detach().clone()
    out = cpu_synthetic_optimizer_step(model, optimizer, two_batches(), step)
    assert out["classification"] == SYNTHETIC_CLASS
    assert out["lr_full_horizon_prefix"] == cosine_lr(step, 30518, 610, 0.0003)
    assert out["optimizer_steps_performed"] == 1
    assert math.isfinite(out["synthetic_loss"])
    assert math.isfinite(out["gradient_norm_before_clip"])
    assert not torch.equal(initial, model.emb.weight.detach())
    for key in ("gpu_allocated", "real_engineering_result", "scientific_evidence",
                "paid_execution_authorized", "seed_consumed", "retry_authorized"):
        assert out[key] is False


def test_gradient_accumulation_matches_exact_mean_of_microbatch_losses():
    torch.manual_seed(7861)
    model = ToyLM()
    reference = ToyLM()
    reference.load_state_dict(model.state_dict())
    batches = two_batches()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    other = torch.optim.AdamW(reference.parameters(), lr=3e-4,
                              betas=(0.9, 0.95), weight_decay=0.1)
    out = cpu_synthetic_optimizer_step(model, optimizer, batches, 610)
    other.zero_grad(set_to_none=True)
    combined = torch.stack([
        F.cross_entropy(reference(x).float().reshape(-1, 19), y.reshape(-1))
        for x, y in batches
    ]).mean()
    combined.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    for group in other.param_groups:
        group["lr"] = cosine_lr(610, 30518, 610, 3e-4)
    other.step()
    assert math.isclose(out["synthetic_loss"], float(combined.detach()), rel_tol=1e-6)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)


def test_nonfinite_loss_rejects_update_and_clears_gradients():
    torch.manual_seed(127)
    model = ToyLM()
    with torch.no_grad():
        model.emb.weight[1, 0] = float("nan")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.raises(FloatingPointError, match="non-finite"):
        cpu_synthetic_optimizer_step(model, opt, two_batches(), 0)
    assert not opt.state
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("step", (-1, 3052, True))
def test_out_of_scope_optimizer_step_rejected(step):
    model = ToyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.raises(ValueError):
        cpu_synthetic_optimizer_step(model, opt, two_batches(), step)


def test_incomplete_microbatch_or_meta_model_is_rejected():
    model = ToyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.raises(ValueError):
        cpu_synthetic_optimizer_step(model, opt, two_batches()[:1], 0)
    with torch.device("meta"):
        meta = ToyLM()
    opt_meta = torch.optim.AdamW(meta.parameters(), lr=3e-4)
    with pytest.raises(RuntimeError, match="CPU-only"):
        cpu_synthetic_optimizer_step(meta, opt_meta, two_batches(), 0)


def test_no_modal_gpu_launch_or_research_data_io():
    from pathlib import Path
    s = (Path(__file__).resolve().parents[1] /
         "tam_research/cortex_attention8_conv7_engineering_cpu_kernel_v1.py").read_text()
    banned = ("import modal", ".remote(", "modal run", "gpu=", "cuda",
              "torch.compile(", ".write_text(", ".write_bytes(", "subprocess.",
              "requests.", "TokenBin(", "h100_train_one(")
    for term in banned:
        assert term not in s
