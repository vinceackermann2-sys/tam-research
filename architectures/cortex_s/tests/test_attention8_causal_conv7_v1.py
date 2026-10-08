from __future__ import annotations

import torch

from architectures.cortex_s.attention8_causal_conv7_v1 import (
    Attention8CausalConv7LM,
    CausalConv7ScheduledBlock,
    CONV7_PARAMETERS,
    EXPECTED_PARAMETERS,
    KERNEL_SIZE,
    VARIANT,
    prototype_contract,
)
from architectures.cortex_s.reduced_attention_200m_panel_v2 import (
    EXPECTED_TRANSFORMER_PARAMETERS,
    ScheduledAttentionBlock,
)


def test_parameter_math_and_prototype_safety():
    contract = prototype_contract()
    assert contract["status"] == "CPU_PROTOTYPE_ONLY_UNTRAINED"
    assert contract["attention_layers_one_based"] == [3, 6, 9, 12, 15, 18, 21, 24]
    assert len(contract["conv_layers_one_based"]) == 16
    assert set(contract["conv_layers_one_based"]).isdisjoint(set(contract["attention_layers_one_based"]))
    assert KERNEL_SIZE == 7 and CONV7_PARAMETERS == 57360
    assert contract["expected_parameters"] == EXPECTED_PARAMETERS == EXPECTED_TRANSFORMER_PARAMETERS + 16
    for key in ("science_seed_assigned", "training_authorized", "gpu_authorized",
                "replication_authorized", "250m_5b_authorized", "breakthrough_claim_allowed"):
        assert contract[key] is False


def test_non_attention_block_is_strictly_causal_cpu():
    torch.manual_seed(1234)
    block = CausalConv7ScheduledBlock(index=0, variant=VARIANT).cpu().eval()
    with torch.no_grad():
        block.conv_gain.fill_(0.5)
        x = torch.randn(2, 14, 512)
        changed = x.clone()
        changed[:, 9:, :] = torch.randn_like(changed[:, 9:, :])
        y1 = block(x)
        y2 = block(changed)
        torch.testing.assert_close(y1[:, :9], y2[:, :9], atol=1e-6, rtol=1e-6)
        # Changing the past can affect future outputs; changing the future cannot affect the past.
        changed_past = x.clone()
        changed_past[:, 4, 12] += 9.0
        y3 = block(changed_past)
        assert not torch.allclose(y1[:, 4:11], y3[:, 4:11])


def test_zero_gain_is_exact_ffn_only_initialization():
    torch.manual_seed(5678)
    baseline = ScheduledAttentionBlock(index=1, variant=VARIANT).cpu().eval()
    enhanced = CausalConv7ScheduledBlock(index=1, variant=VARIANT).cpu().eval()
    enhanced.norm_ff.load_state_dict(baseline.norm_ff.state_dict())
    enhanced.feedforward.load_state_dict(baseline.feedforward.state_dict())
    assert float(enhanced.conv_gain.detach()) == 0.0
    with torch.no_grad():
        x = torch.randn(1, 11, 512)
        torch.testing.assert_close(baseline(x), enhanced(x), atol=0.0, rtol=0.0)


def test_full_model_parameter_count_and_layer_placement_cpu():
    model = Attention8CausalConv7LM().cpu().eval()
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS
    active = [b.layer_number for b in model.blocks if b.has_attention]
    conv = [b.layer_number for b in model.blocks if isinstance(b, CausalConv7ScheduledBlock)]
    assert active == [3, 6, 9, 12, 15, 18, 21, 24]
    assert len(conv) == 16
    assert all(not model.blocks[i - 1].has_attention for i in conv)
