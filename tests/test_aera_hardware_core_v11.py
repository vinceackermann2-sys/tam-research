from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v11 import HardwareAwareAERATextLMV11
from tam_research.models import ModelConfig, ResearchLM


def _aera() -> HardwareAwareAERATextLMV11:
    return HardwareAwareAERATextLMV11(
        HardwareAERAConfig(
            vocab_size=50_257,
            d_model=200,
            n_stages=4,
            n_heads=8,
            chunk_size=256,
            n_experts=8,
            max_active_experts=2,
            expert_mult=4,
            memory_dim=50,
            max_reason_steps=4,
            block_size=4,
        )
    )


def _transformer() -> ResearchLM:
    return ResearchLM(
        ModelConfig(
            vocab_size=50_257,
            d_model=256,
            n_layers=15,
            n_heads=8,
            max_seq_len=1024,
            ff_mult=4,
            architecture="transformer",
        )
    )


def test_v11_matches_gpt_style_embedding_scale_and_ties_head():
    torch.manual_seed(8201)
    model = _aera()
    assert model.lm_head.weight is model.token_emb.weight
    assert 0.018 < float(model.token_emb.weight.std()) < 0.022
    assert 0.018 < float(model.local_pos.weight.std()) < 0.022
    for router in model.stage_routers:
        assert torch.count_nonzero(router.proj.bias) == 0


def test_v11_and_transformer_start_at_comparable_language_nll():
    torch.manual_seed(8201)
    aera = _aera().eval()
    torch.manual_seed(8201)
    transformer = _transformer().eval()
    g = torch.Generator().manual_seed(18_201)
    x = torch.randint(0, 50_257, (2, 32), generator=g)
    y = torch.randint(0, 50_257, (2, 32), generator=g)
    with torch.no_grad():
        aout = aera(x, hard=False, route_mode="soft", update_memory=False)
        alogits = aout["logits"]
        assert isinstance(alogits, torch.Tensor)
        tlogits = transformer(x)
    anll = float(F.cross_entropy(alogits.float().reshape(-1, 50_257), y.reshape(-1)))
    tnll = float(F.cross_entropy(tlogits.float().reshape(-1, 50_257), y.reshape(-1)))
    chance = math.log(50_257)
    assert abs(anll - chance) < 0.75
    assert abs(tnll - chance) < 0.75
    assert abs(anll - tnll) < 0.50
    assert float(alogits.float().std()) < 2.0
