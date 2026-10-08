from __future__ import annotations

import torch
import torch.nn.functional as F

from tam_research.cpw_binding_v2.task import (
    counterfactual_query,
    explicit_key_lookup,
    make_binding_batch,
)
from tam_research.cpw_r1.model import (
    CPWR1ResearchLM,
    CompetitiveRetrievalMixer,
    MODEL_PARAMETERS,
    RETRIEVAL_BLOCK_INDEX,
    SEQUENCE_MIXER_PARAMETERS,
    cpw_r1_parameter_count,
)
from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.models import parameter_count


def test_parameter_match_and_one_global_layer_only() -> None:
    model = CPWR1ResearchLM()
    baseline = SparseWorldCPWResearchLM("sequence_only")
    assert cpw_r1_parameter_count() == MODEL_PARAMETERS == 21_745_408
    assert parameter_count(baseline) == parameter_count(model)
    assert parameter_count(model.blocks[14].mixer) == SEQUENCE_MIXER_PARAMETERS
    assert model.retrieval_layers == (RETRIEVAL_BLOCK_INDEX,)
    assert len(model.blocks) == 15
    for i, block in enumerate(model.blocks):
        if i == RETRIEVAL_BLOCK_INDEX:
            assert isinstance(block.mixer, CompetitiveRetrievalMixer)
            assert block.mixer.attention.n_heads == 1
            assert block.mixer.attention.inner == 48
        else:
            assert block.mixer.world is None
            assert block.mixer.sequence is not None
            assert not any("attention" in n for n, _ in block.named_modules())


def test_fourteen_local_blocks_are_identical_under_paired_initialization() -> None:
    torch.manual_seed(1314)
    original = SparseWorldCPWResearchLM("sequence_only")
    torch.manual_seed(1314)
    hybrid = CPWR1ResearchLM()
    source = original.state_dict()
    target = hybrid.state_dict()
    for name, value in source.items():
        if name.startswith("blocks.14.mixer."):
            continue
        assert name in target
        torch.testing.assert_close(value, target[name], rtol=0, atol=0)


def test_strict_future_token_causality() -> None:
    torch.manual_seed(1315)
    model = CPWR1ResearchLM(CPWV1Config(max_seq_len=64)).eval()
    x = torch.randint(0, model.cfg.vocab_size, (1, 32))
    changed = x.clone()
    changed[:, 24:] = torch.randint(0, model.cfg.vocab_size, (1, 8))
    with torch.no_grad():
        a = model(x)
        b = model(changed)
    torch.testing.assert_close(a[:, :24], b[:, :24], rtol=0, atol=3e-5)


def test_nonlocal_source_dependency_where_sequence_only_cannot_reach() -> None:
    torch.manual_seed(1316)
    model = CPWR1ResearchLM(CPWV1Config(max_seq_len=96)).eval()
    torch.manual_seed(1316)
    baseline = SparseWorldCPWResearchLM(
        "sequence_only", CPWV1Config(max_seq_len=96)
    ).eval()

    x = torch.randint(0, model.cfg.vocab_size, (1, 80))
    changed = x.clone()
    changed[:, 2] = (changed[:, 2] + 1) % model.cfg.vocab_size

    with torch.no_grad():
        a = model.query_logits(x)
        b = model.query_logits(changed)
        slow_a = baseline(x)[:, -1]
        slow_b = baseline(changed)[:, -1]

    # A 15-layer local shift has no path from token 2 to the final token 79.
    torch.testing.assert_close(slow_a, slow_b, rtol=0, atol=1e-6)
    assert float((a - b).abs().max()) > 1e-8


def test_corrected_binding_task_query_only_cpu_gradient() -> None:
    torch.manual_seed(1317)
    model = CPWR1ResearchLM()
    batch = make_binding_batch(
        batch_size=2,
        generator=torch.Generator().manual_seed(1317),
        delay=64,
    )
    alternative, alternative_targets = counterfactual_query(
        batch, new_delay=128
    )
    assert torch.equal(batch.tokens[:, :-1], alternative[:, :-1])
    assert torch.all(batch.targets != alternative_targets)
    torch.testing.assert_close(explicit_key_lookup(batch.tokens), batch.targets)
    torch.testing.assert_close(
        explicit_key_lookup(alternative), alternative_targets
    )

    logits = model.query_logits(batch.tokens)
    assert logits.shape == (2, 50_257)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.float(), batch.targets)
    assert torch.isfinite(loss)
    loss.backward()
    mixer = model.blocks[RETRIEVAL_BLOCK_INDEX].mixer
    assert isinstance(mixer, CompetitiveRetrievalMixer)
    for layer in (mixer.attention.qkv, mixer.attention.out):
        grad = layer.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0

    # This is an integrity smoke only. No expected accuracy or NLL win.
