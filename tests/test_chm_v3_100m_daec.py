from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tam_research.chm_v3_100m_daec as daec


def test_stage_a_manifest_and_parameter_fairness_are_frozen() -> None:
    manifest = daec.protocol_manifest()
    accounting = manifest["parameter_accounting"]

    assert manifest["classification"] == "CHM_V3_100M_DAEC_STAGE_A_PREREGISTRATION_NO_GPU_AUTHORITY"
    assert manifest["issue"] == 1234
    assert manifest["engineering_seed"] == 1_234_001
    assert manifest["local_window"] == 512
    assert manifest["address_dim"] == 32
    assert manifest["retrieval_hops"] == 2
    assert manifest["retrieval_temperature"] == 0.10
    assert manifest["copy_gate_init"] == -4.0
    assert manifest["payload"] == "past_token_id"
    assert manifest["decoder_basis"] == "tied_token_embedding"
    assert manifest["hidden_memory_residual"] is False
    assert manifest["recency_bias"] is False

    assert accounting["hop_update_parameters"] == 16_384
    assert accounting["copy_gate_parameters"] == 513
    assert accounting["daec_added_parameters"] == 16_897
    assert accounting["daec_parameters"] == 101_853_697
    assert accounting["daec_minus_local_parameters"] == 50_177
    assert accounting["within_point_one_percent"] is True

    for field in (
        "gpu_authorized",
        "modal_authorized",
        "paid_compute_authorized",
        "training_execution_authorized",
        "scientific_seed_reserved",
        "stage_b_authorized",
        "stage_c_authorized",
        "stage_d_authorized",
    ):
        assert manifest[field] is False


def test_instantiated_parameter_accounting_matches_analytical_contract() -> None:
    accounting = daec.instantiated_parameter_accounting()
    assert accounting["instantiated_daec_parameters"] == 101_853_697
    assert accounting["instantiated_matches_analytical"] is True
    assert accounting["within_point_one_percent"] is True


def test_copy_distribution_sums_repeated_token_mass_in_vocab_space() -> None:
    weights = torch.tensor([[[0.10, 0.20, 0.30, 0.40]]], dtype=torch.float32)
    token_ids = torch.tensor([[2, 3, 2, 4]], dtype=torch.long)

    probs = daec.copy_distribution(weights, token_ids, vocab_size=6)

    expected = torch.tensor([[[0.0, 0.0, 0.40, 0.20, 0.40, 0.0]]])
    torch.testing.assert_close(probs, expected)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones((1, 1)))

    with pytest.raises(ValueError, match="non-empty"):
        daec.copy_distribution(
            torch.empty((1, 1, 0)),
            torch.empty((1, 0), dtype=torch.long),
            vocab_size=6,
        )
    with pytest.raises(ValueError, match="torch.long"):
        daec.copy_distribution(weights, token_ids.float(), vocab_size=6)
    with pytest.raises(ValueError, match="outside vocabulary"):
        daec.copy_distribution(weights, torch.tensor([[2, 3, 2, 9]]), vocab_size=6)


def test_no_memory_and_probability_mixture_have_exact_probability_semantics() -> None:
    torch.manual_seed(1234)
    base_logits = torch.randn(2, 3, 7)
    base_probs = torch.softmax(base_logits.float(), dim=-1)

    no_memory = daec.no_memory_log_probs(base_logits)
    torch.testing.assert_close(no_memory.exp(), base_probs, rtol=1e-6, atol=1e-7)

    copy_probs = torch.zeros_like(base_probs)
    copy_probs[..., 2] = 0.25
    copy_probs[..., 5] = 0.75

    zero_gate = torch.zeros(2, 3, 1)
    one_gate = torch.ones(2, 3, 1)
    half_gate = torch.full((2, 3, 1), 0.5)

    torch.testing.assert_close(
        daec.mix_lm_and_copy_log_probs(base_logits, copy_probs, zero_gate).exp(),
        base_probs,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        daec.mix_lm_and_copy_log_probs(base_logits, copy_probs, one_gate).exp(),
        copy_probs,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        daec.mix_lm_and_copy_log_probs(base_logits, copy_probs, half_gate).exp(),
        0.5 * base_probs + 0.5 * copy_probs,
        rtol=1e-6,
        atol=1e-7,
    )


def test_two_hop_copy_nll_reaches_query_key_hop_and_gate_parameters() -> None:
    torch.manual_seed(1235)
    d_model = 8
    address_dim = 4
    vocab_size = 11
    batch = 2
    queries = 3
    memory = 5

    core = daec.DecoderAlignedEpisodicCopy(
        d_model=d_model,
        address_dim=address_dim,
    )
    query_projection = nn.Linear(d_model, address_dim, bias=False)
    key_projection = nn.Linear(d_model, address_dim, bias=False)
    token_embedding = nn.Embedding(vocab_size, d_model)
    lm_head = nn.Linear(d_model, vocab_size, bias=False)

    hidden = torch.randn(batch, queries, d_model)
    memory_hidden = torch.randn(batch, memory, d_model)
    memory_token_ids = torch.tensor(
        [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]],
        dtype=torch.long,
    )

    first_query = F.normalize(query_projection(hidden), dim=-1)
    memory_keys = F.normalize(key_projection(memory_hidden), dim=-1)
    base_logits = lm_head(hidden)

    log_probs, trace = daec.two_hop_copy_log_probs(
        core,
        first_query=first_query,
        hidden=hidden,
        memory_keys=memory_keys,
        memory_token_ids=memory_token_ids,
        token_embedding_weight=token_embedding.weight,
        base_logits=base_logits,
    )

    targets = torch.tensor([[1, 5, 9], [2, 6, 10]], dtype=torch.long)
    loss = F.nll_loss(log_probs.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in (
        query_projection.weight,
        key_projection.weight,
        core.hop_update.weight,
        core.copy_gate.weight,
        core.copy_gate.bias,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0

    assert memory_token_ids.requires_grad is False
    torch.testing.assert_close(
        trace["first_hop_weights"].float().sum(dim=-1),
        torch.ones((batch, queries)),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        trace["second_hop_weights"].float().sum(dim=-1),
        torch.ones((batch, queries)),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        trace["copy_probs"].float().sum(dim=-1),
        torch.ones((batch, queries)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_first_hop_can_only_change_second_hop_address() -> None:
    torch.manual_seed(1236)
    core = daec.DecoderAlignedEpisodicCopy(d_model=8, address_dim=4)
    q0 = F.normalize(torch.randn(2, 3, 4), dim=-1)
    hop_embedding_a = torch.randn(2, 3, 8)
    hop_embedding_b = torch.randn(2, 3, 8)

    # Frozen zero init makes Stage-A construction an identity address update.
    q1_a = core.second_query(q0, hop_embedding_a)
    q1_b = core.second_query(q0, hop_embedding_b)
    torch.testing.assert_close(q1_a, q0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(q1_b, q0, rtol=1e-6, atol=1e-6)

    with torch.no_grad():
        core.hop_update.weight.normal_(mean=0.0, std=0.1)
    changed_a = core.second_query(q0, hop_embedding_a)
    changed_b = core.second_query(q0, hop_embedding_b)
    assert not torch.allclose(changed_a, q0)
    assert not torch.allclose(changed_a, changed_b)


class _TinyBackbone(nn.Module):
    def __init__(self, *, vocab_size: int = 17, d_model: int = 8) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList()
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight


class _TinyDAEC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.query_address = nn.Linear(8, 4, bias=False)
        self.key_address = nn.Linear(8, 4, bias=False)
        self.daec = daec.DecoderAlignedEpisodicCopy(d_model=8, address_dim=4)

    def query_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(hidden), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)


def test_session_path_is_local_without_prior_memory_and_writes_after_chunk() -> None:
    torch.manual_seed(1237)
    model = _TinyDAEC()
    first = torch.randint(0, 17, (1, 512))
    second_a = torch.randint(0, 17, (1, 512))
    second_b = torch.randint(0, 17, (1, 512))

    tokens_a = torch.cat((first, second_a), dim=1)
    tokens_b = torch.cat((first, second_b), dim=1)
    out_a = daec.daec_flat_training_session_log_probs(model, tokens_a)
    out_b = daec.daec_flat_training_session_log_probs(model, tokens_b)

    first_hidden = daec._hidden(model.backbone, first)
    expected_first = daec.no_memory_log_probs(model.backbone.lm_head(first_hidden))

    # The current first chunk cannot read itself as episodic memory.
    torch.testing.assert_close(out_a[:, :512], expected_first, rtol=1e-6, atol=1e-6)
    # Changing only the later chunk cannot alter already-produced first-chunk outputs.
    torch.testing.assert_close(out_a[:, :512], out_b[:, :512], rtol=0.0, atol=0.0)


def test_hidden_residual_memory_path_is_explicitly_forbidden() -> None:
    model = daec.CHMV3100MDAECLM()
    hidden = torch.randn(2, 8)
    memory = torch.randn(2, 8)
    with pytest.raises(RuntimeError, match="forbids hidden-state memory residual"):
        model._integrate(hidden, memory)

    source = inspect.getsource(daec)
    assert "model._integrate(" not in source
    assert "value_up" not in source
    assert "value_down" not in source
    assert "QueryConditionedValueAdapter" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "torch.save(" not in source
    assert "import modal" not in source


def test_engineering_seed_guard_refuses_every_consumed_scientific_seed() -> None:
    assert daec.validate_engineering_seed(1_234_001) == 1_234_001
    for seed in daec.CONSUMED_OR_RESERVED_SCIENTIFIC_SEEDS:
        with pytest.raises(RuntimeError, match="historical/reserved"):
            daec.validate_engineering_seed(seed)
    with pytest.raises(RuntimeError, match="engineering seed"):
        daec.validate_engineering_seed(1_234_002)


def test_stage_a_preflight_is_engineering_only() -> None:
    result = daec.stage_a_preflight(instantiate=False)
    assert result["classification"] == "CHM_V3_100M_DAEC_STAGE_A_IMPLEMENTATION_PASS"
    assert result["parameter_accounting"]["within_point_one_percent"] is True
    assert result["gpu_authorized"] is False
    assert result["training_execution_authorized"] is False
    assert result["scientific_seed_reserved"] is False
    assert result["stage_b_authorized"] is False
    assert result["stage_c_authorized"] is False
    assert result["stage_d_authorized"] is False
