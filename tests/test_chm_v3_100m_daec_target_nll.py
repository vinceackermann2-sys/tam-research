from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import tam_research.chm_v3_100m_daec as dense
import tam_research.chm_v3_100m_daec_target_nll as fast


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    payload = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(payload).hexdigest()


def test_reference_daec_module_is_byte_identical() -> None:
    path = Path(dense.__file__).resolve()
    assert _git_blob_sha(path) == fast.REFERENCE_DAEC_BLOB


def test_manifest_is_zero_gpu_systems_only() -> None:
    manifest = fast.optimization_manifest()
    assert manifest["classification"] == "CHM_V3_100M_DAEC_TARGET_NLL_OPTIMIZATION_EQUIVALENCE_PASS"
    assert manifest["issue"] == 1262
    assert manifest["reference_daec_blob"] == "77e9ccbff4383be40e6e2865503e1c3926bbda74"
    assert manifest["retrieval_hops"] == 2
    assert manifest["retrieval_temperature"] == 0.10
    assert manifest["dense_copy_distribution_allocated"] is False
    assert manifest["architecture_changed"] is False
    assert manifest["parameters_changed"] is False
    assert manifest["objective_changed"] is False
    assert manifest["gpu_authorized"] is False
    assert manifest["paid_compute_authorized"] is False
    assert manifest["replacement_stage_b_attempt_authorized"] is False
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["stage_c_authorized"] is False
    assert manifest["stage_d_authorized"] is False


def test_target_copy_probability_matches_dense_scatter_with_repeated_tokens() -> None:
    weights = torch.tensor(
        [
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.40, 0.30, 0.20, 0.10],
            ]
        ],
        dtype=torch.float32,
    )
    memory_token_ids = torch.tensor([[2, 3, 2, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 4]], dtype=torch.long)

    dense_probs = dense.copy_distribution(weights, memory_token_ids, vocab_size=7)
    expected = dense_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    actual = fast.target_copy_probability(weights, memory_token_ids, targets)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, torch.tensor([[0.40, 0.10]]), rtol=0.0, atol=0.0)


def _run_two_hop_path(mode: str):
    torch.manual_seed(1262)
    d_model = 8
    address_dim = 4
    vocab_size = 13
    batch = 2
    queries = 3
    memory = 5

    core = dense.DecoderAlignedEpisodicCopy(d_model=d_model, address_dim=address_dim)
    query_projection = nn.Linear(d_model, address_dim, bias=False)
    key_projection = nn.Linear(d_model, address_dim, bias=False)
    token_embedding = nn.Embedding(vocab_size, d_model)

    hidden = torch.randn(batch, queries, d_model)
    memory_hidden = torch.randn(batch, memory, d_model)
    memory_token_ids = torch.tensor(
        [[1, 3, 1, 7, 9], [2, 4, 6, 2, 10]],
        dtype=torch.long,
    )
    targets = torch.tensor([[1, 7, 9], [2, 6, 10]], dtype=torch.long)

    first_query = F.normalize(query_projection(hidden), dim=-1)
    memory_keys = F.normalize(key_projection(memory_hidden), dim=-1)
    base_logits = F.linear(hidden, token_embedding.weight)

    if mode == "dense":
        log_probs, trace = dense.two_hop_copy_log_probs(
            core,
            first_query=first_query,
            hidden=hidden,
            memory_keys=memory_keys,
            memory_token_ids=memory_token_ids,
            token_embedding_weight=token_embedding.weight,
            base_logits=base_logits,
        )
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        copy_target = trace["copy_probs"].gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
    elif mode == "fast":
        target_log_probs, trace = fast.two_hop_copy_target_log_probs(
            core,
            first_query=first_query,
            hidden=hidden,
            memory_keys=memory_keys,
            memory_token_ids=memory_token_ids,
            token_embedding_weight=token_embedding.weight,
            base_logits=base_logits,
            targets=targets,
        )
        copy_target = trace["copy_target_probs"]
    else:
        raise ValueError(mode)

    loss = -target_log_probs.mean()
    loss.backward()

    grads = {
        "query": query_projection.weight.grad.detach().clone(),
        "key": key_projection.weight.grad.detach().clone(),
        "hop": core.hop_update.weight.grad.detach().clone(),
        "gate_weight": core.copy_gate.weight.grad.detach().clone(),
        "gate_bias": core.copy_gate.bias.grad.detach().clone(),
        "embedding": token_embedding.weight.grad.detach().clone(),
    }
    return target_log_probs.detach(), copy_target.detach(), loss.detach(), grads


def test_target_only_two_hop_matches_dense_loss_and_gradients() -> None:
    dense_target, dense_copy, dense_loss, dense_grads = _run_two_hop_path("dense")
    fast_target, fast_copy, fast_loss, fast_grads = _run_two_hop_path("fast")

    torch.testing.assert_close(fast_copy, dense_copy, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(fast_target, dense_target, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(fast_loss, dense_loss, rtol=2e-6, atol=2e-7)

    assert dense_grads.keys() == fast_grads.keys()
    for name in dense_grads:
        assert torch.isfinite(dense_grads[name]).all(), name
        assert torch.isfinite(fast_grads[name]).all(), name
        torch.testing.assert_close(
            fast_grads[name],
            dense_grads[name],
            rtol=2e-5,
            atol=2e-7,
            msg=f"gradient mismatch for {name}",
        )


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
        self.daec = dense.DecoderAlignedEpisodicCopy(d_model=8, address_dim=4)

    def query_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_address(hidden), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_address(hidden), dim=-1)


def test_full_session_target_log_probs_match_dense_reference() -> None:
    torch.manual_seed(1263)
    model = _TinyDAEC()
    tokens = torch.randint(0, 17, (1, 1024), dtype=torch.long)
    targets = torch.randint(0, 17, (1, 1024), dtype=torch.long)

    dense_log_probs = dense.daec_flat_training_session_log_probs(model, tokens)
    dense_target = dense_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    fast_target = fast.daec_flat_training_session_target_log_probs(
        model,
        tokens,
        targets,
    )
    fast_nll = fast.daec_flat_training_session_nll(model, tokens, targets)

    torch.testing.assert_close(fast_target, dense_target, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(fast_nll, -dense_target.mean(), rtol=2e-6, atol=2e-7)


def test_no_memory_chunk_matches_ordinary_target_cross_entropy() -> None:
    torch.manual_seed(1264)
    model = _TinyDAEC()
    tokens = torch.randint(0, 17, (1, 512), dtype=torch.long)
    targets = torch.randint(0, 17, (1, 512), dtype=torch.long)

    hidden = dense._hidden(model.backbone, tokens)
    base_logits = model.backbone.lm_head(hidden)
    expected = -F.cross_entropy(
        base_logits.float().reshape(-1, 17),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)

    actual = fast.daec_flat_training_session_target_log_probs(model, tokens, targets)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fast_path_has_no_dense_copy_distribution_dependency() -> None:
    source = inspect.getsource(fast)
    assert "copy_distribution(" not in source
    assert "new_zeros((batch, queries, vocab_size" not in source
    assert "scatter_add_" not in source

    torch.manual_seed(1265)
    weights = torch.softmax(torch.randn(2, 3, 5), dim=-1)
    memory_token_ids = torch.tensor(
        [[1, 2, 1, 4, 5], [2, 2, 3, 4, 6]],
        dtype=torch.long,
    )
    targets = torch.tensor([[1, 4, 5], [2, 3, 6]], dtype=torch.long)
    result = fast.target_copy_probability(weights, memory_token_ids, targets)
    assert result.shape == targets.shape


def test_model_parameter_accounting_is_unchanged() -> None:
    accounting = dense.instantiated_parameter_accounting()
    assert accounting["instantiated_daec_parameters"] == 101_853_697
    assert accounting["instantiated_matches_analytical"] is True
