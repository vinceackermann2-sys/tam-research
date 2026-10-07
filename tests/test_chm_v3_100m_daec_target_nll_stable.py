from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import tam_research.chm_v3_100m_daec as dense
import tam_research.chm_v3_100m_daec_target_nll as old_fast
import tam_research.chm_v3_100m_daec_target_nll_stable as stable


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    payload = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(payload).hexdigest()


def test_frozen_reference_files_are_byte_identical() -> None:
    assert _git_blob_sha(Path(dense.__file__).resolve()) == stable.REFERENCE_DAEC_BLOB
    assert (
        _git_blob_sha(Path(old_fast.__file__).resolve())
        == stable.REFERENCE_TARGET_NLL_BLOB
    )


def test_manifest_is_zero_gpu_numerical_only() -> None:
    manifest = stable.stabilization_manifest()
    assert (
        manifest["classification"]
        == "CHM_V3_100M_DAEC_TARGET_NLL_ZERO_MASS_GRADIENT_STABILIZATION_PASS"
    )
    assert manifest["issue"] == 1288
    assert manifest["reference_daec_blob"] == "77e9ccbff4383be40e6e2865503e1c3926bbda74"
    assert (
        manifest["reference_target_nll_blob"]
        == "0b0a68f47186f154a48d1d75e3fe3b236188d69d"
    )
    assert manifest["retrieval_hops"] == 2
    assert manifest["retrieval_temperature"] == 0.10
    assert manifest["forward_probability_model_changed"] is False
    assert manifest["architecture_changed"] is False
    assert manifest["parameters_changed"] is False
    assert manifest["objective_changed"] is False
    assert manifest["dense_copy_distribution_allocated"] is False
    assert manifest["structural_zero_log_domain_guard"] is True
    assert manifest["gpu_authorized"] is False
    assert manifest["paid_compute_authorized"] is False
    assert manifest["replacement_stage_b_attempt_authorized"] is False
    assert manifest["scientific_seed_authorized"] is False
    assert manifest["stage_c_authorized"] is False
    assert manifest["stage_d_authorized"] is False


def test_old_masked_log_zero_reproduces_nan_gradient_and_stable_path_fixes_it() -> None:
    old_copy = torch.tensor([[0.0, 0.25]], requires_grad=True)
    new_copy = old_copy.detach().clone().requires_grad_(True)
    base = torch.tensor([[-2.0, -2.0]])
    gate = torch.tensor([[0.2, 0.2]])

    old_out = old_fast.mix_target_log_probabilities(base, old_copy, gate)
    new_out = stable.mix_target_log_probabilities_stable(base, new_copy, gate)

    torch.testing.assert_close(new_out, old_out, rtol=0.0, atol=0.0)
    assert torch.isfinite(old_out).all()
    assert torch.isfinite(new_out).all()

    (-old_out.mean()).backward()
    (-new_out.mean()).backward()

    assert old_copy.grad is not None
    assert torch.isnan(old_copy.grad[0, 0])
    assert new_copy.grad is not None
    assert torch.isfinite(new_copy.grad).all()
    assert new_copy.grad[0, 0].item() == 0.0


def test_safe_log_has_exact_forward_values_and_finite_zero_derivative() -> None:
    values = torch.tensor([0.0, 1e-8, 0.25, 1.0], requires_grad=True)
    out = stable._log_positive_or_neg_inf(values)
    expected = torch.tensor(
        [float("-inf"), torch.log(torch.tensor(1e-8)), torch.log(torch.tensor(0.25)), 0.0]
    )
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)

    finite_terms = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    finite_terms.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert values.grad[0].item() == 0.0


def test_stable_mixture_matches_old_forward_for_zero_present_and_gate_boundaries() -> None:
    base = torch.tensor([[-2.3, -1.7, -0.4, -4.0]], dtype=torch.float32)
    copy = torch.tensor([[0.0, 0.2, 1.0, 0.0]], dtype=torch.float32)
    gate = torch.tensor([[0.0, 0.25, 1.0, 1.0]], dtype=torch.float32)

    old = old_fast.mix_target_log_probabilities(base, copy, gate)
    new = stable.mix_target_log_probabilities_stable(base, copy, gate)
    torch.testing.assert_close(new, old, rtol=0.0, atol=0.0, equal_nan=True)


def _run_two_hop(mode: str, targets: torch.Tensor):
    torch.manual_seed(1288)
    d_model = 8
    address_dim = 4
    vocab_size = 13
    batch = 2
    queries = 4
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
    elif mode == "old":
        target_log_probs, trace = old_fast.two_hop_copy_target_log_probs(
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
    elif mode == "stable":
        target_log_probs, trace = stable.two_hop_copy_target_log_probs_stable(
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

    def grad(parameter: torch.Tensor) -> torch.Tensor:
        assert parameter.grad is not None
        return parameter.grad.detach().clone()

    grads = {
        "query": grad(query_projection.weight),
        "key": grad(key_projection.weight),
        "hop": grad(core.hop_update.weight),
        "gate_weight": grad(core.copy_gate.weight),
        "gate_bias": grad(core.copy_gate.bias),
        "embedding": grad(token_embedding.weight),
    }
    return target_log_probs.detach(), copy_target.detach(), loss.detach(), grads


def _assert_gradient_maps_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.isfinite(expected[name]).all(), f"dense nonfinite: {name}"
        assert torch.isfinite(actual[name]).all(), f"stable nonfinite: {name}"
        torch.testing.assert_close(
            actual[name],
            expected[name],
            rtol=3e-5,
            atol=3e-7,
            msg=f"gradient mismatch for {name}",
        )


def test_stable_two_hop_matches_dense_for_mixed_present_absent_and_repeated_targets() -> None:
    # 12 and 11 are absent from both memories. 1/7/2/6 are present; 1 and 2
    # are repeated in their respective memory rows.
    targets = torch.tensor(
        [[12, 1, 7, 11], [12, 2, 6, 11]],
        dtype=torch.long,
    )
    dense_target, dense_copy, dense_loss, dense_grads = _run_two_hop("dense", targets)
    stable_target, stable_copy, stable_loss, stable_grads = _run_two_hop("stable", targets)

    torch.testing.assert_close(stable_copy, dense_copy, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(stable_target, dense_target, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(stable_loss, dense_loss, rtol=2e-6, atol=2e-7)
    _assert_gradient_maps_close(stable_grads, dense_grads)

    assert (stable_copy[:, 0] == 0).all()
    assert (stable_copy[:, 3] == 0).all()


def test_old_target_only_path_has_nonfinite_grads_when_targets_are_absent() -> None:
    targets = torch.full((2, 4), 12, dtype=torch.long)
    _, copy_target, loss, grads = _run_two_hop("old", targets)

    assert torch.isfinite(loss)
    assert (copy_target == 0).all()
    assert any(not torch.isfinite(value).all() for value in grads.values())


def test_stable_absent_only_path_matches_dense_and_has_finite_gradients() -> None:
    targets = torch.full((2, 4), 12, dtype=torch.long)
    dense_target, dense_copy, dense_loss, dense_grads = _run_two_hop("dense", targets)
    stable_target, stable_copy, stable_loss, stable_grads = _run_two_hop("stable", targets)

    assert (dense_copy == 0).all()
    assert (stable_copy == 0).all()
    torch.testing.assert_close(stable_target, dense_target, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(stable_loss, dense_loss, rtol=2e-6, atol=2e-7)
    _assert_gradient_maps_close(stable_grads, dense_grads)


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


def _session_model(seed: int) -> _TinyDAEC:
    torch.manual_seed(seed)
    return _TinyDAEC()


def test_full_session_stable_forward_matches_dense_with_absent_second_chunk_targets() -> None:
    torch.manual_seed(1289)
    tokens = torch.cat(
        (
            torch.randint(0, 6, (1, 512), dtype=torch.long),
            torch.randint(0, 6, (1, 512), dtype=torch.long),
        ),
        dim=1,
    )
    targets = torch.cat(
        (
            torch.randint(0, 17, (1, 512), dtype=torch.long),
            torch.full((1, 512), 16, dtype=torch.long),
        ),
        dim=1,
    )

    dense_model = _session_model(1290)
    stable_model = _session_model(1290)

    dense_log_probs = dense.daec_flat_training_session_log_probs(dense_model, tokens)
    dense_target = dense_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    stable_target = stable.daec_flat_training_session_target_log_probs_stable(
        stable_model,
        tokens,
        targets,
    )

    torch.testing.assert_close(stable_target, dense_target, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        stable.daec_flat_training_session_nll_stable(stable_model, tokens, targets),
        -dense_target.mean(),
        rtol=2e-6,
        atol=2e-7,
    )


def test_stable_full_session_backward_is_finite_when_second_chunk_target_is_absent() -> None:
    torch.manual_seed(1291)
    tokens = torch.cat(
        (
            torch.randint(0, 6, (1, 512), dtype=torch.long),
            torch.randint(0, 6, (1, 512), dtype=torch.long),
        ),
        dim=1,
    )
    targets = torch.cat(
        (
            torch.randint(0, 17, (1, 512), dtype=torch.long),
            torch.full((1, 512), 16, dtype=torch.long),
        ),
        dim=1,
    )
    model = _session_model(1292)
    loss = stable.daec_flat_training_session_nll_stable(model, tokens, targets)
    loss.backward()

    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_stable_path_does_not_allocate_dense_copy_vocab_distribution() -> None:
    source = inspect.getsource(stable)
    assert "copy_distribution(" not in source
    assert "new_zeros((batch, queries, vocab_size" not in source
    assert "scatter_add_" not in source
    assert "torch.optim" not in source
    assert "import modal" not in source


def test_parameter_accounting_is_unchanged() -> None:
    accounting = dense.analytical_parameter_accounting()
    assert accounting["daec_parameters"] == 101_853_697
    assert accounting["within_point_one_percent"] is True
