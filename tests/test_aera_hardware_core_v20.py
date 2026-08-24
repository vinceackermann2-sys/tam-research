import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import diagnostic_config
from tam_research.aera_hardware_core_v19 import HardwareAwareAERATextLMV19
from tam_research.aera_hardware_core_v20 import HardwareAwareAERATextLMV20


def _build_pair(seed: int = 9201):
    torch.manual_seed(seed)
    v19 = HardwareAwareAERATextLMV19(diagnostic_config())
    torch.manual_seed(seed)
    v20 = HardwareAwareAERATextLMV20(diagnostic_config())
    return v19, v20


def test_v20_memory_off_is_exact_v19_and_common_checkpoint_keys_match():
    v19, v20 = _build_pair()
    old = v19.state_dict()
    new = v20.state_dict()
    extra = [k for k in new if ".write_extractor." in k]
    assert len(extra) == 2 * v20.cfg.n_stages
    for key, value in old.items():
        assert key in new
        torch.testing.assert_close(new[key], value, atol=0.0, rtol=0.0)

    tokens = torch.randint(0, v19.cfg.vocab_size, (2, 2 * v19.cfg.chunk_size))
    v19.eval(); v20.eval()
    with torch.no_grad():
        left = v19(tokens, hard=True, update_memory=False)
        right = v20(tokens, hard=True, update_memory=False)
    torch.testing.assert_close(left["logits"], right["logits"], atol=0.0, rtol=0.0)
    for ls, rs in zip(left["state"].stages, right["state"].stages):
        torch.testing.assert_close(ls.stream, rs.stream, atol=0.0, rtol=0.0)
        torch.testing.assert_close(ls.memory.matrix, rs.memory.matrix, atol=0.0, rtol=0.0)


def test_v20_pool_weights_are_finite_normalized_and_state_shape_unchanged():
    _, model = _build_pair(9202)
    tokens = torch.randint(0, model.cfg.vocab_size, (3, model.cfg.chunk_size))
    model.eval()
    with torch.no_grad():
        out = model(tokens, hard=True, update_memory=True)
    for stage, state in zip(model.stages, out["state"].stages):
        aw = stage.write_extractor.last_address_weights
        pw = stage.write_extractor.last_payload_weights
        assert aw is not None and pw is not None
        assert torch.isfinite(aw).all() and torch.isfinite(pw).all()
        torch.testing.assert_close(aw.sum(dim=-1), torch.ones(aw.size(0)), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(pw.sum(dim=-1), torch.ones(pw.size(0)), atol=1e-6, rtol=1e-6)
        assert state.memory.matrix.shape == (tokens.size(0), model.cfg.memory_dim, model.cfg.memory_dim)


def test_v20_write_is_future_only_and_cannot_change_current_chunk_logits():
    _, model = _build_pair(9203)
    model.eval()
    c = model.cfg.chunk_size
    first = torch.randint(0, model.cfg.vocab_size, (2, c))
    future_a = torch.randint(0, model.cfg.vocab_size, (2, c))
    future_b = torch.randint(0, model.cfg.vocab_size, (2, c))
    tokens_a = torch.cat((first, future_a), dim=1)
    tokens_b = torch.cat((first, future_b), dim=1)
    with torch.no_grad():
        out_a = model(tokens_a, hard=True, update_memory=True)["logits"]
        out_b = model(tokens_b, hard=True, update_memory=True)["logits"]
    torch.testing.assert_close(out_a[:, :c], out_b[:, :c], atol=0.0, rtol=0.0)


def test_v20_differentiable_memory_reaches_both_pools_and_existing_qkvo():
    _, model = _build_pair(9204)
    model.train()
    c = model.cfg.chunk_size
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 2 * c))
    model.set_memory_pretraining_mode(True)
    try:
        out = model(tokens, hard=True, update_memory=True)
        logits = out["logits"]
        # Only second-chunk predictions depend on the first factorized write.
        loss = F.cross_entropy(
            logits[:, c:-1].float().reshape(-1, model.cfg.vocab_size),
            tokens[:, c + 1:].reshape(-1),
        )
        loss.backward()
    finally:
        model.set_memory_pretraining_mode(False)

    stage = model.stages[0]
    params = {
        "address_pool": stage.write_extractor.address_score.weight,
        "payload_pool": stage.write_extractor.payload_score.weight,
        "q": stage.memory.q.weight,
        "k": stage.memory.k.weight,
        "v": stage.memory.v.weight,
        "out": stage.memory.out.weight,
    }
    for name, parameter in params.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0.0, name


def test_v20_deployment_write_mutates_no_base_parameters_and_state_is_detached():
    _, model = _build_pair(9205)
    model.eval()
    model.set_memory_pretraining_mode(False)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 2 * model.cfg.chunk_size))
    versions = [p._version for p in model.parameters()]
    out = model(tokens, hard=True, update_memory=True)
    assert versions == [p._version for p in model.parameters()]
    for state in out["state"].stages:
        assert not state.memory.matrix.requires_grad


def test_v20_session_state_is_exactly_isolated():
    _, model = _build_pair(9206)
    model.eval()
    model.set_memory_pretraining_mode(False)
    c = model.cfg.chunk_size
    query = torch.randint(0, model.cfg.vocab_size, (2, 2 * c))
    other = torch.randint(0, model.cfg.vocab_size, (2, 3 * c))
    with torch.no_grad():
        fresh_before = model(query, state=None, hard=True, update_memory=False)["logits"]
        _ = model(other, state=None, hard=True, update_memory=True)
        fresh_after = model(query, state=None, hard=True, update_memory=False)["logits"]
    assert torch.equal(fresh_before, fresh_after)


def test_v20_parameter_delta_is_only_two_pool_vectors_per_stage():
    v19, v20 = _build_pair(9207)
    old = sum(p.numel() for p in v19.parameters())
    new = sum(p.numel() for p in v20.parameters())
    assert new - old == 2 * v20.cfg.d_model * v20.cfg.n_stages
