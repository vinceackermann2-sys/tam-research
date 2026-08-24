import torch
import torch.nn.functional as F

from aera_v19_memory_necessity_cpu import diagnostic_config
from tam_research.aera_hardware_core_v19 import HardwareAwareAERATextLMV19
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21


def _build_pair(seed: int = 9401):
    torch.manual_seed(seed)
    v19 = HardwareAwareAERATextLMV19(diagnostic_config())
    torch.manual_seed(seed)
    v21 = HardwareAwareAERATextLMV21(diagnostic_config())
    return v19, v21


def test_v21_memory_off_is_exact_v19_for_common_checkpoint_weights():
    v19, v21 = _build_pair()
    old = v19.state_dict()
    new = v21.state_dict()
    extras = [k for k in new if ".pair_write_gate." in k]
    assert len(extras) == 2 * v21.cfg.n_stages
    for key, value in old.items():
        assert key in new
        torch.testing.assert_close(new[key], value, atol=0.0, rtol=0.0)

    tokens = torch.randint(0, v19.cfg.vocab_size, (2, 2 * v19.cfg.chunk_size))
    v19.eval(); v21.eval()
    with torch.no_grad():
        left = v19(tokens, hard=True, update_memory=False)
        right = v21(tokens, hard=True, update_memory=False)
    torch.testing.assert_close(left["logits"], right["logits"], atol=0.0, rtol=0.0)
    for ls, rs in zip(left["state"].stages, right["state"].stages):
        torch.testing.assert_close(ls.stream, rs.stream, atol=0.0, rtol=0.0)
        torch.testing.assert_close(ls.memory.matrix, rs.memory.matrix, atol=0.0, rtol=0.0)


def test_v21_write_is_future_only_and_current_chunk_logits_ignore_completed_write():
    _, model = _build_pair(9402)
    model.eval()
    c = model.cfg.chunk_size
    first = torch.randint(0, model.cfg.vocab_size, (2, c))
    future_a = torch.randint(0, model.cfg.vocab_size, (2, c))
    future_b = torch.randint(0, model.cfg.vocab_size, (2, c))
    with torch.no_grad():
        no_write = model(first, hard=True, update_memory=False)["logits"]
        with_write = model(first, hard=True, update_memory=True)["logits"]
        out_a = model(torch.cat((first, future_a), dim=1), hard=True, update_memory=True)["logits"]
        out_b = model(torch.cat((first, future_b), dim=1), hard=True, update_memory=True)["logits"]
    torch.testing.assert_close(no_write, with_write, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_a[:, :c], out_b[:, :c], atol=0.0, rtol=0.0)


def test_v21_candidate_count_is_exactly_t_minus_one_and_gate_is_finite_bounded():
    _, model = _build_pair(9403)
    model.eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (3, model.cfg.chunk_size))
    with torch.no_grad():
        model(tokens, hard=True, update_memory=True)
    for stage in model.stages:
        assert stage.last_candidate_count == model.cfg.chunk_size - 1
        gate = stage.last_pair_gate
        assert gate is not None
        assert gate.shape == (tokens.size(0), model.cfg.chunk_size - 1, 1)
        assert torch.isfinite(gate).all()
        assert bool((gate >= 0.0).all() and (gate <= 1.0).all())

    one = torch.randint(0, model.cfg.vocab_size, (3, 1))
    with torch.no_grad():
        model(one, hard=True, update_memory=True)
    for stage in model.stages:
        assert stage.last_candidate_count == 0


def test_v21_differentiable_later_chunk_loss_reaches_gate_and_existing_qkvo():
    _, model = _build_pair(9404)
    model.train()
    c = model.cfg.chunk_size
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 2 * c))
    model.set_memory_pretraining_mode(True)
    try:
        out = model(tokens, hard=True, update_memory=True)
        logits = out["logits"]
        loss = F.cross_entropy(
            logits[:, c:-1].float().reshape(-1, model.cfg.vocab_size),
            tokens[:, c + 1:].reshape(-1),
        )
        loss.backward()
    finally:
        model.set_memory_pretraining_mode(False)

    stage = model.stages[0]
    params = {
        "pair_gate_weight": stage.pair_write_gate.weight,
        "pair_gate_bias": stage.pair_write_gate.bias,
        "q": stage.memory.q.weight,
        "k": stage.memory.k.weight,
        "v": stage.memory.v.weight,
        "out": stage.memory.out.weight,
    }
    for name, parameter in params.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0.0, name


def test_v21_deployment_write_mutates_no_base_parameters_and_memory_is_detached():
    _, model = _build_pair(9405)
    model.eval()
    model.set_memory_pretraining_mode(False)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 2 * model.cfg.chunk_size))
    versions = [p._version for p in model.parameters()]
    out = model(tokens, hard=True, update_memory=True)
    assert versions == [p._version for p in model.parameters()]
    for state in out["state"].stages:
        assert not state.memory.matrix.requires_grad


def test_v21_session_state_is_explicit_and_exactly_isolated():
    _, model = _build_pair(9406)
    model.eval()
    c = model.cfg.chunk_size
    fresh = torch.randint(0, model.cfg.vocab_size, (2, 2 * c))
    other = torch.randint(0, model.cfg.vocab_size, (2, 3 * c))
    with torch.no_grad():
        before = model(fresh, state=None, hard=True, update_memory=False)["logits"]
        _ = model(other, state=None, hard=True, update_memory=True)
        after = model(fresh, state=None, hard=True, update_memory=False)["logits"]
    torch.testing.assert_close(before, after, atol=0.0, rtol=0.0)


def test_v21_parameter_delta_is_one_pair_gate_per_stage():
    v19, v21 = _build_pair(9407)
    old = sum(p.numel() for p in v19.parameters())
    new = sum(p.numel() for p in v21.parameters())
    expected = v21.cfg.n_stages * (2 * v21.cfg.d_model + 1)
    assert new - old == expected
