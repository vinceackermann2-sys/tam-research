from __future__ import annotations

import torch

from tam_research.cpw_v1.model import CPWV1Config, CPWV1ResearchLM
from tam_research.cpw_v1.protocol import EXPECTED_CPW_V1_PARAMETERS
from tam_research.cpw_v1_fast.model import (
    TritonScanWorldPredictor,
    convert_cpw_v1_to_triton_scan,
    fast_backend_status,
)
from tam_research.models import parameter_count


def test_conversion_preserves_parameters_names_and_objects() -> None:
    torch.manual_seed(1249)
    model = CPWV1ResearchLM(CPWV1Config(n_layers=2))
    before = {name: p for name, p in model.named_parameters()}
    count = parameter_count(model)
    convert_cpw_v1_to_triton_scan(model)
    after = {name: p for name, p in model.named_parameters()}
    assert count == parameter_count(model)
    assert before.keys() == after.keys()
    assert all(before[name] is after[name] for name in before)
    assert all(
        isinstance(block.mixer.world, TritonScanWorldPredictor)
        for block in model.blocks
    )


def test_full_frozen_geometry_parameter_count_is_unchanged() -> None:
    model = CPWV1ResearchLM()
    convert_cpw_v1_to_triton_scan(model)
    assert parameter_count(model) == EXPECTED_CPW_V1_PARAMETERS


def test_cpu_forward_and_gradients_match_frozen_model() -> None:
    torch.manual_seed(1249001)
    base = CPWV1ResearchLM(CPWV1Config(n_layers=2))
    fast = CPWV1ResearchLM(CPWV1Config(n_layers=2))
    fast.load_state_dict(base.state_dict())
    convert_cpw_v1_to_triton_scan(fast)

    tokens = torch.randint(0, base.cfg.vocab_size, (1, 16))
    y0 = base(tokens)
    y1 = fast(tokens)
    torch.testing.assert_close(y0, y1, rtol=2e-5, atol=2e-6)

    loss0 = y0.float().square().mean()
    loss1 = y1.float().square().mean()
    loss0.backward()
    loss1.backward()
    g0 = dict(base.named_parameters())
    g1 = dict(fast.named_parameters())
    for name in (
        "blocks.0.mixer.world.candidate.weight",
        "blocks.0.mixer.world.keep.weight",
        "blocks.0.mixer.world.out.weight",
    ):
        torch.testing.assert_close(
            g0[name].grad,
            g1[name].grad,
            rtol=2e-4,
            atol=2e-6,
        )


def test_future_token_causality_survives_conversion() -> None:
    torch.manual_seed(42)
    model = CPWV1ResearchLM(CPWV1Config(n_layers=2)).eval()
    convert_cpw_v1_to_triton_scan(model)
    a = torch.randint(0, model.cfg.vocab_size, (1, 24))
    b = a.clone()
    b[:, 16:] = torch.randint(0, model.cfg.vocab_size, (1, 8))
    with torch.no_grad():
        ya = model(a)
        yb = model(b)
    torch.testing.assert_close(ya[:, :16], yb[:, :16], rtol=0, atol=2e-5)


def test_backend_status_is_systems_only() -> None:
    status = fast_backend_status()
    assert status["scientific_equations_changed"] is False
    assert status["parameter_count_delta"] == 0
