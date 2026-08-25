import torch

from tam_research.aera_hardware_core_v24 import causal_contextualize


def test_v24_context_prefix_is_bit_exact_under_future_perturbation():
    g = torch.Generator().manual_seed(9520)
    x = torch.randn(3, 32, 20, generator=g)
    y = x.clone()
    y[:, 17:] = torch.randn(3, 15, 20, generator=g) * 1000.0
    cx = causal_contextualize(x)
    cy = causal_contextualize(y)
    torch.testing.assert_close(cx[:, :17], cy[:, :17], atol=0.0, rtol=0.0)
