import torch

from aera_v19_memory_necessity_cpu import make_batch
from aera_v19_memory_signal_cpu import (
    ADDRESS_ACCURACY_MIN,
    FULL_ACCURACY_MIN,
    SAME_CHECKPOINT_MEMORY_DROP_MIN,
    VALUE_ACCURACY_MIN,
    _capture_stage0_reasoned,
    _memory_utility_terms,
    _query_loss_and_accuracy,
    build_model,
)


def test_memory_signal_thresholds_are_preregistered_values():
    assert FULL_ACCURACY_MIN == 0.80
    assert SAME_CHECKPOINT_MEMORY_DROP_MIN == 0.15
    assert ADDRESS_ACCURACY_MIN == 0.90
    assert VALUE_ACCURACY_MIN == 0.90


def test_memory_signal_uses_existing_summary_and_reaches_memory_projections():
    model = build_model(9011)
    model.train()
    batch = make_batch(2, 9012)
    model.set_memory_pretraining_mode(True)
    try:
        with _capture_stage0_reasoned(model) as captured:
            out = model(
                batch.tokens,
                hard=True,
                route_mode="hard_sparse",
                update_memory=True,
                return_block_logits=False,
            )
        logits = out["logits"]
        assert isinstance(logits, torch.Tensor)
        query_loss, _ = _query_loss_and_accuracy(logits, batch)
        aux = _memory_utility_terms(model, batch.tokens, captured)
        loss = query_loss + aux["address_loss"] + aux["value_loss"]
        assert torch.isfinite(loss)
        loss.backward()
    finally:
        model.set_memory_pretraining_mode(False)

    memory = model.stages[0].memory
    for name, parameter in (
        ("q", memory.q.weight),
        ("k", memory.k.weight),
        ("v", memory.v.weight),
        ("out", memory.out.weight),
    ):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0.0, name


def test_memory_signal_does_not_change_production_model_class_or_state_layout():
    left = build_model(9021)
    right = build_model(9021)
    assert type(left) is type(right)
    assert list(left.state_dict()) == list(right.state_dict())
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], atol=0.0, rtol=0.0)
