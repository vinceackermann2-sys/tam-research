import torch

import aera_v19_memory_necessity_cpu as v19diag
import aera_v20_memory_necessity_cpu as v20diag
from tam_research.aera_hardware_core_v20 import HardwareAwareAERATextLMV20


def test_v20_diagnostic_reuses_original_frozen_task_constants_and_thresholds():
    names = [
        "BATCH_SIZE",
        "TRAIN_STEPS",
        "LEARNING_RATE",
        "TASK_VALIDITY_MIN",
        "FULL_ACCURACY_MIN",
        "FULL_OVER_STREAM_MIN",
        "SAME_CHECKPOINT_MEMORY_DROP_MIN",
        "OVERWRITE_ACCURACY_MIN",
        "STALE_ERROR_MAX",
        "FRESH_SESSION_CHANCE_TOLERANCE",
        "N_VALUES",
    ]
    for name in names:
        assert getattr(v20diag, name) == getattr(v19diag, name), name


def test_v20_diagnostic_builds_factorized_write_model_and_forces_routes_open():
    model = v20diag.build_model(9301)
    assert isinstance(model, HardwareAwareAERATextLMV20)
    for router in model.stage_routers:
        assert not any(p.requires_grad for p in router.parameters())
        torch.testing.assert_close(
            router.proj.bias.detach(),
            torch.full_like(router.proj.bias.detach(), 12.0),
            atol=0.0,
            rtol=0.0,
        )


def test_v20_full_and_stream_only_start_bit_exact():
    full = v20diag.build_model(v20diag.SEED)
    stream = v20diag.build_model(v20diag.SEED)
    left = full.state_dict()
    right = stream.state_dict()
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0.0, rtol=0.0)


def test_v20_short_training_is_ordinary_query_ce_without_synthetic_auxiliary():
    full, stream, training = v20diag.train_pair(steps=1)
    assert isinstance(full, HardwareAwareAERATextLMV20)
    assert isinstance(stream, HardwareAwareAERATextLMV20)
    assert len(training["history"]) == 1
    row = training["history"][0]
    assert torch.isfinite(torch.tensor(row["full_loss"]))
    assert torch.isfinite(torch.tensor(row["stream_only_loss"]))
    # The v20 diagnostic is intentionally a thin reuse of the #278 query CE path;
    # no diagnostic address/value supervision is imported from the #281 experiment.
    assert "address" not in row
    assert "value" not in row
