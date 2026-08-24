import torch

import aera_v19_memory_necessity_cpu as v19diag
import aera_v21_memory_necessity_cpu as v21diag
from tam_research.aera_hardware_core_v21 import HardwareAwareAERATextLMV21


def test_v21_diagnostic_reuses_original_frozen_constants_and_thresholds():
    for name in [
        "BATCH_SIZE", "TRAIN_STEPS", "LEARNING_RATE", "TASK_VALIDITY_MIN",
        "FULL_ACCURACY_MIN", "FULL_OVER_STREAM_MIN",
        "SAME_CHECKPOINT_MEMORY_DROP_MIN", "OVERWRITE_ACCURACY_MIN",
        "STALE_ERROR_MAX", "FRESH_SESSION_CHANCE_TOLERANCE", "N_VALUES",
    ]:
        assert getattr(v21diag, name) == getattr(v19diag, name), name


def test_v21_diagnostic_builds_event_pair_model_with_routes_forced_open():
    model = v21diag.build_model(9501)
    assert isinstance(model, HardwareAwareAERATextLMV21)
    for router in model.stage_routers:
        assert not any(p.requires_grad for p in router.parameters())
        torch.testing.assert_close(router.proj.bias.detach(), torch.full_like(router.proj.bias.detach(), 12.0), atol=0.0, rtol=0.0)


def test_v21_short_training_is_plain_delayed_query_ce_without_auxiliary_terms():
    full, stream, training = v21diag.train_pair(steps=1)
    assert isinstance(full, HardwareAwareAERATextLMV21)
    assert isinstance(stream, HardwareAwareAERATextLMV21)
    assert len(training["history"]) == 1
    row = training["history"][0]
    assert torch.isfinite(torch.tensor(row["full_loss"]))
    assert torch.isfinite(torch.tensor(row["stream_only_loss"]))
    assert set(row) == {"step", "full_loss", "stream_only_loss", "full_accuracy", "stream_only_accuracy"}
