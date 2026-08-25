from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn.functional as F

from tam_research import aera_v25_post8471_triage as base
from tam_research import aera_v25_post8471_triage_repair1 as repair


def test_repair_protocol_preserves_frozen_issue369_sampling() -> None:
    protocol = repair.repair_protocol()
    assert protocol["research_issue"] == 369
    assert protocol["repair_issue"] == 372
    assert protocol["source_failed_trigger"] == 371
    assert protocol["source_checkpoint_seed"] == 8471
    assert protocol["diagnostic_sampling_seed"] == 138471
    assert protocol["memory_batches"] == 64
    assert protocol["memory_batch_size"] == 8
    assert protocol["adaptivity_batches"] == 256
    assert protocol["adaptivity_batch_size"] == 8
    assert protocol["read_alphas"] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert protocol["bootstrap_resamples"] == 2000
    assert protocol["systems_batch_sizes"] == [8, 64]
    assert protocol["systems_warmup_calls"] == 3
    assert protocol["systems_timed_calls_per_round"] == 20
    assert protocol["systems_rounds"] == 5
    assert protocol["loss_time_slice_tokens"] == 32
    assert protocol["training_performed"] is False
    assert protocol["scientific_protocol_changed"] is False


def test_sliced_token_nll_matches_original_vectorized_definition() -> None:
    generator = torch.Generator().manual_seed(37201)
    logits = torch.randn(3, 65, 29, generator=generator)
    y = torch.randint(0, 29, (3, 65), generator=generator)
    expected = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).reshape(3, 65)
    actual = repair.sliced_token_nll(logits, y)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.device.type == "cpu"
    assert actual.shape == (3, 65)


def test_sliced_second_range_preserves_example_and_token_order() -> None:
    generator = torch.Generator().manual_seed(37202)
    logits = torch.randn(4, 73, 17, generator=generator)
    y = torch.randint(0, 17, (4, 73), generator=generator)
    expected_tokens = F.cross_entropy(
        logits[:, 32:].float().reshape(-1, logits.size(-1)),
        y[:, 32:].reshape(-1),
        reduction="none",
    ).reshape(4, 41)
    actual_tokens = repair.sliced_token_nll(logits, y, start=32)
    actual_examples = repair.sliced_per_example_nll(logits, y, start=32)
    torch.testing.assert_close(actual_tokens, expected_tokens, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual_examples, expected_tokens.mean(dim=1), rtol=1e-6, atol=1e-6
    )


def test_sliced_chunk_difficulty_matches_original_two_chunk_definition() -> None:
    generator = torch.Generator().manual_seed(37203)
    logits = torch.randn(2, base.SEQ_LEN, 19, generator=generator)
    y = torch.randint(0, 19, (2, base.SEQ_LEN), generator=generator)
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).reshape(2, base.SEQ_LEN)
    expected = torch.stack(
        (
            token_loss[:, : base.CHUNK_SIZE].mean(dim=1),
            token_loss[:, base.CHUNK_SIZE :].mean(dim=1),
        ),
        dim=1,
    )
    actual = repair.sliced_chunk_mean_nll(logits, y)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == (2, 2)


def test_repair_runner_builds_models_before_inference_measurement_scope() -> None:
    source = inspect.getsource(repair.run_checkpoint_triage_repair1)
    load_index = source.index("base._load_models")
    inference_index = source.index("with torch.inference_mode()")
    assert load_index < inference_index
    assert "base._condition_losses = bounded_condition_losses" in source
    assert "finally:" in source
    assert "base._condition_losses = original_condition_losses" in source


def test_repair_module_contains_no_training_or_checkpoint_write_api() -> None:
    source = inspect.getsource(repair)
    forbidden = (
        ".backward(",
        "torch.optim",
        "optimizer.step",
        "save_checkpoint",
        "torch.save(",
        "resume_training",
    )
    for token in forbidden:
        assert token not in source


def test_repair_workflow_contract_is_distinct_and_source_guarded() -> None:
    workflow = Path(".github/workflows/aera-v25-post8471-triage-repair1.yml").read_text()
    assert "[aera-v25-post8471-triage-repair1]" in workflow
    assert "issue view 371" in workflow
    assert "checkpoint triage implementation/runtime failure" in workflow
    assert "AERA-v25 post-seed8471 checkpoint triage result" in workflow
    assert "github.run_attempt != 1" in workflow
    assert "run-attempt=1" in workflow
    assert "`run-attempt=1`" not in workflow
    assert "AERA_V25_POST8471_TRIAGE_REPAIR1_RESULT_JSON=" in workflow


def test_repair_modal_launcher_uses_unique_result_and_allocator_hygiene() -> None:
    launcher = Path("modal_aera_v25_post8471_triage_repair1_app.py").read_text()
    assert "v25-post8471-issue369-repair1/result.json" in launcher
    assert "v25-post8471-issue369/result.json" in launcher
    assert '"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"' in launcher
    assert "MAX_GPU_SECONDS = 900" in launcher
