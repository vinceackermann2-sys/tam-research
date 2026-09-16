from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tam_research.chm_v1_100m_stage_c_execution import (
    CONTROL_ISSUE,
    PHASE,
    RESULT_ROOT,
    TRIGGER_TITLE,
    _exact_nearest_positions,
    build_start_plan,
    build_training_start_plan,
    build_validation_start_plan,
    eiem_exact_flat_two_chunk_logits,
    gather_batch_from_source,
    start_plan_sha256,
    validate_execution_contract,
)
from tam_research.chm_v1_100m_stage_c_run_control_prep import (
    GRAD_ACCUM,
    MICRO_BATCH,
    OPTIMIZER_STEPS_PER_MODEL,
    SCIENTIFIC_SEED,
    TRAIN_STREAM_GENERATOR_SEED,
)
from tam_research.chm_v1_exact_index import ExactEpisodicIndex
from tam_research.chm_v1_small_lm import EpisodicState, LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden
from tam_research.models import ModelConfig, ResearchLM

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "modal_chm_v1_100m_stage_c_990_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "modal-chm-v1-100m-stage-c-990-v1.yml"


def test_execution_contract_is_frozen_but_does_not_self_authorize() -> None:
    manifest = validate_execution_contract()
    assert CONTROL_ISSUE == 990
    assert SCIENTIFIC_SEED == 977001
    assert PHASE == "chm-v1-100m-stage-c-990-v1"
    assert TRIGGER_TITLE == "[modal-chm-v1-100m-stage-c-988-seed-977001-v1]"
    assert RESULT_ROOT == "/vol/chm-v1/100m-stage-c/issue-988/seed-977001-v1"
    assert manifest["trigger_authorized_by_module"] is False
    assert manifest["gpu_allocation_authorized_by_module"] is False
    assert manifest["scientific_seed_consumed_by_module"] is False


def test_training_plan_is_exact_deterministic_pair_stream() -> None:
    plan_a = build_training_start_plan(2_000_000_000)
    plan_b = build_training_start_plan(2_000_000_000)
    assert tuple(plan_a.shape) == (
        OPTIMIZER_STEPS_PER_MODEL,
        GRAD_ACCUM,
        MICRO_BATCH,
    )
    assert plan_a.numel() == 32_768
    assert TRAIN_STREAM_GENERATOR_SEED == 987001
    assert torch.equal(plan_a, plan_b)
    assert start_plan_sha256(plan_a) == start_plan_sha256(plan_b)
    assert int(plan_a.min()) >= 0
    assert int(plan_a.max()) < 2_000_000_000 - 1024 - 1


def test_validation_plan_is_frozen_and_distinct_from_training_plan() -> None:
    val_a = build_validation_start_plan(5_000_000)
    val_b = build_validation_start_plan(5_000_000)
    assert tuple(val_a.shape) == (128, 1, 8)
    assert torch.equal(val_a, val_b)
    assert start_plan_sha256(val_a) == start_plan_sha256(val_b)
    train_prefix = build_start_plan(
        shard_tokens=5_000_000,
        seq_len=1024,
        steps=128,
        batches_per_step=1,
        batch_size=8,
        seed=TRAIN_STREAM_GENERATOR_SEED,
    )
    assert start_plan_sha256(val_a) != start_plan_sha256(train_prefix)


def test_start_plan_refuses_illegal_geometry() -> None:
    with pytest.raises(ValueError):
        build_start_plan(
            shard_tokens=1024,
            seq_len=1024,
            steps=1,
            batches_per_step=1,
            batch_size=1,
            seed=1,
        )
    with pytest.raises(ValueError):
        build_start_plan(
            shard_tokens=4096,
            seq_len=0,
            steps=1,
            batches_per_step=1,
            batch_size=1,
            seed=1,
        )


def test_explicit_gather_matches_frozen_offsets() -> None:
    source = torch.arange(80, dtype=torch.int32)
    starts = torch.tensor([0, 7, 20], dtype=torch.int64)
    x, y = gather_batch_from_source(source, starts, seq_len=5)
    assert x.tolist() == [
        [0, 1, 2, 3, 4],
        [7, 8, 9, 10, 11],
        [20, 21, 22, 23, 24],
    ]
    assert y.tolist() == [
        [1, 2, 3, 4, 5],
        [8, 9, 10, 11, 12],
        [21, 22, 23, 24, 25],
    ]


def test_vectorized_exact_nearest_matches_flat_reference_and_tie_rule() -> None:
    generator = torch.Generator(device="cpu").manual_seed(99001)
    keys = torch.randn(2, 17, 8, generator=generator, dtype=torch.float32)
    queries = torch.randn(2, 11, 8, generator=generator, dtype=torch.float32)
    positions = _exact_nearest_positions(keys, queries, query_block=3).cpu()

    for batch in range(keys.shape[0]):
        index = ExactEpisodicIndex(
            keys[batch].numpy(),
            np.arange(keys.shape[1], dtype=np.int64),
            leaf_size=4,
        )
        expected = [index.flat_search(query.numpy()).position for query in queries[batch]]
        assert positions[batch].tolist() == expected

    tied_keys = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]])
    tied_query = torch.tensor([[[1.0, 0.0]]])
    tied = _exact_nearest_positions(tied_keys, tied_query)
    assert tied.item() == 0


class _TinyEIEM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = ModelConfig(
            vocab_size=64,
            d_model=16,
            n_layers=1,
            n_heads=4,
            max_seq_len=1024,
            ff_mult=2,
            architecture="transformer",
        )
        self.backbone = ResearchLM(cfg)
        self.query_address = nn.Linear(cfg.d_model, 8, bias=False)
        self.key_address = nn.Linear(cfg.d_model, 8, bias=False)
        self.memory_gate_logit = nn.Parameter(torch.full((cfg.d_model,), -4.0))

    def query_for(self, representation: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.query_address(representation), dim=-1)

    def key_for(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.key_address(hidden), dim=-1)

    def _integrate(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return hidden + torch.sigmoid(self.memory_gate_logit).to(hidden.dtype) * memory


@torch.no_grad()
def _reference_two_chunk_logits(model: _TinyEIEM, tokens: torch.Tensor) -> torch.Tensor:
    assert tokens.shape[0] == 1
    first = tokens[:, :LOCAL_WINDOW]
    second = tokens[:, LOCAL_WINDOW:]
    first_hidden = _hidden(model.backbone, first)
    first_logits = model.backbone.lm_head(first_hidden)
    state = EpisodicState("stage-c-990-test")
    state.write(model.key_for(first_hidden)[0], first_hidden[0])

    second_hidden = _hidden(model.backbone, second)
    fused = second_hidden.clone()
    for token_index in range(LOCAL_WINDOW):
        query_state = second_hidden[0, token_index]
        for _ in range(RETRIEVAL_HOPS):
            query = model.query_for(query_state)
            value, _, _, _, _, _ = state.retrieve(
                query,
                mode="flat",
                verify_indexed_exactness=False,
            )
            query_state = model._integrate(query_state, value)
        fused[0, token_index] = query_state
    return torch.cat((first_logits, model.backbone.lm_head(fused)), dim=1)


def test_vectorized_eiem_validation_matches_exact_flat_reference() -> None:
    torch.manual_seed(99002)
    model = _TinyEIEM().eval()
    tokens = torch.randint(0, 64, (1, 1024), generator=torch.Generator().manual_seed(99003))
    expected = _reference_two_chunk_logits(model, tokens)
    actual = eiem_exact_flat_two_chunk_logits(model, tokens)
    assert actual.shape == expected.shape == (1, 1024, 64)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_runner_is_one_shot_and_has_no_checkpoint_resume_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "SCIENTIFIC_SEED = 977_001" in source
    assert "977_201" not in source
    assert "retries=0" in source
    assert '"ATTEMPT_CONSUMED.json"' in source
    assert '"ATTEMPT_FAILURE.json"' in source
    assert '"RESULT.json"' in source
    assert "checkpoint_resume_authorized" in source
    assert "torch.load(" not in source
    assert "resume_from" not in source
    assert "automatic_retry_authorized" in source
    assert "stage_d_automatically_authorized" in source


def test_workflow_requires_final_authority_and_has_no_manual_dispatch() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "issues:" in source
    assert "types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert TRIGGER_TITLE in source
    assert "CHM_V1_990_FINAL_LAUNCHER_AUTHORITY_V1" in source
    assert "github.run_attempt" in source or "GITHUB_RUN_ATTEMPT" in source
    assert "run_attempt" in source
    assert "modal billing rates --json" in source
    assert "43200" in source
    assert "25.00" in source
    assert "--phase preflight" in source
    assert "--phase reserve" in source
    assert "--phase run" in source
    assert "--phase inspect" in source
    assert "scientific_seed=977001" in source
    assert "trigger_authorized=true" in source
    assert "automatic_retry_authorized=false" in source
    assert "checkpoint_resume_authorized=false" in source
