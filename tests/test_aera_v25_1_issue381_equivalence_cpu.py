import ast
import inspect
import textwrap

import torch

from aera_v19_memory_necessity_cpu import (
    EVAL_SEED,
    _force_all_stages_run,
    diagnostic_config,
    make_batch,
)
from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState
from tam_research.aera_hardware_core_v25 import HardwareAwareAERATextLMV25
from tam_research.aera_hardware_core_v25_1 import (
    ExecutionEquivalentFICEMStage,
    HardwareAwareAERATextLMV251,
)

ISSUE381_RTOL = 1e-6
ISSUE381_ATOL = 1e-6


def _models(seed: int, *, chunk_size: int | None = None):
    cfg = diagnostic_config()
    if chunk_size is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "chunk_size": chunk_size})
    torch.manual_seed(seed)
    baseline = HardwareAwareAERATextLMV25(cfg)
    torch.manual_seed(seed + 91)
    candidate = HardwareAwareAERATextLMV251(cfg)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    candidate.eval()
    return baseline, candidate


def _assert_epi_issue381(a: ContextualEpisodicMemoryState, b: ContextualEpisodicMemoryState):
    torch.testing.assert_close(a.keys, b.keys, rtol=ISSUE381_RTOL, atol=ISSUE381_ATOL)
    torch.testing.assert_close(a.values, b.values, rtol=ISSUE381_RTOL, atol=ISSUE381_ATOL)
    torch.testing.assert_close(
        a.strengths, b.strengths, rtol=ISSUE381_RTOL, atol=ISSUE381_ATOL
    )
    assert torch.equal(a.valid, b.valid)


def test_issue381_strict_two_chunk_model_output_state_and_routes_match_v25():
    baseline, candidate = _models(138101)
    _force_all_stages_run(baseline)
    _force_all_stages_run(candidate)
    tokens = make_batch(2, EVAL_SEED + 138101).tokens[:, : 2 * baseline.cfg.chunk_size]

    with torch.no_grad():
        old = baseline(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
            return_block_logits=False,
        )
        new = candidate(
            tokens,
            hard=True,
            route_mode="hard_sparse",
            update_memory=True,
            return_block_logits=False,
        )

    torch.testing.assert_close(
        old["logits"], new["logits"], rtol=ISSUE381_RTOL, atol=ISSUE381_ATOL
    )
    for old_stage_state, new_stage_state in zip(old["state"].stages, new["state"].stages):
        torch.testing.assert_close(
            old_stage_state.stream,
            new_stage_state.stream,
            rtol=ISSUE381_RTOL,
            atol=ISSUE381_ATOL,
        )
        _assert_epi_issue381(old_stage_state.memory, new_stage_state.memory)

    for old_chunk, new_chunk in zip(old["stage_routes"], new["stage_routes"]):
        for old_route, new_route in zip(old_chunk, new_chunk):
            assert torch.equal(old_route["stage_route_gate"], new_route["stage_route_gate"])
            torch.testing.assert_close(
                old_route["stage_route_probability"],
                new_route["stage_route_probability"],
                rtol=ISSUE381_RTOL,
                atol=ISSUE381_ATOL,
            )
            assert old_route["executed_fraction"] == new_route["executed_fraction"]

    for old_stage, new_stage in zip(baseline.stages, candidate.stages):
        assert torch.equal(old_stage.last_selected_indices, new_stage.last_selected_indices)
        assert old_stage.last_selected_count == new_stage.last_selected_count
        assert old_stage.last_vectorized_update_calls == new_stage.last_vectorized_update_calls == 1
        assert new_stage._runtime_factor_cache is None


def test_issue381_strict_ficem_nonempty_read_matches_v25():
    baseline, candidate = _models(138102)
    old_memory = baseline.stages[0].memory
    new_memory = candidate.stages[0].memory
    b, t, d = 2, baseline.cfg.chunk_size, baseline.cfg.d_model
    g = torch.Generator().manual_seed(138102)
    write_identity = torch.randn(b, 2, d, generator=g)
    write_context = torch.randn(b, 2, d, generator=g)
    payload = torch.randn(b, 2, d, generator=g)
    strength = torch.ones(b, 2, 1)
    state = old_memory.empty_state(b, torch.device("cpu"), torch.float32)
    state = old_memory.update_block(
        write_identity, write_context, payload, strength, state
    )
    identity = torch.randn(b, t, d, generator=g)
    context = torch.randn(b, t, d, generator=g)

    with torch.no_grad():
        old = old_memory.read(identity, context, state)
        new = new_memory.read(identity, context, state)

    torch.testing.assert_close(old, new, rtol=ISSUE381_RTOL, atol=ISSUE381_ATOL)
    assert new_memory.empty_read_fastpath_calls == 0


def test_issue381_execution_guards_keep_population_and_vectorized_paths():
    # V24/V25 already routes each executed optional-stage population in one batched
    # stage invocation. V25.1 must not introduce a per-example fallback while adding
    # its stage0 direct-dispatch and FICEM execution repairs.
    route_source = inspect.getsource(HardwareAwareAERATextLMV251._route_one_stage)
    stage_source = inspect.getsource(ExecutionEquivalentFICEMStage.forward_chunk)
    stage_tree = ast.parse(textwrap.dedent(stage_source))
    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(stage_tree))
    assert "stage.forward_chunk(" in route_source
    assert "super()._route_one_stage(" in route_source
    assert "self.memory.update_block(" in stage_source
    assert "last_vectorized_update_calls = 1" in stage_source


def test_issue381_production_shape_remains_16_255_1_under_strict_state_equivalence():
    baseline, candidate = _models(138103, chunk_size=256)
    old_stage = baseline.stages[0]
    new_stage = candidate.stages[0]
    g = torch.Generator().manual_seed(138103)
    events = torch.randn(1, 256, baseline.cfg.d_model, generator=g)

    with torch.no_grad():
        old_out, old_state, _ = old_stage.forward_chunk(
            events, None, hard=True, update_memory=True
        )
        new_out, new_state, _ = new_stage.forward_chunk(
            events, None, hard=True, update_memory=True
        )

    torch.testing.assert_close(
        old_out, new_out, rtol=ISSUE381_RTOL, atol=ISSUE381_ATOL
    )
    torch.testing.assert_close(
        old_state.stream,
        new_state.stream,
        rtol=ISSUE381_RTOL,
        atol=ISSUE381_ATOL,
    )
    _assert_epi_issue381(old_state.memory, new_state.memory)
    assert old_stage.last_candidate_count == new_stage.last_candidate_count == 255
    assert old_stage.last_selected_count == new_stage.last_selected_count == 16
    assert old_stage.last_vectorized_update_calls == new_stage.last_vectorized_update_calls == 1
    assert torch.equal(old_stage.last_selected_indices, new_stage.last_selected_indices)
