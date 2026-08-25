from pathlib import Path

import torch

from tam_research import aera_v25_systems as sys25
from tam_research.aera_hardware_core_v23 import sparse_write_budget
from tam_research.aera_hardware_core_v25 import (
    FactorizedIdentityContextEpisodicMemoryStage,
)
from tam_research.aera_real_language import GRAD_ACCUM, SEQ_LEN, TOTAL_STEPS


def test_v25_systems_frozen_production_geometry_and_payload_budget():
    assert sys25.CHUNK_SIZE == 256
    assert sparse_write_budget(255) == 16
    assert sys25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH == 256
    assert sys25.MAX_PAYLOAD_EVENTS_PER_OPTIMIZER_STEP == 1024
    assert sys25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH * GRAD_ACCUM == 1024
    assert SEQ_LEN == 512
    assert TOTAL_STEPS == 512


def test_v25_systems_cpu_preflight_proves_16_of_255_gradients_and_isolation():
    result = sys25.cpu_preflight()
    assert result["gpu_authorized"] is False
    assert result["scientific_training_performed"] is False
    assert result["optimizer_steps_performed"] == 0
    assert result["checkpoint_written"] is False
    assert result["corpus_reader_used"] is False
    assert result["chunk_size"] == 256
    assert result["memory_dim"] == 50
    assert result["n_stages"] == 4
    assert result["production_candidates"] == 255
    assert result["production_selected_writes"] == 16
    assert result["vectorized_update_calls"] == 1
    assert result["state_bytes"] == 77_760
    assert abs(result["parameter_accounting"]["stored_parameter_delta_fraction"]) <= 0.05
    for value in result["gradient_l1"].values():
        assert value > 0.0


def test_v25_systems_payload_teaching_is_decoder_aligned_and_address_label_free():
    torch.manual_seed(12540)
    model = sys25.build_aera(torch.device("cpu"))
    assert isinstance(model.stages[0], FactorizedIdentityContextEpisodicMemoryStage)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, SEQ_LEN))
    model.zero_grad(set_to_none=True)
    terms = sys25.payload_teaching_terms(model, tokens, step=0, max_events=8)
    assert set(terms) == {
        "memory_payload_token_loss",
        "memory_payload_token_accuracy",
        "sampled_payload_events",
    }
    assert torch.isfinite(terms["memory_payload_token_loss"])
    assert float(terms["sampled_payload_events"]) == 8.0
    terms["memory_payload_token_loss"].backward()
    stage = model.stages[0]
    assert stage.memory.v.weight.grad is not None
    assert stage.memory.out.weight.grad is not None
    assert float(stage.memory.v.weight.grad.abs().sum()) > 0.0
    assert float(stage.memory.out.weight.grad.abs().sum()) > 0.0
    assert stage.memory.identity_proj.weight.grad is None
    assert stage.memory.context_proj.weight.grad is None
    assert stage.pair_write_gate.weight.grad is None


def test_v25_systems_payload_cap_refuses_over_budget_microbatch():
    torch.manual_seed(12541)
    model = sys25.build_aera(torch.device("cpu"))
    tokens = torch.randint(0, model.cfg.vocab_size, (1, SEQ_LEN))
    try:
        sys25.payload_teaching_terms(
            model,
            tokens,
            step=0,
            max_events=sys25.MAX_PAYLOAD_EVENTS_PER_MICROBATCH + 1,
        )
    except ValueError as exc:
        assert "exceeds frozen per-microbatch cap" in str(exc)
    else:
        raise AssertionError("v25 systems payload cap did not refuse oversized sample")


def test_v25_systems_benchmark_source_has_no_corpus_optimizer_or_checkpoint_path():
    source = Path("aera_v25_ficem_systems_l4.py").read_text()
    helper = Path("tam_research/aera_v25_systems.py").read_text()
    joined = source + "\n" + helper
    assert "TokenBin" not in joined
    assert "torch.optim" not in joined
    assert ".step()" not in joined
    assert "torch.save" not in joined
    assert "train.bin" not in joined
    assert "val.bin" not in joined
    assert "scientific_training_performed\": True" not in joined


def test_v25_systems_route_schedule_has_a_hard_sparse_phase():
    modes = [sys25.route_mode_for_step(step) for step in range(TOTAL_STEPS)]
    assert "straight_through" in modes
    assert "hard_sparse" in modes
