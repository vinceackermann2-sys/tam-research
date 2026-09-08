from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from tam_research import aera_issue748_memory_capability_seed1_harness_base as base
from tam_research import aera_issue770_integrated_event_memory_cpu as integ
from tam_research.aera import AERAState, FastMemoryState
from tam_research.aera_hardware_core import HardwareAERAState
from tam_research.aera_hardware_core_v18 import PretrainableDeltaFastMemory


def _pad(items: list[int], fill: int = 5000) -> torch.Tensor:
    if len(items) > 16:
        raise ValueError("probe chunk too long")
    return torch.tensor([items + [fill] * (16 - len(items))], dtype=torch.long)


def _forward(model, chunk: torch.Tensor, *, state=None, update_memory=True):
    return model.forward_answer_tokens(
        chunk,
        torch.tensor([min(15, chunk.size(1) - 1)]),
        torch.zeros(1, chunk.size(1) // 16, dtype=torch.bool),
        route_mode="straight_through",
        intrinsic_hard=False,
        update_memory=update_memory,
        state=state,
    )


def _query_trace(result, stage: int = 0):
    events = result["stage_routes"][0][stage]["memory_events"][0]
    return next(event for event in events if event["kind"] == "query")


def _identity_memory() -> PretrainableDeltaFastMemory:
    memory = PretrainableDeltaFastMemory(4, 4, lr=1.0, decay=1.0)
    eye = torch.eye(4)
    with torch.no_grad():
        for layer in (memory.q, memory.k, memory.v, memory.out):
            layer.weight.copy_(eye)
    return memory


def test_authority_is_cpu_only_and_scientific_seeds_stay_locked() -> None:
    snapshot = integ.authority_snapshot()
    assert snapshot["research_issue"] == 770
    assert snapshot["cpu_test_seed"] == 770_999
    assert snapshot["consumed_scientific_seed"] == 17_641
    for key in (
        "gpu_authorized", "modal_authorized", "scientific_training_authorized",
        "scientific_evaluation_authorized", "scientific_seed_authorized",
        "seeds_2_3_authorized", "scientific_result_authorized",
        "scientific_checkpoint_authorized", "systems_optimization_authorized",
        "architecture_freeze_authorized", "scaling_authorized", "breakthrough_proven",
    ):
        assert snapshot[key] is False
    for seed in base.SCIENTIFIC_MODEL_SEEDS:
        try:
            integ.build_cpu_model("B_backbone_plus_memory", seed=seed)
        except PermissionError:
            pass
        else:
            raise AssertionError("scientific seed unexpectedly accepted")


def test_ft1_actual_path_query_key_dependence_and_legacy_control() -> None:
    cfg = integ.cpu_probe_config()
    legacy = base.CapabilityAblationLM("B_backbone_plus_memory", cfg)
    seen: list[torch.Tensor] = []
    original = legacy.stages[0].memory.read
    def record(x, state):
        seen.append(x.detach().clone())
        return original(x, state)
    legacy.stages[0].memory.read = record
    a = _pad([99, integ.QUERY, 1000, integ.ANSWER])
    b = _pad([99, integ.QUERY, 1001, integ.ANSWER])
    _forward(legacy, a, update_memory=False)
    _forward(legacy, b, update_memory=False)
    assert len(seen) >= 2
    assert torch.equal(seen[-2], seen[-1])

    model = integ.build_cpu_model("B_backbone_plus_memory", cfg=cfg)
    ra = _query_trace(_forward(model, a, update_memory=False))
    rb = _query_trace(_forward(model, b, update_memory=False))
    assert not torch.equal(ra["read_input"], rb["read_input"])


def test_ft2_query_read_cannot_depend_on_future_tokens() -> None:
    model = integ.build_cpu_model("B_backbone_plus_memory")
    a = _pad([77, integ.QUERY, 1000, integ.ANSWER, 5001, 5002], fill=5003)
    b = _pad([77, integ.QUERY, 1000, integ.ANSWER, 5901, 5902], fill=5903)
    template = model.empty_state(a)
    torch.manual_seed(770_222)
    filled = HardwareAERAState([
        AERAState(s.stream.clone(), FastMemoryState(torch.randn_like(s.memory.matrix)))
        for s in template.stages
    ])
    qa = _query_trace(_forward(model, a, state=filled, update_memory=False))
    qb = _query_trace(_forward(model, b, state=filled, update_memory=False))
    assert torch.equal(qa["read_input"], qb["read_input"])
    assert torch.equal(qa["recall"], qb["recall"])


def test_ft3_event_write_is_invariant_to_trailing_noise() -> None:
    model = integ.build_cpu_model("B_backbone_plus_memory")
    a = _pad([integ.WRITE, 1000, 2000, integ.SEP, 5001, 5002], fill=5003)
    b = _pad([integ.WRITE, 1000, 2000, integ.SEP, 5901, 5902], fill=5903)
    sa = _forward(model, a)["state"]
    sb = _forward(model, b)["state"]
    for left, right in zip(sa.stages, sb.stages):
        assert torch.equal(left.memory.matrix, right.memory.matrix)


def test_ft4_update_moves_same_key_to_new_value_on_production_memory() -> None:
    memory = _identity_memory()
    state = memory.empty_state(1, torch.device("cpu"), torch.float32)
    key = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    old = torch.tensor([[[0.0, 0.0, 0.5, 0.0]]])
    new = torch.tensor([[[0.0, 0.0, -0.5, 0.0]]])
    strength = torch.ones(1, 1, 1)
    state = integ.event_aligned_delta_update(memory, key, old, strength, state)
    old_read = memory.read(key, state)
    state = integ.event_aligned_delta_update(memory, key, new, strength, state)
    new_read = memory.read(key, state)
    target = torch.tanh(new)
    assert torch.linalg.vector_norm(new_read - target) < torch.linalg.vector_norm(old_read - target)


def test_ft5_two_production_memory_keys_remain_separable() -> None:
    memory = _identity_memory()
    state = memory.empty_state(1, torch.device("cpu"), torch.float32)
    k0 = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    k1 = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    v0 = torch.tensor([[[0.0, 0.0, 0.5, 0.0]]])
    v1 = torch.tensor([[[0.0, 0.0, 0.0, 0.5]]])
    one = torch.ones(1, 1, 1)
    state = integ.event_aligned_delta_update(memory, k0, v0, one, state)
    state = integ.event_aligned_delta_update(memory, k1, v1, one, state)
    r0, r1 = memory.read(k0, state), memory.read(k1, state)
    assert not torch.equal(r0, r1)
    assert torch.allclose(r0, torch.tanh(v0), atol=1e-6)
    assert torch.allclose(r1, torch.tanh(v1), atol=1e-6)


def test_ft6_reset_before_query_clears_old_session_memory() -> None:
    model = integ.build_cpu_model("B_backbone_plus_memory")
    chunk = _pad([integ.RESET, integ.SEP, integ.QUERY, 1000, integ.ANSWER])
    template = model.empty_state(chunk)
    filled = HardwareAERAState([
        AERAState(s.stream.clone(), FastMemoryState(torch.ones_like(s.memory.matrix)))
        for s in template.stages
    ])
    query = _query_trace(_forward(model, chunk, state=filled, update_memory=False))
    assert torch.equal(query["recall"], torch.zeros_like(query["recall"]))


def test_ft7_integrated_v18_pretraining_path_is_differentiable() -> None:
    model = integ.build_cpu_model("B_backbone_plus_memory")
    model.set_memory_pretraining_mode(True)
    chunk = _pad([
        integ.WRITE, 1000, 2000, integ.SEP,
        integ.QUERY, 1000, integ.ANSWER, 2000,
    ])
    result = _forward(model, chunk, update_memory=True)
    loss = result["answer_logits"].square().mean()
    loss.backward()
    memory = model.stages[0].memory
    for layer in (memory.q, memory.k, memory.v, memory.out):
        grad = layer.weight.grad
        assert grad is not None
        assert bool(torch.isfinite(grad).all())
        assert float(grad.abs().sum()) > 0.0


def test_ft8_interface_parameterization_and_variant_gating_match_parent() -> None:
    cfg = integ.cpu_probe_config()
    torch.manual_seed(integ.CPU_TEST_SEED)
    parent = base.CapabilityAblationLM("B_backbone_plus_memory", cfg)
    child = integ.build_cpu_model("B_backbone_plus_memory", cfg=cfg)
    assert set(parent.state_dict()) == set(child.state_dict())
    assert sum(p.numel() for p in parent.parameters()) == sum(p.numel() for p in child.parameters())
    assert all(isinstance(stage.memory, PretrainableDeltaFastMemory) for stage in child.stages)
    expected = {
        "A_backbone": (False, False),
        "B_backbone_plus_memory": (True, False),
        "C_backbone_plus_routing": (False, True),
        "D_combined": (True, True),
    }
    chunk = _pad([5001, 5002, 5003])
    for variant, flags in expected.items():
        model = integ.build_cpu_model(variant, cfg=cfg)
        assert (model.memory_enabled, model.adaptive_routing_enabled) == flags
        out = _forward(model, chunk, update_memory=True)
        assert set(("answer_logits", "state", "stage_routes", "routing_mode", "prediction_positions")) <= set(out)


def test_ft9_frozen_sources_are_byte_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "tam_research/aera_issue763_key_conditioned_memory_repair_cpu.py": integ.FROZEN_REPAIR_BLOB,
        "tests/test_aera_issue763_key_conditioned_memory_repair_cpu.py": integ.FROZEN_REPAIR_TEST_BLOB,
        "tam_research/aera_issue748_memory_capability_seed1_harness.py": integ.FROZEN_WRAPPER_BLOB,
        "tam_research/aera_issue748_memory_capability_seed1_harness_base.py": integ.FROZEN_BASE_BLOB,
        "tam_research/aera_memory_capability_gate_v1.py": integ.FROZEN_FIXTURE_BLOB,
        "tam_research/aera_hardware_core_v18.py": integ.FROZEN_V18_BLOB,
        "tam_research/aera_delta_memory.py": integ.FROZEN_DELTA_BLOB,
    }
    for relative, blob in expected.items():
        actual = subprocess.check_output(["git", "hash-object", str(root / relative)], text=True).strip()
        assert actual == blob


def test_protocol_matches_implementation_and_has_no_scientific_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "docs/aera_issue770_integrated_event_memory_protocol.json").read_text())
    assert protocol["research_issue"] == integ.RESEARCH_ISSUE
    assert protocol["source_main"] == integ.SOURCE_MAIN
    assert protocol["cpu_test_seed"] == integ.CPU_TEST_SEED
    assert protocol["falsifiers"] == [f"FT{i}" for i in range(1, 10)]
    assert all(value is False for value in protocol["authority"].values())
