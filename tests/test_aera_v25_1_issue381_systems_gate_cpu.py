from __future__ import annotations

import ast
from dataclasses import fields
import inspect
from pathlib import Path

import torch

from tests.test_aera_v25_1_execution_equivalent_cpu import _models
from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState
from tam_research.aera_hardware_core_v25_1 import (
    ExecutionEquivalentFactorizedIdentityContextMemory,
    HardwareAwareAERATextLMV251,
    _KNOWN_EMPTY_HINT,
    execution_equivalent_v25_1_protocol,
)
from tam_research import aera_v25_1_systems as systems


def test_issue381_known_empty_hint_is_ephemeral_and_cleared_after_update():
    _, candidate = _models(138201)
    memory = candidate.stages[0].memory
    assert isinstance(memory, ExecutionEquivalentFactorizedIdentityContextMemory)
    state = memory.empty_state(2, torch.device("cpu"), torch.float32)
    assert isinstance(state, ContextualEpisodicMemoryState)
    assert getattr(state, _KNOWN_EMPTY_HINT) is True
    assert [field.name for field in fields(ContextualEpisodicMemoryState)] == [
        "keys",
        "values",
        "strengths",
        "valid",
    ]
    assert not any(_KNOWN_EMPTY_HINT in key for key in candidate.state_dict())

    g = torch.Generator().manual_seed(138202)
    d_model = candidate.cfg.d_model
    identity = torch.randn(2, 2, d_model, generator=g)
    context = torch.randn(2, 2, d_model, generator=g)
    payload = torch.randn(2, 2, d_model, generator=g)
    strengths = torch.ones(2, 2, 1)
    updated = memory.update_block(identity, context, payload, strengths, state)
    assert getattr(updated, _KNOWN_EMPTY_HINT) is False


def test_issue381_cuda_paths_do_not_require_python_bool_of_cuda_memory_or_stage0_gate():
    read_source = inspect.getsource(
        ExecutionEquivalentFactorizedIdentityContextMemory.read
    )
    route_source = inspect.getsource(HardwareAwareAERATextLMV251._route_one_stage)
    assert 'state.valid.device.type == "cpu" and not bool(state.valid.any())' in read_source
    assert 'gate.device.type == "cpu" and not bool((gate[:, 0] >= 0.5).all())' in route_source
    protocol = execution_equivalent_v25_1_protocol()
    assert protocol["cuda_scalar_empty_read_sync"] is False
    assert protocol["cuda_scalar_foundation_invariant_sync"] is False
    assert protocol["known_empty_hint_persistent"] is False
    assert protocol["state_dict_schema_changed"] is False
    assert protocol["state_bytes_real_language_four_stage_memory_dim50"] == 77_760


def test_issue381_systems_protocol_preserves_frozen_gate_and_no_scale_authorization():
    check = systems.cpu_contract_preflight()
    protocol = check["protocol"]
    assert protocol["research_issue"] == 381
    assert protocol["source_checkpoint_seed"] == 8471
    assert protocol["batch_sizes"] == [8, 64]
    assert protocol["warmup_calls"] == 3
    assert protocol["timed_calls_per_round"] == 20
    assert protocol["rounds"] == 5
    assert protocol["batch8_min_full_speed_ratio"] == 0.25
    assert protocol["batch64_min_full_speed_ratio"] == 1.25
    assert protocol["production_write_geometry"] == [16, 255, 1]
    assert protocol["persistent_state_bytes_per_session"] == 77_760
    assert protocol["max_gpu_seconds"] == 600
    assert protocol["actions_attempt"] == 1
    assert protocol["training_performed"] is False
    assert protocol["optimizer_created"] is False
    assert protocol["backward_performed"] is False
    assert protocol["corpus_accessed"] is False
    assert protocol["checkpoint_write_authorized"] is False
    assert protocol["scientific_seed_consumed"] is False
    assert protocol["architecture_freeze_authorized"] is False
    assert protocol["s2_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_issue381_systems_module_has_no_training_or_optimizer_call_path():
    source = inspect.getsource(systems)
    assert "TokenBin" not in source
    tree = ast.parse(source)
    forbidden_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {
            "backward",
            "step",
            "zero_grad",
        }:
            forbidden_calls.append(func.attr)
        if isinstance(func, ast.Name) and func.id.lower().startswith("optimizer"):
            forbidden_calls.append(func.id)
    assert forbidden_calls == []


def test_issue381_workflow_is_single_attempt_l4_and_launcher_has_no_corpus_binding():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/aera-v25-1-systems-l4.yml").read_text()
    launcher = (root / "modal_aera_v25_1_systems_app.py").read_text()
    assert "[aera-v25-1-systems-l4]" in workflow
    assert "GITHUB_RUN_ATTEMPT" in workflow
    assert "issue #381" in workflow.lower()
    assert workflow.count("modal run modal_aera_v25_1_systems_app.py") == 1
    assert "rerun" not in workflow.lower()
    assert 'gpu="L4"' in launcher
    assert "MAX_GPU_SECONDS = 600" in launcher
    assert "timeout=MAX_GPU_SECONDS" in launcher
    assert "/vol/aera-real-language/v25-dev-seed8471" in launcher
    assert "/vol/aera-real-language/v25-1-issue381-systems/result.json" in launcher
    assert "DATA_DIR" not in launcher
    assert "TokenBin" not in launcher
