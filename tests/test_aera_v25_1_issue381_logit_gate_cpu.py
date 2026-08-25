from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch

from tam_research import aera_v25_1_systems_guarded as guarded
from tam_research import aera_v25_post8471_triage as triage


def _base_result() -> dict:
    hashes = {"aera": "a", "transformer": "t"}
    return {
        "rows": {"8": {}, "64": {}},
        "per_batch_pass": {"8": True, "64": True},
        "parameter_versions_unchanged": True,
        "checkpoint_hashes_before": dict(hashes),
        "checkpoint_hashes_after": dict(hashes),
        "checkpoint_hashes_unchanged": True,
        "overall_pass": True,
        "decision": "PASS_SYSTEMS_EQUIVALENT_CANDIDATE",
    }


def test_issue381_fixed_tokens_preserve_issue379_seed_rule():
    device = torch.device("cpu")
    actual = guarded._fixed_system_tokens(8, device)
    generator = torch.Generator(device="cpu").manual_seed(
        triage.DIAGNOSTIC_SEED + 10_000 + 8
    )
    expected = torch.randint(
        0,
        triage.VOCAB_SIZE,
        (8, triage.SEQ_LEN),
        generator=generator,
    )
    assert torch.equal(actual, expected)


def test_issue381_logit_equivalence_uses_frozen_atol_and_rtol(monkeypatch):
    def close_call(model, tokens, *, update_memory):
        assert update_memory is True
        if model == "original":
            logits = torch.tensor([[[1.0, -2.0]]], dtype=torch.float32)
        else:
            logits = torch.tensor([[[1.0 + 5e-7, -2.0]]], dtype=torch.float32)
        return {"logits": logits}

    monkeypatch.setattr(guarded.systems, "_model_call", close_call)
    close = guarded._logit_equivalence("original", "candidate", torch.zeros(1, 1))
    assert close["pass"] is True
    assert close["atol"] == 1e-6
    assert close["rtol"] == 1e-6

    def far_call(model, tokens, *, update_memory):
        if model == "original":
            logits = torch.tensor([[[1.0, -2.0]]], dtype=torch.float32)
        else:
            logits = torch.tensor([[[1.0 + 2e-5, -2.0]]], dtype=torch.float32)
        return {"logits": logits}

    monkeypatch.setattr(guarded.systems, "_model_call", far_call)
    far = guarded._logit_equivalence("original", "candidate", torch.zeros(1, 1))
    assert far["pass"] is False
    assert far["max_abs"] > 1e-6


def test_issue381_logit_gate_is_authoritative_for_final_decision():
    hashes = {"aera": "a", "transformer": "t"}
    equivalence = {
        "8": {"pass": True, "atol": 1e-6, "rtol": 1e-6, "max_abs": 0.0},
        "64": {"pass": False, "atol": 1e-6, "rtol": 1e-6, "max_abs": 2e-5},
    }
    result = guarded._apply_logit_gate(
        _base_result(),
        equivalence,
        guard_parameter_versions_unchanged=True,
        hashes_before_guard=hashes,
        hashes_after_guard=hashes,
    )
    assert result["rows"]["8"]["logit_equivalence"]["pass"] is True
    assert result["rows"]["64"]["logit_equivalence"]["pass"] is False
    assert result["per_batch_pass"]["8"] is True
    assert result["per_batch_pass"]["64"] is False
    assert result["overall_pass"] is False
    assert result["decision"] == "FAIL_FROZEN_SYSTEMS_GATE"


def test_issue381_guarded_path_remains_inference_only_and_launcher_uses_it():
    source = inspect.getsource(guarded)
    assert "TokenBin" not in source
    tree = ast.parse(source)
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {
            "backward",
            "step",
            "zero_grad",
        }:
            forbidden.append(func.attr)
        if isinstance(func, ast.Name) and func.id.lower().startswith("optimizer"):
            forbidden.append(func.id)
    assert forbidden == []

    root = Path(__file__).resolve().parents[1]
    launcher = (root / "modal_aera_v25_1_systems_app.py").read_text()
    assert "run_guarded_systems_comparison" in launcher
    assert "run_systems_comparison(run_dir=SOURCE_RUN_DIR)" not in launcher
    assert 'gpu="L4"' in launcher
    assert "MAX_GPU_SECONDS = 600" in launcher
