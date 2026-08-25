from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from aera_v19_memory_necessity_cpu import diagnostic_config
from tam_research import aera_v25_1_systems as systems
from tam_research import aera_v25_1_systems_guarded as guarded
from tam_research.aera_hardware_core_v25_1_nohost import (
    HardwareAwareAERATextLMV251NoHostTelemetry,
)


def test_issue381_final_candidate_binding_is_scoped_and_restored():
    prior = systems.HardwareAwareAERATextLMV251
    assert prior is not HardwareAwareAERATextLMV251NoHostTelemetry

    with guarded._final_candidate_binding():
        assert systems.HardwareAwareAERATextLMV251 is HardwareAwareAERATextLMV251NoHostTelemetry

    assert systems.HardwareAwareAERATextLMV251 is prior


def test_issue381_final_candidate_binding_restores_after_exception():
    prior = systems.HardwareAwareAERATextLMV251
    with pytest.raises(RuntimeError, match="intentional integration probe"):
        with guarded._final_candidate_binding():
            assert systems.HardwareAwareAERATextLMV251 is HardwareAwareAERATextLMV251NoHostTelemetry
            raise RuntimeError("intentional integration probe")
    assert systems.HardwareAwareAERATextLMV251 is prior


def test_issue381_guarded_loader_constructs_final_candidate_and_restores(monkeypatch):
    prior = systems.HardwareAwareAERATextLMV251
    cfg = diagnostic_config()
    torch.manual_seed(138701)
    final_candidate = HardwareAwareAERATextLMV251NoHostTelemetry(cfg).eval()
    seen: dict[str, object] = {}

    def fake_load_models(*, run_dir: str, device: torch.device):
        assert run_dir == "/tmp/issue381-read-only"
        assert device == torch.device("cpu")
        seen["candidate_constructor"] = systems.HardwareAwareAERATextLMV251
        return object(), final_candidate, object()

    monkeypatch.setattr(systems, "load_models", fake_load_models)
    loaded = guarded.load_guarded_models(
        run_dir="/tmp/issue381-read-only",
        device=torch.device("cpu"),
    )
    assert loaded[1] is final_candidate
    assert seen["candidate_constructor"] is HardwareAwareAERATextLMV251NoHostTelemetry
    assert systems.HardwareAwareAERATextLMV251 is prior


def test_issue381_guarded_loader_rejects_wrong_candidate(monkeypatch):
    prior = systems.HardwareAwareAERATextLMV251

    def fake_wrong_loader(*, run_dir: str, device: torch.device):
        return object(), object(), object()

    monkeypatch.setattr(systems, "load_models", fake_wrong_loader)
    with pytest.raises(RuntimeError, match="did not construct final v25.1 candidate"):
        guarded.load_guarded_models(
            run_dir="/tmp/issue381-read-only",
            device=torch.device("cpu"),
        )
    assert systems.HardwareAwareAERATextLMV251 is prior


def test_issue381_frozen_evaluator_stays_unmodified_by_final_integration():
    source = inspect.getsource(systems)
    assert "HardwareAwareAERATextLMV251NoHostTelemetry" not in source
    assert systems.CPU_EQUIVALENCE_ATOL == 1e-6
    assert systems.CPU_EQUIVALENCE_RTOL == 1e-6
    assert systems.SYSTEM_BATCH_SIZES == (8, 64)
    assert systems.BATCH8_MIN_FULL_SPEED_RATIO == 0.25
    assert systems.BATCH64_MIN_FULL_SPEED_RATIO == 1.25
    assert systems.EXPECTED_SELECTED_WRITES == 16
    assert systems.EXPECTED_CANDIDATES == 255
    assert systems.EXPECTED_VECTORIZED_UPDATES == 1
    assert systems.EXPECTED_STATE_BYTES == 77_760


def test_issue381_guarded_timed_path_and_modal_preflight_use_final_binding():
    guarded_source = inspect.getsource(guarded.run_guarded_systems_comparison)
    assert "with _final_candidate_binding():" in guarded_source
    assert "systems.run_systems_comparison(run_dir=run_dir)" in guarded_source
    assert "systems.load_models(" in guarded_source
    assert "HardwareAwareAERATextLMV251NoHostTelemetry" in guarded_source

    root = Path(__file__).resolve().parents[1]
    launcher = (root / "modal_aera_v25_1_systems_app.py").read_text()
    assert "from tam_research.aera_v25_1_systems_guarded import load_guarded_models" in launcher
    assert "original, candidate, transformer = load_guarded_models(" in launcher
    assert "strict_v25_and_final_v25_1_checkpoint_load_cpu" in launcher
    assert "load_models," not in launcher
    assert 'gpu="L4"' in launcher
    assert "MAX_GPU_SECONDS = 600" in launcher
    assert "TokenBin" not in launcher
