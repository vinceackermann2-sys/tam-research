from __future__ import annotations

import json
from pathlib import Path

import torch

from architectures.chm_v1 import cpu_mechanism_screen as screen


def test_frozen_result_is_exploratory_only() -> None:
    path = Path(__file__).with_name("exploratory_result_seed60430.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["seed"] == 60430
    assert result["classification"] == "EXPLORATORY_CPU_MECHANISM_SCREEN_ONLY"
    assert result["gpu_authorized"] is False
    assert result["scientific_seed_authorized"] is False
    assert result["decision"] == "PASS_MECHANISM_ONLY_PREREGISTER_SMALL_LM_TEST"


def test_seed60430_mechanism_gate_reproduces() -> None:
    # This is intentionally a bounded CPU rerun of the exploratory mechanism gate,
    # not a scientific LM experiment. Seed 60430 is frozen to this one mechanism record.
    torch.manual_seed(screen.SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = screen.AddressEncoder()
    train = screen.train_encoder(model)
    rare = screen.evaluate_rare_fact(model, 1024)
    overwrite = screen.evaluate_overwrite(model, 1024)
    multihop = screen.evaluate_multihop(model, 1024)

    assert train["last_batch_accuracy"] >= 0.99
    assert rare["indexed_matches_flat"] == 1.0
    assert rare["indexed_accuracy"] >= 0.95
    assert rare["fraction_flat_vector_reads"] <= 0.15
    assert overwrite["indexed_matches_flat"] == 1.0
    assert overwrite["indexed_accuracy"] >= 0.95
    assert overwrite["indexed_stale_error"] <= 0.02
    assert multihop["indexed_matches_flat"] == 1.0
    assert multihop["indexed_accuracy"] >= 0.90
    assert multihop["fraction_flat_vector_reads"] <= 0.15


def test_exact_search_matches_flat_for_frozen_eval_sizes() -> None:
    torch.manual_seed(screen.SEED)
    model = screen.AddressEncoder()
    screen.train_encoder(model)
    for n in screen.EVAL_SIZES:
        result = screen.evaluate_rare_fact(model, n)
        assert result["indexed_matches_flat"] == 1.0
