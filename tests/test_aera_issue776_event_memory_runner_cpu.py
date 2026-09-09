from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tam_research import aera_issue748_memory_capability_seed1_harness_base as base
from tam_research import aera_memory_capability_gate_v1 as gate
from tam_research import aera_issue776_event_memory_scientific_adapter as adapter
from tam_research.aera_issue770_integrated_event_memory_cpu import (
    IntegratedEventMemoryCapabilityLM,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "tam_research/aera_issue770_integrated_event_memory_cpu.py": "b695b1b7b7476be3a16433c96dff32870f2e1f49",
    "tests/test_aera_issue770_integrated_event_memory_cpu.py": "68458cffb5c9cc9d11c1ea84d811c4fa45e3647a",
    "tam_research/aera_issue748_memory_capability_seed1_harness_base.py": "40003b68987d026265b1d53fcb637f18277f9cac",
    "tam_research/aera_memory_capability_gate_v1.py": "981602432b684989f7a5011ff953e6965368c5e5",
    "tam_research/aera_hardware_core_v18.py": "97861a2407876f62665b13140c2135b4a11d4597",
    "tam_research/aera_delta_memory.py": "ec0b5d29b3d4ac27bd60fd9c152480b9f177c3e9",
}


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue776_protocol_is_exact_adapter_snapshot_and_zero_authority():
    protocol = json.loads(
        (ROOT / "docs/aera_issue776_event_memory_repair_protocol.json").read_text()
    )
    assert protocol == adapter.protocol_snapshot()
    assert protocol["candidate_model_seed"] == 27641
    assert protocol["consumed_model_seed"] == 17641
    assert protocol["future_model_seed_locked"] == 37641
    assert all(value is False for value in protocol["authority"].values())


def test_issue776_scientific_seed_guard_fails_closed():
    with pytest.raises(PermissionError):
        adapter.build_repaired_model("A_backbone", 27641)
    with pytest.raises(PermissionError):
        adapter.build_repaired_model(
            "A_backbone", 17641, scientific_seed_authorized=True
        )
    with pytest.raises(PermissionError):
        adapter.build_repaired_model(
            "A_backbone", 37641, scientific_seed_authorized=True
        )


@pytest.mark.parametrize("variant", adapter.TRAINED_VARIANTS)
def test_issue776_all_scientific_variants_use_repaired_class_and_frozen_scientific_config(
    variant,
):
    model = adapter.build_repaired_model(variant, gate.CPU_TEST_SEED)
    assert isinstance(model, IntegratedEventMemoryCapabilityLM)
    assert model.cfg.d_model == 128
    assert model.cfg.n_stages == 4
    assert model.cfg.n_heads == 4
    assert model.cfg.chunk_size == 256
    assert model.cfg.n_experts == 8
    assert model.cfg.max_active_experts == 2
    assert model.cfg.memory_dim == 32
    assert model.cfg.fast_memory_lr == pytest.approx(0.2)
    assert model.cfg.fast_memory_decay == pytest.approx(0.999)


def test_issue776_reuses_frozen_training_builder_seam_and_restores_it(monkeypatch):
    original_builder = base.build_model
    seen = {}

    def fake_train_variant(
        variant_name,
        model_seed,
        *,
        device,
        scientific_seed_authorized=False,
        max_steps=None,
        checkpoint_dir=None,
    ):
        model = base.build_model(
            variant_name,
            model_seed,
            scientific_seed_authorized=scientific_seed_authorized,
            device=device,
        )
        seen["model"] = model
        return model, {"fake": True}

    monkeypatch.setattr(base, "train_variant", fake_train_variant)
    model, info = adapter.train_variant(
        "B_backbone_plus_memory",
        gate.CPU_TEST_SEED,
        device="cpu",
        scientific_seed_authorized=False,
        max_steps=1,
        checkpoint_dir=None,
    )
    assert info == {"fake": True}
    assert model is seen["model"]
    assert isinstance(model, IntegratedEventMemoryCapabilityLM)
    assert base.build_model is original_builder


def test_issue776_reexports_frozen_eval_and_decision_contract_without_copying():
    assert adapter.evaluation_cases is base.evaluation_cases
    assert adapter.evaluate_model is base.evaluate_model
    assert adapter.evaluate_simple_retrieval is base.evaluate_simple_retrieval
    assert adapter.measure_model_inference_latency_ms is base.measure_model_inference_latency_ms
    assert adapter.select_equal_gpu_time_checkpoints is base.select_equal_gpu_time_checkpoints
    assert adapter.seed_decision is base.seed1_decision


def test_issue776_frozen_training_and_threshold_geometry_is_unchanged():
    p = adapter.protocol_snapshot()
    t = p["training"]
    e = p["evaluation"]
    x = p["thresholds"]
    assert t["trained_variants"] == [
        "A_backbone",
        "B_backbone_plus_memory",
        "C_backbone_plus_routing",
        "D_combined",
    ]
    assert t["token_budget_per_variant"] == 16_777_216
    assert t["optimizer_steps"] == 1024
    assert t["tokens_per_optimizer_step"] == 16_384
    assert t["checkpoint_interval_tokens"] == 2_097_152
    assert t["checkpoint_every_steps"] == 128
    assert t["max_gpu_seconds_per_variant"] == 3600
    assert e["cases"] == e["unique_case_ids"] == 432
    assert e["nonreset_per_distance"] == {"2": 108, "8": 108, "32": 108}
    assert e["reset_per_distance"] == {"8": 54, "32": 54}
    assert x == {
        "long_distance_accuracy_gain_min": 0.10,
        "paired_ci_lower_accuracy_gain_min": 0.05,
        "correction_accuracy_gain_min": 0.10,
        "stale_value_error_reduction_min": 0.10,
        "routing_control_accuracy_margin_min": 0.05,
        "session_reset_leakage_max": 0.01,
        "simple_retrieval_dominance_margin": 0.05,
        "memory_inference_latency_multiplier_max": 2.0,
        "memory_train_time_multiplier_max": 1.5,
        "parameter_delta_fraction_max": 0.05,
    }


def test_issue776_frozen_aera_sources_and_ft_contract_are_byte_identical():
    for rel, expected in EXPECTED.items():
        assert git_blob(ROOT / rel) == expected


def test_issue776_runner_has_one_gpu_site_and_cpu_preflight_cannot_train_or_write():
    source = (ROOT / "modal_aera_issue776_event_memory_repair_runner.py").read_text()
    assert source.count('gpu="L4"') == 1
    assert 'gpu="A100"' not in source
    assert 'gpu="H100"' not in source
    preflight = source.split("def preflight()", 1)[1].split("def _write_result", 1)[0]
    assert "train_variant(" not in preflight
    assert "evaluate_model(" not in preflight
    assert "build_repaired_model(" not in preflight
    assert ".write_text(" not in preflight
    assert "torch.cuda" not in preflight
    assert "/vol/aera-capability/issue748-memory-capability-seed1/" not in source
    assert "[aera-issue748-memory-capability-seed1-" not in source
    assert adapter.PREAUTH_PREFIX in source
    assert adapter.L4_PREFIX in source
    assert adapter.RESULT_PATH in source
    assert adapter.CHECKPOINT_DIR in source
    assert "HEARTBEAT_SECONDS = 60" in source


def test_issue776_workflow_is_owner_only_exact_namespace_and_requires_future_explicit_auth():
    source = (
        ROOT / ".github/workflows/aera-issue776-event-memory-repair-runner.yml"
    ).read_text()
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert adapter.PREAUTH_PREFIX in source
    assert adapter.L4_PREFIX in source
    assert "## #776 repaired seed 27641 L4 scientific authorization" in source
    assert "Authorize seed: `27641`" in source
    assert "Authorize seed: `37641`" not in source
    assert "[aera-issue748-memory-capability-seed1-" not in source
    assert "/vol/aera-capability/issue748-memory-capability-seed1/" not in source
    assert "GITHUB_RUN_ATTEMPT" in source
    assert "modal_aera_issue776_event_memory_repair_runner.py::preauth_main" in source
    assert "modal_aera_issue776_event_memory_repair_runner.py::l4_main" in source


def test_issue776_positive_decision_changes_only_next_seed_wording():
    frozen = {
        "recommendation": "CONFIRM_SEEDS_2_3_RECOMMENDED_REQUIRES_SEPARATE_AUTHORIZATION",
        "capability_screen_pass": True,
        "checks": {"sentinel": 1},
    }
    out = adapter.normalize_repaired_decision(frozen)
    assert out["checks"] is frozen["checks"]
    assert out["capability_screen_pass"] is True
    assert (
        out["legacy_frozen_recommendation"]
        == "CONFIRM_SEEDS_2_3_RECOMMENDED_REQUIRES_SEPARATE_AUTHORIZATION"
    )
    assert (
        out["recommendation"]
        == "REPAIRED_SEED27641_POSITIVE_CONFIRM_SEED37641_REQUIRES_SEPARATE_AUTHORIZATION"
    )
