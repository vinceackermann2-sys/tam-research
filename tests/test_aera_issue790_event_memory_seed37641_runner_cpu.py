from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tam_research import aera_issue790_event_memory_seed37641_scientific_adapter as adapter
from tam_research import aera_issue785_eval_materialization_bridge_cpu as bridge
from tam_research import aera_issue748_memory_capability_seed1_harness as corrected
from tam_research import aera_memory_capability_gate_v1 as gate
from tam_research.aera_issue770_integrated_event_memory_cpu import (
    IntegratedEventMemoryCapabilityLM,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "tam_research/aera_issue785_eval_materialization_bridge_cpu.py": "aa90b3ae3266786323b0517de8385135869c3653",
    "tam_research/aera_issue748_memory_capability_seed1_harness.py": "afc939a69633f68ded05eb585c95a599a8f5c981",
    "tam_research/aera_issue776_event_memory_scientific_adapter.py": "1489df753f2eee0f3acc5aba049c0eeb40fce10d",
    "tam_research/aera_issue748_memory_capability_seed1_harness_base.py": "40003b68987d026265b1d53fcb637f18277f9cac",
    "tam_research/aera_memory_capability_gate_v1.py": "981602432b684989f7a5011ff953e6965368c5e5",
    "tam_research/aera_issue770_integrated_event_memory_cpu.py": "b695b1b7b7476be3a16433c96dff32870f2e1f49",
    "tam_research/aera_hardware_core_v18.py": "97861a2407876f62665b13140c2135b4a11d4597",
    "tam_research/aera_delta_memory.py": "ec0b5d29b3d4ac27bd60fd9c152480b9f177c3e9",
}


def _blob(path: str) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(ROOT / path)],
        text=True,
    ).strip()


def test_issue790_frozen_lineage_blobs_are_exact() -> None:
    for path, expected in EXPECTED.items():
        assert _blob(path) == expected


def test_issue790_protocol_is_exact_adapter_snapshot_and_zero_authority() -> None:
    protocol = json.loads(
        (ROOT / "docs/aera_issue790_event_memory_seed37641_protocol.json").read_text()
    )
    assert protocol == adapter.protocol_snapshot()
    assert protocol["candidate_model_seed"] == 37641
    assert protocol["consumed_model_seeds"] == [17641, 27641]
    assert all(value is False for value in protocol["authority"].values())


def test_issue790_adapter_binds_authoritative_bridge_and_corrected_materializer() -> None:
    assert adapter.evaluation_cases is bridge.evaluation_cases
    assert bridge.evaluation_cases is corrected.evaluation_cases
    assert adapter.base.evaluation_cases is corrected.evaluation_cases


def test_issue790_corrected_materialization_is_deterministic_balanced_unique_heldout() -> None:
    first = adapter.evaluation_cases()
    second = adapter.evaluation_cases()
    authoritative = corrected.evaluation_cases()
    assert first == second == authoritative
    assert len(first) == 432
    assert len({case.sample_index for case in first}) == 432

    nonreset = [case for case in first if not case.reset_before_query]
    reset = [case for case in first if case.reset_before_query]
    assert len(nonreset) == 324
    assert len(reset) == 108
    for distance in (2, 8, 32):
        assert sum(case.retention_distance_chunks == distance for case in nonreset) == 108
    for distance in (8, 32):
        assert sum(case.retention_distance_chunks == distance for case in reset) == 54
    assert all(case.split == "eval" for case in first)
    assert all(gate.pair_split(case.target_key, case.original_value) == "eval" for case in first)
    for case in first:
        if case.correction:
            assert gate.pair_split(case.target_key, case.latest_value) == "eval"


def test_issue790_adapter_does_not_use_raw_base_materializer(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_raw_base_materializer():
        raise AssertionError("raw #748 base materializer executed")

    monkeypatch.setattr(adapter.base, "evaluation_cases", forbidden_raw_base_materializer)
    cases = adapter.evaluation_cases()
    assert len(cases) == 432


def test_issue790_scientific_seed_guard_rejects_consumed_and_locks_37641() -> None:
    for seed in (17_641, 27_641, 37_641):
        with pytest.raises(PermissionError):
            adapter.build_repaired_model(
                "A_backbone",
                seed,
                scientific_seed_authorized=False,
                device="cpu",
            )
    for seed in (17_641, 27_641):
        with pytest.raises(PermissionError):
            adapter._assert_candidate_seed_allowed(
                seed, scientific_seed_authorized=True
            )
    adapter._assert_candidate_seed_allowed(
        37_641, scientific_seed_authorized=True
    )
    assert adapter.CONSUMED_MODEL_SEEDS == (17_641, 27_641)
    assert adapter.CANDIDATE_MODEL_SEED == 37_641


@pytest.mark.parametrize("variant", adapter.TRAINED_VARIANTS)
def test_issue790_all_variants_use_repaired_class_at_frozen_scientific_config(variant) -> None:
    model = adapter.build_repaired_model(
        variant,
        gate.CPU_TEST_SEED,
        scientific_seed_authorized=False,
        device="cpu",
    )
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


def test_issue790_reuses_frozen_training_builder_seam_and_restores_it(monkeypatch) -> None:
    original_builder = adapter.base.build_model
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
        model = adapter.base.build_model(
            variant_name,
            model_seed,
            scientific_seed_authorized=scientific_seed_authorized,
            device=device,
        )
        seen["model"] = model
        return model, {"fake": True}

    monkeypatch.setattr(adapter.base, "train_variant", fake_train_variant)
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
    assert adapter.base.build_model is original_builder


def test_issue790_frozen_training_threshold_geometry_is_unchanged() -> None:
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


def test_issue790_runner_has_one_future_l4_site_and_preflight_is_zero_science() -> None:
    source = (ROOT / "modal_aera_issue790_event_memory_seed37641_runner.py").read_text()
    assert source.count('gpu="L4"') == 1
    assert 'gpu="A100"' not in source
    assert 'gpu="H100"' not in source
    preflight = source.split("def preflight(", 1)[1].split("def _write_result", 1)[0]
    assert "train_variant(" not in preflight
    assert "evaluate_model(" not in preflight
    assert "build_repaired_model(" not in preflight
    assert ".write_text(" not in preflight
    assert "torch.cuda" not in preflight
    assert adapter.PREAUTH_PREFIX in source
    assert adapter.L4_PREFIX in source
    assert adapter.RESULT_PATH in source
    assert adapter.CHECKPOINT_DIR in source
    assert "HEARTBEAT_SECONDS = 60" in source
    assert "/event-memory-repair-v1-seed27641/" not in source


def test_issue790_workflow_is_owner_only_fresh_namespace_and_explicit_auth_gated() -> None:
    source = (
        ROOT / ".github/workflows/aera-issue790-event-memory-seed37641-runner.yml"
    ).read_text()
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert adapter.PREAUTH_PREFIX in source
    assert adapter.L4_PREFIX in source
    assert "## #790 repaired seed 37641 L4 scientific authorization" in source
    assert "Authorize seed: `37641`" in source
    assert "GITHUB_RUN_ATTEMPT" in source
    assert "modal_aera_issue790_event_memory_seed37641_runner.py::preauth_main" in source
    assert "modal_aera_issue790_event_memory_seed37641_runner.py::l4_main" in source
    assert "[aera-event-memory-repair-seed27641-l4-v1]" not in source
    assert "/event-memory-repair-v1-seed27641/" not in source


def test_issue790_final_positive_decision_changes_only_post_result_wording() -> None:
    frozen = {
        "recommendation": "CONFIRM_SEEDS_2_3_RECOMMENDED_REQUIRES_SEPARATE_AUTHORIZATION",
        "capability_screen_pass": True,
        "checks": {"sentinel": 1},
    }
    out = adapter.normalize_final_seed_decision(frozen)
    assert out["checks"] is frozen["checks"]
    assert out["capability_screen_pass"] is True
    assert out["legacy_frozen_recommendation"] == frozen["recommendation"]
    assert (
        out["recommendation"]
        == "REPAIRED_SEED37641_POSITIVE_BOUNDED_CAPABILITY_EVIDENCE_REVIEW_REQUIRED"
    )
