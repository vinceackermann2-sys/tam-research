from __future__ import annotations

import pytest

import tam_research.chm_v1_100m_stage_b as stage_b


def _row(*, params: int, tps: float, peak_gib: float = 10.0) -> dict[str, object]:
    return {
        "trainable_parameters": params,
        "measured_tokens": stage_b.MEASURED_TOKENS_PER_MODEL,
        "finite_loss": True,
        "finite_parameters": True,
        "tokens_per_second": tps,
        "peak_vram_bytes": int(peak_gib * 1024**3),
    }


def test_stage_b_contract_is_frozen() -> None:
    manifest = stage_b.validate_contract()
    assert stage_b.PARENT_RESEARCH_ISSUE == 977
    assert stage_b.SYSTEMS_ISSUE == 983
    assert stage_b.ENGINEERING_SEED == 977_201
    assert stage_b.RESERVED_SCIENTIFIC_SEED == 977_001
    assert stage_b.GPU_CLASS == "L4"
    assert stage_b.CPU_CORES == 4
    assert stage_b.RAM_GIB == 16
    assert stage_b.MAX_GPU_SECONDS == 1_200
    assert stage_b.MAX_STAGE_B_COMPUTE_USD == 0.50
    assert stage_b.WARMUP_STEPS == 2
    assert stage_b.MEASURED_STEPS == 8
    assert stage_b.TOKENS_PER_OPTIMIZER_STEP == 16_384
    assert stage_b.MEASURED_TOKENS_PER_MODEL == 131_072
    assert manifest["scientific_execution_authorized"] is False
    assert manifest["future_screen_tokens_per_model_not_authorized"] == 33_554_432


def test_stage_b_seed_guard_refuses_scientific_and_prior_seeds() -> None:
    assert stage_b.validate_engineering_seed(977_201) == 977_201
    for seed in stage_b.BLOCKED_PRIOR_SEEDS:
        with pytest.raises(RuntimeError, match="blocked/scientific"):
            stage_b.validate_engineering_seed(seed)
    with pytest.raises(RuntimeError, match="accepts only engineering seed"):
        stage_b.validate_engineering_seed(983_001)


def test_stage_b_pass_classifier() -> None:
    result = stage_b.classify_stage_b(
        local=_row(params=101_803_520, tps=5_000.0),
        eiem=_row(params=101_836_800, tps=3_500.0),
        live_hourly_resource_usd=1.20,
    )
    assert result["classification"] == "CHM_V1_100M_STAGE_B_SYSTEMS_PASS"
    assert result["passed"] is True
    assert result["stop_reasons"] == []
    assert result["projection"]["eiem_local_throughput_ratio"] == pytest.approx(0.7)
    assert result["projection"]["pair_projected_seconds"] < 21_600
    assert result["projection"]["pair_projected_compute_usd"] < 6.0


def test_stage_b_stop_classifier_is_predeclared() -> None:
    result = stage_b.classify_stage_b(
        local=_row(params=101_803_520, tps=5_000.0),
        eiem=_row(params=101_836_800, tps=2_000.0, peak_gib=22.5),
        live_hourly_resource_usd=1.20,
    )
    assert result["classification"] == "CHM_V1_100M_STAGE_B_SYSTEMS_STOP"
    assert result["passed"] is False
    assert "eiem_throughput_below_gate" in result["stop_reasons"]
    assert "eiem_peak_vram_above_gate" in result["stop_reasons"]
    assert "eiem_local_throughput_ratio_below_gate" in result["stop_reasons"]


def test_projection_requires_positive_observed_values() -> None:
    with pytest.raises(ValueError):
        stage_b.project_from_throughput(
            local_tokens_per_second=0.0,
            eiem_tokens_per_second=3_000.0,
            live_hourly_resource_usd=1.0,
        )
