from __future__ import annotations

from tam_research.chm_v1_gate_metrics import summarize_preregistered_probe_metrics
from tam_research.chm_v1_long_memory_eval import EncodedProbe


def _overwrite_probe(case_id: int, *, answer: int = 10, stale: tuple[int, int] = (11, 12)) -> EncodedProbe:
    return EncodedProbe(
        family="overwrite",
        case_id=case_id,
        prompt_text="synthetic-metric-only",
        prompt_ids=(1, 2),
        answer_token_id=answer,
        candidate_token_ids=(10, 11, 12, 13),
        evidence_end_token=0,
        query_token=1,
        evidence_distance=513,
        stale_token_ids=stale,
    )


def test_stale_gate_passes_at_exact_fifty_percent_relative_reduction() -> None:
    probes = [_overwrite_probe(i) for i in range(4)]
    local = [11, 11, 12, 12]  # 100% stale
    eiem = [11, 12, 10, 10]  # 50% stale

    summary = summarize_preregistered_probe_metrics(probes, local, eiem)
    overwrite = summary["overwrite"]

    assert overwrite["local_stale_error"] == 1.0
    assert overwrite["eiem_stale_error"] == 0.5
    assert overwrite["stale_relative_reduction"] == 0.5
    assert overwrite["stale_gate_pass"] is True
    assert overwrite["stale_gate_rule"] == "eiem<=0.5*local"


def test_stale_gate_fails_without_required_relative_reduction() -> None:
    probes = [_overwrite_probe(i) for i in range(4)]
    local = [11, 12, 10, 10]  # 50% stale
    eiem = [11, 12, 10, 10]  # 50% stale, no reduction

    summary = summarize_preregistered_probe_metrics(probes, local, eiem)
    overwrite = summary["overwrite"]

    assert overwrite["local_stale_error"] == 0.5
    assert overwrite["eiem_stale_error"] == 0.5
    assert overwrite["stale_relative_reduction"] == 0.0
    assert overwrite["stale_gate_pass"] is False


def test_low_local_stale_error_uses_half_percentage_point_exception() -> None:
    probes = [_overwrite_probe(i) for i in range(200)]
    local = [10] * 200
    local[0] = 11  # 0.5%, therefore the <1% exception applies

    eiem_at_limit = [10] * 200
    eiem_at_limit[0] = 11
    eiem_at_limit[1] = 12  # 1.0% = local + 0.5 pp
    passed = summarize_preregistered_probe_metrics(probes, local, eiem_at_limit)["overwrite"]
    assert passed["local_stale_error"] == 0.005
    assert passed["eiem_stale_error"] == 0.01
    assert passed["stale_relative_reduction"] is None
    assert passed["stale_gate_pass"] is True
    assert passed["stale_gate_rule"] == "eiem<=local+0.005_when_local<0.01"

    eiem_too_high = [10] * 200
    eiem_too_high[0:3] = [11, 12, 11]  # 1.5%, above allowed 1.0%
    failed = summarize_preregistered_probe_metrics(probes, local, eiem_too_high)["overwrite"]
    assert failed["eiem_stale_error"] == 0.015
    assert failed["stale_gate_pass"] is False


def test_wrapper_preserves_existing_accuracy_metrics() -> None:
    probes = [_overwrite_probe(0), _overwrite_probe(1)]
    local = [10, 11]
    eiem = [10, 10]

    summary = summarize_preregistered_probe_metrics(probes, local, eiem)
    overwrite = summary["overwrite"]

    assert overwrite["local_accuracy"] == 0.5
    assert overwrite["eiem_accuracy"] == 1.0
    assert overwrite["absolute_gain"] == 0.5
