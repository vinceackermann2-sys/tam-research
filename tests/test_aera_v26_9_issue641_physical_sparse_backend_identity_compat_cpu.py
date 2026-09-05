from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tam_research import aera_v26_5_end_to_end_systems as base
from tam_research import aera_v26_9_issue641_physical_sparse_backend_identity_compat as repair

ROOT = Path(__file__).resolve().parents[1]

FROZEN_BLOBS = {
    "tam_research/aera_v26_5_end_to_end_systems.py": "c9731cae7e386f09b2a190b045532591c4fa00be",
    "tam_research/aera_v26_6_issue530_end_to_end_systems.py": "9d5a3c31f4a3862f96b957540baa2e0ec6a84c6b",
    "tam_research/aera_v26_8_issue562_end_to_end_systems.py": "3534103eea21f7c4d9d31798ad34601fd47090d6",
    "tam_research/aera_v26_8_issue569_end_to_end_systems_memory_safe.py": "1eeaa80adb2ae960e2a8dad06031c4ed5ca99203",
    "tam_research/aera_hardware_core_v26_9_ficem_read_identity_weight_visibility.py": "b81cc209f5d95abbe1fb8bd620c78e87c067bc19",
    "tam_research/aera_v26_9_issue625_schema_guard_repair1.py": "92d06a4954bca1b302355e81f5bf09b06fcee222",
}

CORRECT_ISSUE630_RESULT_SHA256 = (
    "ef92c85b55484b3ce191cd4016be86bf52da997a153f737194976164b29554b4"
)
WRONG_TRANSCRIBED_ISSUE630_SHA256 = (
    "ef92c85b4107bb756513c0130755190485c4d8fe77e7e68fabfdb7a522cae9c5"
)


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _historical_row(
    names: list[str],
    *,
    sparse: bool = True,
    coalesced: bool = True,
    activity: bool = True,
    historical_exact: bool | None = None,
) -> dict[str, object]:
    if historical_exact is None:
        historical_exact = bool(
            names and all(name == repair.HISTORICAL_BACKEND_NAME for name in names)
        )
    return {
        "pass": bool(sparse and coalesced and activity and historical_exact),
        "optional_executed_fractions": [1.0, 0.5],
        "sparse_route_exercised": sparse,
        "coalesced_select_merge_positive": coalesced,
        "backend_activity_positive": activity,
        "backend_names_exact": historical_exact,
        "backend_names": list(names),
        "backend_read_calls": 4,
        "backend_update_calls": 2,
        "backend_projected_update_calls": 1,
        "coalesced_float_state_select_calls": 3,
        "coalesced_valid_select_calls": 3,
        "coalesced_float_state_merge_calls": 3,
        "coalesced_valid_merge_calls": 3,
        "dense_masked_sparse_credit": False,
    }


def test_issue641_frozen_lineage_and_corrected_issue630_evidence_are_exact() -> None:
    assert repair.SOURCE_MAIN == "a5708223f2bba17c0eb931d63507fee93e98605b"
    assert repair.SOURCE_TREE == "388597aa7a07b919a691156694ea2732abdbe9f1"
    assert {_path: _blob(ROOT / _path) for _path in FROZEN_BLOBS} == FROZEN_BLOBS
    assert repair.ISSUE571_RESULT_SHA256 == (
        "afeeb62351cc4fb97d272c5b55c9621839e26f83753ae1fb237733d58a5ee472"
    )
    assert repair.ISSUE571_RUN == 33741700781
    assert repair.ISSUE571_JOB == 100604889696
    assert repair.ISSUE630_TRIGGER == 640
    assert repair.ISSUE630_RUN == 33985543569
    assert repair.ISSUE630_JOB == 101358253857
    assert repair.ISSUE630_RESULT_SHA256 == CORRECT_ISSUE630_RESULT_SHA256
    assert repair.ISSUE630_RESULT_SHA256 != WRONG_TRANSCRIBED_ISSUE630_SHA256


def test_issue641_backend_identity_is_single_exact_version_name() -> None:
    assert repair.HISTORICAL_BACKEND_NAME == (
        "triton-fused-ficem-read-repair5-write-tail-v26.4"
    )
    assert repair.V26_9_BACKEND_NAME == (
        "triton-ficem-read-v26.9-identity-weight-visibility-"
        "write-v26.6-materialize-cast"
    )
    assert repair.HISTORICAL_BACKEND_NAME != repair.V26_9_BACKEND_NAME


def test_issue641_pure_transform_preserves_every_nonidentity_field() -> None:
    historical = _historical_row([repair.V26_9_BACKEND_NAME] * 4, historical_exact=False)
    adapted = repair.adapt_frozen_physical_sparse_result_v26_9(historical)
    assert adapted["historical_backend_names_exact"] is False
    assert adapted["backend_names_exact"] is True
    assert adapted["pass"] is True

    identity_fields = {
        "pass",
        "backend_names_exact",
        "historical_backend_names_exact",
    }
    for key, value in historical.items():
        if key not in identity_fields:
            assert adapted[key] == value
            assert type(adapted[key]) is type(value)
    assert historical["backend_names_exact"] is False
    assert historical["pass"] is False


@pytest.mark.parametrize(
    "names",
    [
        [],
        [repair.HISTORICAL_BACKEND_NAME] * 4,
        [repair.V26_9_BACKEND_NAME, repair.HISTORICAL_BACKEND_NAME],
        [repair.V26_9_BACKEND_NAME + "-suffix"],
        ["triton"],
        ["wrong-backend"],
    ],
)
def test_issue641_nonexact_backend_name_arrays_fail(names: list[str]) -> None:
    historical = _historical_row(names)
    adapted = repair.adapt_frozen_physical_sparse_result_v26_9(historical)
    assert adapted["backend_names_exact"] is False
    assert adapted["pass"] is False


@pytest.mark.parametrize(
    "failed_gate",
    [
        "sparse_route_exercised",
        "coalesced_select_merge_positive",
        "backend_activity_positive",
    ],
)
def test_issue641_exact_v26_9_name_cannot_override_other_subgate_failure(
    failed_gate: str,
) -> None:
    kwargs = {"sparse": True, "coalesced": True, "activity": True}
    mapping = {
        "sparse_route_exercised": "sparse",
        "coalesced_select_merge_positive": "coalesced",
        "backend_activity_positive": "activity",
    }
    kwargs[mapping[failed_gate]] = False
    historical = _historical_row(
        [repair.V26_9_BACKEND_NAME] * 4,
        historical_exact=False,
        **kwargs,
    )
    adapted = repair.adapt_frozen_physical_sparse_result_v26_9(historical)
    assert adapted["backend_names_exact"] is True
    assert adapted["pass"] is False


def test_issue641_rejects_schema_or_dense_masked_credit_drift() -> None:
    missing = _historical_row([repair.V26_9_BACKEND_NAME])
    missing.pop("backend_read_calls")
    with pytest.raises(RuntimeError, match="schema drifted"):
        repair.adapt_frozen_physical_sparse_result_v26_9(missing)

    dense = _historical_row([repair.V26_9_BACKEND_NAME])
    dense["dense_masked_sparse_credit"] = True
    with pytest.raises(RuntimeError, match="dense-masked"):
        repair.adapt_frozen_physical_sparse_result_v26_9(dense)

    bad_names = _historical_row([repair.V26_9_BACKEND_NAME])
    bad_names["backend_names"] = [repair.V26_9_BACKEND_NAME, 7]
    with pytest.raises(RuntimeError, match="backend_names schema"):
        repair.adapt_frozen_physical_sparse_result_v26_9(bad_names)


def test_issue641_wrapper_swaps_once_calls_once_and_restores_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = repair._FROZEN_PHYSICAL_SPARSE_PROOF
    calls = {"supplied": 0}

    def supplied() -> str:
        calls["supplied"] += 1
        assert base._physical_sparse_proof is repair.physical_sparse_proof_v26_9
        return "ok"

    assert base._physical_sparse_proof is original
    assert repair.with_v26_9_physical_sparse_evaluator(supplied) == "ok"
    assert calls == {"supplied": 1}
    assert base._physical_sparse_proof is original


def test_issue641_wrapper_restores_on_exception_and_rejects_preexisting_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = repair._FROZEN_PHYSICAL_SPARSE_PROOF

    def boom() -> None:
        assert base._physical_sparse_proof is repair.physical_sparse_proof_v26_9
        raise ValueError("sentinel")

    with pytest.raises(ValueError, match="sentinel"):
        repair.with_v26_9_physical_sparse_evaluator(boom)
    assert base._physical_sparse_proof is original

    def drifted(*args, **kwargs):
        return {}

    monkeypatch.setattr(base, "_physical_sparse_proof", drifted)
    with pytest.raises(RuntimeError, match="pre-existing frozen evaluator drift"):
        repair.with_v26_9_physical_sparse_evaluator(lambda: None)
    assert base._physical_sparse_proof is drifted


def test_issue641_evaluator_calls_frozen_proof_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = _historical_row(
        [repair.V26_9_BACKEND_NAME] * 4,
        historical_exact=False,
    )
    calls = {"frozen": 0}

    def frozen(model, output):
        calls["frozen"] += 1
        assert model == "model"
        assert output == {"output": True}
        return historical

    monkeypatch.setattr(repair, "_FROZEN_PHYSICAL_SPARSE_PROOF", frozen)
    adapted = repair.physical_sparse_proof_v26_9("model", {"output": True})
    assert calls == {"frozen": 1}
    assert adapted["pass"] is True


def test_issue641_has_no_execution_or_result_write_surface() -> None:
    module = (
        ROOT / "tam_research/aera_v26_9_issue641_physical_sparse_backend_identity_compat.py"
    ).read_text()
    lowered = module.lower()
    forbidden = (
        "import modal",
        "from modal",
        "@app.function",
        "modal.",
        "torch.load(",
        "torch.save(",
        "volume.commit(",
        "write_text(",
        "write_bytes(",
        "open(",
        "subprocess",
        "workflow_dispatch",
        "github.event",
        "gpu=\"",
    )
    assert all(token not in lowered for token in forbidden)


def test_issue641_higher_authorizations_remain_false() -> None:
    protocol = repair.issue641_protocol()
    assert protocol["cpu_only"] is True
    false_keys = (
        "end_to_end_systems_authorized",
        "gpu_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
    )
    assert all(protocol[key] is False for key in false_keys)
