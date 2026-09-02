from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from tam_research.aera_hardware_core_v24 import ContextualEpisodicMemoryState
from tam_research import aera_v26_4_ficem_write_probe as historical
from tam_research import aera_v26_5_issue514_ficem_write_mixed_dtype_probe as frozen
from tam_research.aera_v26_6_issue525_mixed_dtype_write_oracle import (
    SOURCE_ISSUE519_ATTEMPT,
    SOURCE_ISSUE519_CANDIDATE_BLOB,
    SOURCE_ISSUE519_JOB,
    SOURCE_ISSUE519_PROBE_BLOB,
    SOURCE_ISSUE519_RESULT_SHA256,
    SOURCE_ISSUE519_RUN,
    SOURCE_ISSUE522_ATTEMPT,
    SOURCE_ISSUE522_DIRECT_EXCEPTION_COUNT,
    SOURCE_ISSUE522_EDGE_EXCEPTION_COUNT,
    SOURCE_ISSUE522_EXCEPTION,
    SOURCE_ISSUE522_JOB,
    SOURCE_ISSUE522_RUN,
    SOURCE_MAIN,
    durable_mixed_dtype_reference_tail,
    issue525_oracle_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "tam_research" / "aera_v26_6_issue525_mixed_dtype_write_oracle.py"
HISTORICAL_REFERENCE_PATH = ROOT / "tam_research" / "aera_v26_4_ficem_write_probe.py"
STABLE_COMPACTION_PATH = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"

EXPECTED_SOURCE_MAIN = "6d5cfddd7b5b9359fb6e7e31c2da3f14c65203f3"
EXPECTED_HISTORICAL_REFERENCE_BLOB = "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
EXPECTED_STABLE_COMPACTION_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
EXPECTED_RESULT_SHA = "b9fba0fca96644ef8db9bc46faf2c73d0c0cc1f1aaac6a321abe2411d3703cd5"
EXPECTED_EXCEPTION = "scatter(): Expected self.dtype to be equal to src.dtype"
FROZEN_PASS_MASKS = (
    0, 1, 2, 3, 36, 37, 38, 39, 72, 73, 74, 75, 108, 109, 110, 111,
    144, 145, 146, 147, 180, 181, 182, 183, 216, 217, 218, 219,
    252, 253, 254, 255,
)


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _assert_state_exact(a: ContextualEpisodicMemoryState, b: ContextualEpisodicMemoryState) -> None:
    assert a.keys.dtype is b.keys.dtype
    assert a.values.dtype is b.values.dtype
    assert a.strengths.dtype is b.strengths.dtype
    assert torch.equal(a.keys, b.keys)
    assert torch.equal(a.values, b.values)
    assert torch.equal(a.strengths, b.strengths)
    assert torch.equal(a.valid, b.valid)


def _independent_durable_expectation(inputs: historical.TailInputs) -> ContextualEpisodicMemoryState:
    """Loop-based stable materialization independent of the scatter implementation."""
    batch, k_count, _ = inputs.new_keys.shape
    capacity = inputs.state.keys.size(1)
    position = torch.arange(k_count, device=inputs.new_keys.device)
    later = position[None, :, None] < position[None, None, :]
    shadowed = (
        inputs.incoming_similarity.ge(historical.DUPLICATE_THRESHOLD)
        & inputs.new_valid[:, :, None]
        & inputs.new_valid[:, None, :]
        & later
    ).any(dim=2)
    surviving_new = inputs.new_valid & ~shadowed
    duplicate_old = (
        inputs.old_similarity.ge(historical.DUPLICATE_THRESHOLD)
        & surviving_new[:, :, None]
        & inputs.state.valid[:, None, :]
    ).any(dim=1)
    keep_old = inputs.state.valid & ~duplicate_old

    new_keys = inputs.new_keys.flip(1)
    new_values = inputs.new_values.flip(1)
    new_strengths = inputs.new_strengths.flip(1)
    new_valid = surviving_new.flip(1)

    out_keys = torch.zeros_like(inputs.state.keys)
    out_values = torch.zeros_like(inputs.state.values)
    out_strengths = torch.zeros_like(inputs.state.strengths)
    out_valid = torch.zeros_like(inputs.state.valid)

    for batch_index in range(batch):
        sources: list[tuple[bool, int, bool]] = []
        for slot in range(k_count):
            sources.append((True, slot, bool(new_valid[batch_index, slot])))
        for slot in range(capacity):
            sources.append((False, slot, bool(keep_old[batch_index, slot])))
        ordered = [source for source in sources if source[2]] + [
            source for source in sources if not source[2]
        ]
        retained = ordered[:capacity]
        valid_count = min(sum(int(source[2]) for source in sources), capacity)
        out_valid[batch_index, :valid_count] = True

        for destination, (from_new, slot, _is_valid) in enumerate(retained):
            if from_new:
                out_keys[batch_index, destination] = new_keys[batch_index, slot].to(
                    dtype=out_keys.dtype
                )
                out_values[batch_index, destination] = new_values[batch_index, slot].to(
                    dtype=out_values.dtype
                )
                out_strengths[batch_index, destination] = new_strengths[
                    batch_index, slot
                ].to(dtype=out_strengths.dtype)
            else:
                out_keys[batch_index, destination] = inputs.state.keys[batch_index, slot]
                out_values[batch_index, destination] = inputs.state.values[batch_index, slot]
                out_strengths[batch_index, destination] = inputs.state.strengths[
                    batch_index, slot
                ]

    return ContextualEpisodicMemoryState(
        keys=out_keys,
        values=out_values,
        strengths=out_strengths,
        valid=out_valid,
    )


def test_issue525_freezes_localization_and_historical_oracle_blobs() -> None:
    assert SOURCE_MAIN == EXPECTED_SOURCE_MAIN
    assert SOURCE_ISSUE519_RUN == 33672232063
    assert SOURCE_ISSUE519_JOB == 100388368044
    assert SOURCE_ISSUE519_ATTEMPT == 1
    assert SOURCE_ISSUE519_RESULT_SHA256 == EXPECTED_RESULT_SHA
    assert SOURCE_ISSUE519_CANDIDATE_BLOB == "d45c262314a0b4691f26812a279937a225043ad9"
    assert SOURCE_ISSUE519_PROBE_BLOB == "ec22807434192f58e292bffc3de9828be2b44272"
    assert SOURCE_ISSUE522_RUN == 33675476637
    assert SOURCE_ISSUE522_JOB == 100398984660
    assert SOURCE_ISSUE522_ATTEMPT == 1
    assert SOURCE_ISSUE522_EXCEPTION == EXPECTED_EXCEPTION
    assert SOURCE_ISSUE522_DIRECT_EXCEPTION_COUNT == 224
    assert SOURCE_ISSUE522_EDGE_EXCEPTION_COUNT == 16
    assert _blob(HISTORICAL_REFERENCE_PATH) == EXPECTED_HISTORICAL_REFERENCE_BLOB
    assert _blob(STABLE_COMPACTION_PATH) == EXPECTED_STABLE_COMPACTION_BLOB


def test_issue525_adapter_scope_is_oracle_only_and_cpu_pure() -> None:
    source = ORACLE_PATH.read_text()
    assert "historical._reference_tail(memory, durable_inputs)" in source
    assert "inputs.new_keys.to(dtype=inputs.state.keys.dtype)" in source
    assert "inputs.new_values.to(dtype=inputs.state.values.dtype)" in source
    assert "inputs.new_strengths.to(dtype=inputs.state.strengths.dtype)" in source
    assert "aera_hardware_core_v26_6_ficem_write_materialize_cast" not in source
    assert "_candidate_tail" not in source
    assert "torch.cuda" not in source
    assert "gpu=" not in source
    protocol = issue525_oracle_protocol()
    assert protocol["production_backend_changed"] is False
    assert protocol["gpu_authorized"] is False
    assert protocol["scientific_seed_authorized"] is False
    assert protocol["end_to_end_systems_authorized"] is False
    assert protocol["100m_authorized"] is False
    assert protocol["breakthrough_proven"] is False


def test_issue525_all_256_masks_are_evaluable_and_match_independent_durable_semantics() -> None:
    device = torch.device("cpu")
    memory = frozen._build_memory(device)
    base = frozen._base_matrix_inputs(device)

    observed_same_pair: list[int] = []
    for mask in frozen.MATRIX_MASKS:
        inputs = frozen._cast_tail_fields(base, frozen._mask_dtypes(mask))
        before = frozen._clone_tail(inputs)
        repaired = durable_mixed_dtype_reference_tail(memory, inputs)
        expected = _independent_durable_expectation(inputs)

        assert repaired.keys.dtype is inputs.state.keys.dtype
        assert repaired.values.dtype is inputs.state.values.dtype
        assert repaired.strengths.dtype is inputs.state.strengths.dtype
        _assert_state_exact(repaired, expected)
        assert frozen._tail_sources_unchanged(inputs, before)

        dtypes = frozen._mask_dtypes(mask)
        same_pair = dtypes[2] is dtypes[5] and dtypes[3] is dtypes[6] and dtypes[4] is dtypes[7]
        if same_pair:
            historical_result = historical._reference_tail(memory, inputs)
            _assert_state_exact(repaired, historical_result)
            observed_same_pair.append(mask)

    assert tuple(observed_same_pair) == FROZEN_PASS_MASKS


def test_issue525_all_32_frozen_edge_rows_are_evaluable_and_exact() -> None:
    device = torch.device("cpu")
    memory = frozen._build_memory(device)
    observed = 0

    for fixture_name in frozen.EDGE_FIXTURES:
        base = historical.make_edge_fixture(
            fixture_name,
            dtype_name="float32",
            device=device,
        )
        for layout in frozen.EDGE_LAYOUTS:
            inputs = frozen._cast_tail_fields(base, frozen._layout_dtypes(layout))
            before = frozen._clone_tail(inputs)
            repaired = durable_mixed_dtype_reference_tail(memory, inputs)
            expected = _independent_durable_expectation(inputs)
            _assert_state_exact(repaired, expected)
            assert repaired.keys.dtype is inputs.state.keys.dtype
            assert repaired.values.dtype is inputs.state.values.dtype
            assert repaired.strengths.dtype is inputs.state.strengths.dtype
            assert frozen._tail_sources_unchanged(inputs, before)
            observed += 1

    assert observed == 32
