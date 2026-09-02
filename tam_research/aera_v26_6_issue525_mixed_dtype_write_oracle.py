from __future__ import annotations

"""Issue #525 CPU-first mixed-dtype-safe adapter for the frozen WRITE oracle.

This module changes no production backend and runs no experiment.  It preserves the
frozen #488 reference duplicate/newest-wins decisions and stable-compaction path,
while making the materialization inputs evaluable when compute/new dtypes differ
from durable state dtypes.
"""

from typing import Any

import torch

from .aera_hardware_core_v24 import ContextualEpisodicMemoryState
from . import aera_v26_4_ficem_write_probe as historical

RESEARCH_ISSUE = 525
SOURCE_MAIN = "6d5cfddd7b5b9359fb6e7e31c2da3f14c65203f3"
SOURCE_ISSUE519_RUN = 33672232063
SOURCE_ISSUE519_JOB = 100388368044
SOURCE_ISSUE519_ATTEMPT = 1
SOURCE_ISSUE519_RESULT_SHA256 = (
    "b9fba0fca96644ef8db9bc46faf2c73d0c0cc1f1aaac6a321abe2411d3703cd5"
)
SOURCE_ISSUE519_CANDIDATE_BLOB = "d45c262314a0b4691f26812a279937a225043ad9"
SOURCE_ISSUE519_PROBE_BLOB = "ec22807434192f58e292bffc3de9828be2b44272"
SOURCE_ISSUE522_RUN = 33675476637
SOURCE_ISSUE522_JOB = 100398984660
SOURCE_ISSUE522_ATTEMPT = 1
SOURCE_ISSUE522_EXCEPTION = "scatter(): Expected self.dtype to be equal to src.dtype"
SOURCE_ISSUE522_DIRECT_EXCEPTION_COUNT = 224
SOURCE_ISSUE522_EDGE_EXCEPTION_COUNT = 16


def issue525_oracle_protocol() -> dict[str, Any]:
    return {
        "version": "aera-v26.6-issue525-mixed-dtype-write-oracle-adapter",
        "research_issue": RESEARCH_ISSUE,
        "source_main": SOURCE_MAIN,
        "source_issue519_run": SOURCE_ISSUE519_RUN,
        "source_issue519_job": SOURCE_ISSUE519_JOB,
        "source_issue519_attempt": SOURCE_ISSUE519_ATTEMPT,
        "source_issue519_result_sha256": SOURCE_ISSUE519_RESULT_SHA256,
        "source_issue522_run": SOURCE_ISSUE522_RUN,
        "source_issue522_job": SOURCE_ISSUE522_JOB,
        "source_issue522_attempt": SOURCE_ISSUE522_ATTEMPT,
        "source_issue522_exception": SOURCE_ISSUE522_EXCEPTION,
        "durable_field_conversion": {
            "new_keys": "state.keys.dtype",
            "new_values": "state.values.dtype",
            "new_strengths": "state.strengths.dtype",
        },
        "similarity_inputs_changed": False,
        "validity_inputs_changed": False,
        "duplicate_decisions_changed": False,
        "newest_wins_changed": False,
        "stable_order_changed": False,
        "capacity_semantics_changed": False,
        "production_backend_changed": False,
        "gpu_authorized": False,
        "scientific_seed_authorized": False,
        "end_to_end_systems_authorized": False,
        "architecture_freeze_authorized": False,
        "independent_replication_credit": False,
        "100m_authorized": False,
        "breakthrough_proven": False,
    }


def durable_mixed_dtype_reference_tail(
    memory: Any,
    inputs: historical.TailInputs,
) -> ContextualEpisodicMemoryState:
    """Evaluate frozen #488 tail semantics in each durable field's destination dtype.

    Duplicate and keep-old decisions are still computed by the frozen historical
    `_reference_tail` from the original similarity and validity tensors.  Only the
    three new materialization fields are copied into the corresponding durable state
    dtype before the historical stable-compaction scatter is entered.  State tensors
    and all caller-owned inputs remain untouched.
    """

    durable_inputs = historical.TailInputs(
        incoming_similarity=inputs.incoming_similarity,
        old_similarity=inputs.old_similarity,
        new_keys=inputs.new_keys.to(dtype=inputs.state.keys.dtype),
        new_values=inputs.new_values.to(dtype=inputs.state.values.dtype),
        new_strengths=inputs.new_strengths.to(dtype=inputs.state.strengths.dtype),
        new_valid=inputs.new_valid,
        state=inputs.state,
    )
    return historical._reference_tail(memory, durable_inputs)
