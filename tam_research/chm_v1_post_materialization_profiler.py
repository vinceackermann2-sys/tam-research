from __future__ import annotations

"""Zero-credit query-kernel profiler for CHM-v1 successor issue #932.

This module is additive systems instrumentation only. It uses the merged
contiguous-state prototype from #930 and the frozen ExactEpisodicIndex without
changing either implementation. No scientific seeds, GPU/Modal execution, or
scientific result namespaces are reachable from this profiler.
"""

from dataclasses import dataclass
import time
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .chm_v1_contiguous_state_prototype import (
    ContiguousEpisodicStatePrototype,
    prototype_accounting,
)

SCIENTIFIC_SEEDS = (8611, 8612, 8613)
SYNTHETIC_PROFILE_SEED = 932_001
QueryGeometry = Literal["exact", "near", "random"]


@dataclass(frozen=True)
class QueryBatch:
    queries: torch.Tensor
    anchor_positions: tuple[int, ...]
    geometry: QueryGeometry
    noise_std: float


def validate_profile_seed(seed: int) -> int:
    seed = int(seed)
    if seed in SCIENTIFIC_SEEDS:
        raise RuntimeError(f"scientific seed refused by zero-credit profiler: {seed}")
    return seed


def _normalized_random(
    rows: int,
    width: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return F.normalize(
        torch.randn(rows, width, generator=generator, device="cpu"),
        dim=-1,
    ).float()


def synthetic_state_payload(
    *,
    session_length: int,
    key_width: int = 32,
    value_width: int = 256,
    seed: int = SYNTHETIC_PROFILE_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed = validate_profile_seed(seed)
    if session_length < 1:
        raise ValueError("session_length must be positive")
    if key_width < 1 or value_width < 1:
        raise ValueError("key/value widths must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = _normalized_random(session_length, key_width, generator=generator)
    values = torch.randn(
        session_length,
        value_width,
        generator=generator,
        device="cpu",
    ).float()
    return keys, values


def build_query_batch(
    keys: torch.Tensor,
    *,
    query_count: int,
    geometry: QueryGeometry,
    noise_std: float = 0.05,
    seed: int = SYNTHETIC_PROFILE_SEED + 1,
) -> QueryBatch:
    seed = validate_profile_seed(seed)
    if keys.ndim != 2 or keys.shape[0] < 1:
        raise ValueError("keys must be a non-empty [items, width] CPU matrix")
    if keys.device.type != "cpu":
        raise RuntimeError("zero-credit profiler accepts CPU tensors only")
    if query_count < 1 or query_count > keys.shape[0]:
        raise ValueError("query_count must be in [1, session_length]")
    if geometry not in ("exact", "near", "random"):
        raise ValueError(f"unknown query geometry {geometry!r}")
    if geometry == "near" and noise_std <= 0.0:
        raise ValueError("near geometry requires positive noise_std")
    if geometry != "near" and noise_std < 0.0:
        raise ValueError("noise_std cannot be negative")

    positions = np.linspace(
        0,
        keys.shape[0] - 1,
        num=query_count,
        dtype=np.int64,
    )
    anchors = keys[torch.as_tensor(positions, dtype=torch.long)]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if geometry == "exact":
        queries = anchors.clone()
        used_noise = 0.0
    elif geometry == "near":
        noise = torch.randn(
            anchors.shape,
            generator=generator,
            device="cpu",
            dtype=anchors.dtype,
        )
        queries = F.normalize(anchors + float(noise_std) * noise, dim=-1).float()
        used_noise = float(noise_std)
    else:
        queries = _normalized_random(
            query_count,
            keys.shape[1],
            generator=generator,
        )
        used_noise = 0.0

    return QueryBatch(
        queries=queries,
        anchor_positions=tuple(int(x) for x in positions.tolist()),
        geometry=geometry,
        noise_std=used_noise,
    )


def profile_query_kernel_once(
    *,
    session_length: int,
    query_count: int,
    geometry: QueryGeometry,
    noise_std: float = 0.05,
    key_width: int = 32,
    value_width: int = 256,
    seed: int = SYNTHETIC_PROFILE_SEED,
) -> dict[str, Any]:
    """Profile one deterministic CPU configuration.

    Timings are descriptive outputs only. Tests must gate exactness and
    accounting, never timing magnitudes or ratios.
    """

    seed = validate_profile_seed(seed)
    keys, values = synthetic_state_payload(
        session_length=session_length,
        key_width=key_width,
        value_width=value_width,
        seed=seed,
    )
    batch = build_query_batch(
        keys,
        query_count=query_count,
        geometry=geometry,
        noise_std=noise_std,
        seed=seed + 1,
    )

    state = ContiguousEpisodicStatePrototype(f"kernel-profile-{seed}")
    state.write(keys, values)
    index = state.index()

    flat_ns = 0
    indexed_ns = 0
    wrapper_ns = 0
    indexed_reads = 0
    directory_nodes = 0
    exact_matches = 0
    wrapper_matches = 0
    flat_item_ids: list[int] = []
    indexed_item_ids: list[int] = []

    for query in batch.queries:
        query_np = query.detach().float().cpu().numpy().astype(np.float32, copy=False)

        started = time.perf_counter_ns()
        flat = index.flat_search(query_np)
        flat_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        indexed = index.indexed_search(query_np)
        indexed_ns += time.perf_counter_ns() - started

        flat_item_ids.append(int(flat.item_id))
        indexed_item_ids.append(int(indexed.item_id))
        exact = flat.item_id == indexed.item_id and flat.position == indexed.position
        exact_matches += int(exact)
        if not exact:
            raise AssertionError(
                f"flat/indexed mismatch: flat={flat} indexed={indexed}"
            )

        indexed_reads += int(indexed.address_vector_reads)
        directory_nodes += int(indexed.directory_nodes_visited)

        started = time.perf_counter_ns()
        _, wrapped, _, _, _, _ = state.retrieve(
            query,
            mode="indexed",
            verify_indexed_exactness=False,
        )
        wrapper_ns += time.perf_counter_ns() - started
        wrapper_match = (
            wrapped.item_id == indexed.item_id
            and wrapped.position == indexed.position
        )
        wrapper_matches += int(wrapper_match)
        if not wrapper_match:
            raise AssertionError(
                f"wrapper/index mismatch: wrapper={wrapped} indexed={indexed}"
            )

    flat_reads = int(session_length * query_count)
    accounting = prototype_accounting(state)
    if accounting["warm_retrieve_full_state_materializations"] != 0:
        raise AssertionError("contiguous prototype unexpectedly materialized full state")

    return {
        "measurement_kind": "zero_credit_systems_diagnostic_cpu",
        "scientific_credit": False,
        "paid_compute": False,
        "modal_trigger": False,
        "device": "cpu",
        "seed": seed,
        "session_length": int(session_length),
        "query_count": int(query_count),
        "geometry": geometry,
        "noise_std": float(batch.noise_std),
        "exact_match_count": int(exact_matches),
        "wrapper_match_count": int(wrapper_matches),
        "flat_item_ids": flat_item_ids,
        "indexed_item_ids": indexed_item_ids,
        "flat_address_vector_reads": flat_reads,
        "indexed_address_vector_reads": int(indexed_reads),
        "indexed_address_read_fraction": float(indexed_reads / max(flat_reads, 1)),
        "directory_nodes_visited": int(directory_nodes),
        "directory_nodes_per_query": float(directory_nodes / query_count),
        "flat_query_ns": int(flat_ns),
        "indexed_query_ns": int(indexed_ns),
        "contiguous_wrapper_ns": int(wrapper_ns),
        "indexed_over_flat_time_ratio": float(indexed_ns / max(flat_ns, 1)),
        "lower_bound_call_count": None,
        "lower_bound_measurement": (
            "not instrumented because frozen ExactEpisodicIndex is intentionally unchanged"
        ),
        "accounting": accounting,
        "timing_is_ci_gate": False,
        "interpretation_ceiling": (
            "Zero-credit software attribution only; not scientific evidence and not "
            "authority for seed 8612/8613 or any paid/GPU execution."
        ),
    }


def deterministic_signature(result: dict[str, Any]) -> dict[str, Any]:
    """Return the non-timing fields suitable for deterministic CI assertions."""

    excluded = {
        "flat_query_ns",
        "indexed_query_ns",
        "contiguous_wrapper_ns",
        "indexed_over_flat_time_ratio",
    }
    return {key: value for key, value in result.items() if key not in excluded}
