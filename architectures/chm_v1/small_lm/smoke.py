from __future__ import annotations

"""CPU-only implementation smoke for CHM-v1 / EIEM issue #854.

This is deliberately non-scientific.  It uses seed 12345 and refuses the three
reserved scientific seeds.  It touches no dataset, Modal resource, or GPU.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from tam_research.chm_v1_exact_index import ExactEpisodicIndex
from tam_research.chm_v1_small_lm import (
    NON_SCIENTIFIC_SMOKE_SEED,
    CHMV1EIEMLM,
    EpisodicState,
    parameter_accounting,
    parameter_digest,
)


def run() -> dict[str, object]:
    torch.manual_seed(NON_SCIENTIFIC_SMOKE_SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    accounting = parameter_accounting()
    if not accounting["within_preregistered_one_percent"]:
        raise RuntimeError(f"parameter fairness failed: {accounting}")

    rng = np.random.default_rng(NON_SCIENTIFIC_SMOKE_SEED)
    points = rng.normal(size=(257, 32)).astype(np.float32)
    # Force an exact duplicate in separate tree regions to exercise tie handling.
    points[256] = points[0]
    ids = np.arange(1000, 1257, dtype=np.int64)
    ids[0], ids[256] = 2000, 3
    index = ExactEpisodicIndex(points, ids, leaf_size=8)
    queries = np.concatenate((points[[0]], rng.normal(size=(63, 32)).astype(np.float32)), axis=0)
    reads = 0
    flat_reads = 0
    node_visits = 0
    for query in queries:
        flat, indexed = index.assert_exact(query)
        reads += indexed.address_vector_reads
        flat_reads += flat.address_vector_reads
        node_visits += indexed.directory_nodes_visited
    if index.flat_search(points[0]).item_id != 3:
        raise RuntimeError("shared equal-distance insertion-ID tie rule failed")

    model = CHMV1EIEMLM()
    model.train()
    tokens = torch.randint(0, 50_257, (1, 8))
    targets = torch.randint(0, 50_257, (1, 8))
    logits = model.forward_flat_differentiable(tokens)
    loss = F.cross_entropy(logits.reshape(-1, 50_257), targets.reshape(-1))
    loss.backward()
    address_gradient = float(
        model.query_address.weight.grad.abs().sum()
        + model.key_address.weight.grad.abs().sum()
    )
    gate_gradient = float(model.memory_gate_logit.grad.abs().sum())
    if not np.isfinite(address_gradient) or address_gradient <= 0:
        raise RuntimeError("differentiable flat path did not train address projections")
    if not np.isfinite(gate_gradient) or gate_gradient <= 0:
        raise RuntimeError("differentiable flat path did not train memory integration gate")

    model.eval()
    state_a = EpisodicState("smoke-a")
    state_b = EpisodicState("smoke-b")
    before = parameter_digest(model)
    first = torch.randint(0, 50_257, (1, 12))
    _, first_stats = model.forward_session_chunk(first, [state_a], mode="indexed")
    if first_stats.calls != 0 or len(state_a) != 12 or len(state_b) != 0:
        raise RuntimeError("post-query write/session-isolation invariant failed")
    second = torch.randint(0, 50_257, (1, 7))
    flat_logits, flat_stats = model.forward_session_chunk(
        second, [state_a], mode="flat", update_memory=False
    )
    indexed_logits, indexed_stats = model.forward_session_chunk(
        second, [state_a], mode="indexed", update_memory=False
    )
    if indexed_stats.exact_match_rate != 1.0:
        raise RuntimeError("stateful indexed/flat retrieval identity failed")
    if not torch.equal(flat_logits, indexed_logits):
        raise RuntimeError("same-memory flat/indexed logits are not bit-identical")
    if parameter_digest(model) != before:
        raise RuntimeError("inference memory mutated model parameters")

    state_a.reset()
    if len(state_a) != 0 or len(state_b) != 0:
        raise RuntimeError("session reset/isolation failed")

    return {
        "classification": "ZERO_CREDIT_IMPLEMENTATION_SMOKE_ONLY",
        "research_issue": 854,
        "scientific_seed_used": False,
        "smoke_seed": NON_SCIENTIFIC_SMOKE_SEED,
        "gpu_authorized": False,
        "gpu_used": False,
        "parameter_accounting": accounting,
        "exact_index_random_queries": int(len(queries)),
        "exact_index_match_rate": 1.0,
        "exact_index_address_read_fraction": reads / flat_reads,
        "exact_index_directory_nodes_per_query": node_visits / len(queries),
        "address_gradient_l1": address_gradient,
        "gate_gradient_l1": gate_gradient,
        "first_chunk_external_retrieval_calls": first_stats.calls,
        "second_chunk_flat_calls": flat_stats.calls,
        "second_chunk_indexed_calls": indexed_stats.calls,
        "stateful_flat_index_logits_bit_identical": True,
        "base_parameter_nonmutation": True,
        "session_reset_and_isolation": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
