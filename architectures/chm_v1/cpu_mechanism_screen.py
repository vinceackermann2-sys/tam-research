from __future__ import annotations

"""CHM-v1 exploratory CPU/mechanical falsification.

Candidate: Exact Indexable Episodic Memory (EIEM).
This is not a language-model or scientific GPU result. It isolates the mechanism:
learn query/key address representations on 64-event retrieval, preserve every episodic
key/value exactly, and use a zero-parameter exact branch-and-bound hierarchy to
reduce memory reads without changing the flat nearest-neighbor answer.

The hierarchy never averages/compresses episodic evidence. Internal nodes contain
only bounding boxes over address vectors. Search is exact: a subtree is pruned only
when its geometric lower bound is worse than the current best exact candidate.

Exploratory seed 60430 is single-use and must not become a scientific seed.
"""

import heapq
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 60430
N_A = 64
N_B = 64
N_IDENTITIES = N_A * N_B
EMBED_DIM = 8
ADDRESS_DIM = 8
TRAIN_EVENTS = 64
TRAIN_BATCH = 128
TRAIN_STEPS = 900
TEMPERATURE = 20.0
LEAF_SIZE = 16
EVAL_SIZES = (64, 256, 1024)
EVAL_QUERIES = 1024

# Hold out 25% of compositional identities. Every component still appears in training,
# but evaluation pairs are never sampled by the retrieval-training stream.
TRAIN_ID_LIST = [i for i in range(N_IDENTITIES) if (((i // N_B) * 17 + (i % N_B) * 31) % 4) != 0]
EVAL_ID_LIST = [i for i in range(N_IDENTITIES) if (((i // N_B) * 17 + (i % N_B) * 31) % 4) == 0]


def _ids_to_parts(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ids // N_B, ids % N_B


class AddressEncoder(nn.Module):
    """Compositional memory/query encoders trained only on 64-way retrieval."""

    def __init__(self) -> None:
        super().__init__()
        self.mem_a = nn.Embedding(N_A, EMBED_DIM)
        self.mem_b = nn.Embedding(N_B, EMBED_DIM)
        self.query_a = nn.Embedding(N_A, EMBED_DIM)
        self.query_b = nn.Embedding(N_B, EMBED_DIM)
        self.mem_proj = nn.Linear(2 * EMBED_DIM, ADDRESS_DIM, bias=False)
        self.query_proj = nn.Linear(2 * EMBED_DIM, ADDRESS_DIM, bias=False)

    def memory(self, ids: torch.Tensor) -> torch.Tensor:
        a, b = _ids_to_parts(ids)
        x = torch.cat((self.mem_a(a), self.mem_b(b)), dim=-1)
        return F.normalize(self.mem_proj(x), dim=-1)

    def query(self, ids: torch.Tensor) -> torch.Tensor:
        a, b = _ids_to_parts(ids)
        x = torch.cat((self.query_a(a), self.query_b(b)), dim=-1)
        return F.normalize(self.query_proj(x), dim=-1)


def _sample_unique_rows(batch: int, width: int, generator: torch.Generator) -> torch.Tensor:
    pool = torch.tensor(TRAIN_ID_LIST, dtype=torch.long)
    scores = torch.rand(batch, len(TRAIN_ID_LIST), generator=generator)
    positions = scores.topk(width, dim=1, largest=False).indices
    return pool[positions]


def train_encoder(model: AddressEncoder) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    started = time.perf_counter()
    last_loss = float("nan")
    last_acc = float("nan")
    model.train()
    for _ in range(TRAIN_STEPS):
        memory_ids = _sample_unique_rows(TRAIN_BATCH, TRAIN_EVENTS, generator)
        target_slot = torch.randint(TRAIN_EVENTS, (TRAIN_BATCH,), generator=generator)
        query_ids = memory_ids[torch.arange(TRAIN_BATCH), target_slot]
        keys = model.memory(memory_ids)
        queries = model.query(query_ids)
        logits = torch.einsum("bd,bnd->bn", queries, keys) * TEMPERATURE
        loss = F.cross_entropy(logits, target_slot)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        last_acc = float((logits.argmax(-1) == target_slot).float().mean())
    return {
        "seconds": time.perf_counter() - started,
        "last_loss": last_loss,
        "last_batch_accuracy": last_acc,
    }


@dataclass
class KDNode:
    lo: np.ndarray
    hi: np.ndarray
    idxs: np.ndarray | None = None
    left: "KDNode | None" = None
    right: "KDNode | None" = None


def _build_tree(points: np.ndarray, idxs: np.ndarray) -> KDNode:
    block = points[idxs]
    lo = block.min(axis=0)
    hi = block.max(axis=0)
    if len(idxs) <= LEAF_SIZE:
        return KDNode(lo=lo, hi=hi, idxs=idxs.copy())
    axis = int(np.argmax(hi - lo))
    order = np.argsort(block[:, axis], kind="mergesort")
    ordered = idxs[order]
    mid = len(ordered) // 2
    return KDNode(
        lo=lo,
        hi=hi,
        left=_build_tree(points, ordered[:mid]),
        right=_build_tree(points, ordered[mid:]),
    )


def _bbox_lower_bound_sq(query: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(lo - query, query - hi))
    return float(delta @ delta)


def exact_tree_search(points: np.ndarray, root: KDNode, query: np.ndarray) -> tuple[int, int, int]:
    """Return exact NN index, exact-key comparisons, and directory-node visits."""
    best_dist = float("inf")
    best_idx = -1
    key_comparisons = 0
    node_visits = 0
    queue: list[tuple[float, int, KDNode]] = [
        (_bbox_lower_bound_sq(query, root.lo, root.hi), id(root), root)
    ]
    while queue:
        lower_bound, _, node = heapq.heappop(queue)
        if lower_bound > best_dist:
            break
        node_visits += 1
        if node.idxs is not None:
            block = points[node.idxs]
            dist = ((block - query) ** 2).sum(axis=1)
            key_comparisons += len(node.idxs)
            local = int(np.argmin(dist))
            value = float(dist[local])
            if value < best_dist:
                best_dist = value
                best_idx = int(node.idxs[local])
            continue
        assert node.left is not None and node.right is not None
        for child in (node.left, node.right):
            lb = _bbox_lower_bound_sq(query, child.lo, child.hi)
            if lb <= best_dist:
                heapq.heappush(queue, (lb, id(child), child))
    if best_idx < 0:
        raise RuntimeError("exact tree search returned no candidate")
    return best_idx, key_comparisons, node_visits


def _memory_ids(n: int, seed: int) -> np.ndarray:
    if n > len(EVAL_ID_LIST):
        raise ValueError(f"requested {n} held-out identities but only {len(EVAL_ID_LIST)} exist")
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(EVAL_ID_LIST, dtype=np.int64), size=n, replace=False)


@torch.no_grad()
def _representations(model: AddressEncoder, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.tensor(ids, dtype=torch.long)
    return model.memory(tensor).cpu().numpy(), model.query(tensor).cpu().numpy()


def _flat_top1(keys: np.ndarray, query: np.ndarray) -> int:
    # Unit vectors => max cosine == min squared Euclidean distance.
    return int(np.argmax(keys @ query))


@torch.no_grad()
def evaluate_rare_fact(model: AddressEncoder, n: int) -> dict[str, float | int]:
    ids = _memory_ids(n, SEED + 1000 + n)
    keys, queries = _representations(model, ids)
    tree = _build_tree(keys, np.arange(n, dtype=np.int64))
    sample = min(EVAL_QUERIES, n)
    positions = np.random.default_rng(SEED + 2000 + n).choice(n, size=sample, replace=False)

    flat_correct = 0
    tree_correct = 0
    exact_match = 0
    key_comparisons = 0
    node_visits = 0
    for pos in positions:
        q = queries[pos]
        flat = _flat_top1(keys, q)
        tree_idx, comps, visits = exact_tree_search(keys, tree, q)
        flat_correct += int(flat == pos)
        tree_correct += int(tree_idx == pos)
        exact_match += int(tree_idx == flat)
        key_comparisons += comps
        node_visits += visits

    vector_reads = key_comparisons + 2 * node_visits
    flat_vector_reads = sample * n
    return {
        "queries": sample,
        "flat_accuracy": flat_correct / sample,
        "indexed_accuracy": tree_correct / sample,
        "indexed_matches_flat": exact_match / sample,
        "mean_key_comparisons": key_comparisons / sample,
        "mean_directory_nodes": node_visits / sample,
        "mean_estimated_vector_reads": vector_reads / sample,
        "fraction_flat_vector_reads": vector_reads / flat_vector_reads,
    }


@torch.no_grad()
def evaluate_overwrite(model: AddressEncoder, n_unique: int, samples: int = 1024) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED + 3000 + n_unique)
    ids = _memory_ids(n_unique, SEED + 3100 + n_unique)
    keys, queries = _representations(model, ids)

    # Dynamic writes are keyed only by exact learned address bytes, never by latent ID.
    latest: dict[bytes, tuple[int, int]] = {}
    stale_values: dict[bytes, set[int]] = {}
    for idx, vector in enumerate(keys):
        value = int(rng.integers(1_000_000))
        latest[vector.tobytes()] = (idx, value)
        stale_values[vector.tobytes()] = set()

    target_positions = rng.integers(0, n_unique, size=samples)
    expected: list[int] = []
    stale_for_query: list[set[int]] = []
    for pos in target_positions:
        fingerprint = keys[pos].tobytes()
        repeats = int(rng.integers(2, 5))
        for _ in range(repeats):
            old = latest[fingerprint][1]
            stale_values[fingerprint].add(old)
            latest[fingerprint] = (int(pos), int(rng.integers(1_000_000)))
        expected.append(0)  # filled from the final memory state after all writes
        stale_for_query.append(set())

    for qn, pos in enumerate(target_positions):
        fingerprint = keys[pos].tobytes()
        expected[qn] = latest[fingerprint][1]
        stale_for_query[qn] = set(stale_values[fingerprint])

    # Retrieval index stores one exact key vector with the current value: overwrite replaces,
    # rather than adding another lossy trace.
    tree = _build_tree(keys, np.arange(n_unique, dtype=np.int64))
    flat_correct = tree_correct = exact_match = stale_errors = 0
    key_comparisons = node_visits = 0
    for qn, pos in enumerate(target_positions):
        q = queries[pos]
        flat = _flat_top1(keys, q)
        tree_idx, comps, visits = exact_tree_search(keys, tree, q)
        flat_value = latest[keys[flat].tobytes()][1]
        tree_value = latest[keys[tree_idx].tobytes()][1]
        flat_correct += int(flat_value == expected[qn])
        tree_correct += int(tree_value == expected[qn])
        exact_match += int(tree_idx == flat)
        stale_errors += int(tree_value in stale_for_query[qn])
        key_comparisons += comps
        node_visits += visits

    vector_reads = key_comparisons + 2 * node_visits
    return {
        "queries": samples,
        "flat_accuracy": flat_correct / samples,
        "indexed_accuracy": tree_correct / samples,
        "indexed_matches_flat": exact_match / samples,
        "indexed_stale_error": stale_errors / samples,
        "mean_estimated_vector_reads": vector_reads / samples,
        "fraction_flat_vector_reads": vector_reads / (samples * n_unique),
    }


@torch.no_grad()
def evaluate_multihop(model: AddressEncoder, n: int, samples: int = 1024) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED + 4000 + n)
    ids = _memory_ids(n, SEED + 4100 + n)
    keys, queries = _representations(model, ids)
    tree = _build_tree(keys, np.arange(n, dtype=np.int64))
    next_pos = rng.permutation(n)
    chosen = rng.integers(0, n, size=samples)
    flat_correct = tree_correct = exact_match = 0
    key_comparisons = node_visits = 0
    for start in chosen:
        p1f = _flat_top1(keys, queries[start])
        midf = int(next_pos[p1f])
        p2f = _flat_top1(keys, queries[midf])
        answer_f = int(next_pos[p2f])

        p1, c1, v1 = exact_tree_search(keys, tree, queries[start])
        mid = int(next_pos[p1])
        p2, c2, v2 = exact_tree_search(keys, tree, queries[mid])
        answer = int(next_pos[p2])

        expected_mid = int(next_pos[start])
        expected = int(next_pos[expected_mid])
        flat_correct += int(answer_f == expected)
        tree_correct += int(answer == expected)
        exact_match += int(answer == answer_f)
        key_comparisons += c1 + c2
        node_visits += v1 + v2

    vector_reads = key_comparisons + 2 * node_visits
    return {
        "queries": samples,
        "hops_per_query": 2,
        "flat_accuracy": flat_correct / samples,
        "indexed_accuracy": tree_correct / samples,
        "indexed_matches_flat": exact_match / samples,
        "mean_estimated_vector_reads": vector_reads / samples,
        "fraction_flat_vector_reads": vector_reads / (samples * n * 2),
    }


@torch.no_grad()
def untrained_ablation(n: int = 1024) -> dict[str, float]:
    torch.manual_seed(SEED + 99)
    raw = AddressEncoder()
    ids = _memory_ids(n, SEED + 5000)
    keys, queries = _representations(raw, ids)
    pred = np.argmax(queries @ keys.T, axis=1)
    return {"flat_accuracy": float((pred == np.arange(n)).mean())}


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = AddressEncoder()
    train_result = train_encoder(model)
    result = {
        "classification": "EXPLORATORY_CPU_MECHANISM_SCREEN_ONLY",
        "candidate": "CHM-v1 / Exact Indexable Episodic Memory (EIEM)",
        "seed": SEED,
        "scientific_seed_authorized": False,
        "gpu_authorized": False,
        "train_horizon_events": TRAIN_EVENTS,
        "train": train_result,
        "rare_fact": {},
        "overwrite": {},
        "multihop": {},
        "ablations": {"untrained_address_encoder": untrained_ablation()},
    }
    for n in EVAL_SIZES:
        result["rare_fact"][str(n)] = evaluate_rare_fact(model, n)
    for n in (256, 1024):
        result["overwrite"][str(n)] = evaluate_overwrite(model, n)
        result["multihop"][str(n)] = evaluate_multihop(model, n)

    r1024 = result["rare_fact"]["1024"]
    o1024 = result["overwrite"]["1024"]
    m1024 = result["multihop"]["1024"]
    pass_gate = (
        r1024["indexed_matches_flat"] == 1.0
        and r1024["indexed_accuracy"] >= 0.95
        and r1024["fraction_flat_vector_reads"] <= 0.15
        and o1024["indexed_accuracy"] >= 0.95
        and o1024["indexed_stale_error"] <= 0.02
        and m1024["indexed_accuracy"] >= 0.90
        and m1024["fraction_flat_vector_reads"] <= 0.15
    )
    result["decision"] = (
        "PASS_MECHANISM_ONLY_PREREGISTER_SMALL_LM_TEST" if pass_gate
        else "FAIL_DO_NOT_SPEND_GPU"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
