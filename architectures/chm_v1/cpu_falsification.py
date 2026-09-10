from __future__ import annotations

"""CHM-v1 exploratory CPU falsification.

CHM-v1 keeps the exact episodic substrate and makes only the *routing* sparse.
It trains a dual semantic address encoder, builds multiple random-hyperplane
routes over exact event embeddings, probes low-margin neighboring buckets, and
performs exact similarity scoring only inside the routed candidate set.

The flat baseline shares the exact same trained encoder and scoring rule but
scores every event. This file is an exploratory mechanism test, not an LM
training result and not evidence of a breakthrough.
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 61941
LATENT_DIM = 32
OBS_DIM = 48
EMBED_DIM = 32
N_KEYS = 4096
TRAIN_STEPS = 800
TRAIN_BATCH = 256
NOISE_STD = 0.05
N_TABLES = 12
PROBES_PER_TABLE = 3
CLUSTER_THRESHOLD = 0.80


class DualAddressEncoder(nn.Module):
    """Learn query/event views of the same latent address in one metric space."""

    def __init__(self) -> None:
        super().__init__()
        self.memory_proj = nn.Linear(OBS_DIM, EMBED_DIM, bias=False)
        self.query_proj = nn.Linear(OBS_DIM, EMBED_DIM, bias=False)

    def memory(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.memory_proj(x), dim=-1)

    def query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_proj(x), dim=-1)


@dataclass
class World:
    latent: torch.Tensor
    memory_view: torch.Tensor
    query_view: torch.Tensor


def make_world(seed: int) -> World:
    g = torch.Generator().manual_seed(seed)
    latent = F.normalize(torch.randn(N_KEYS, LATENT_DIM, generator=g), dim=-1)
    memory_view = torch.randn(LATENT_DIM, OBS_DIM, generator=g) / math.sqrt(LATENT_DIM)
    query_view = torch.randn(LATENT_DIM, OBS_DIM, generator=g) / math.sqrt(LATENT_DIM)
    return World(latent, memory_view, query_view)


def observe(
    world: World,
    ids: torch.Tensor,
    *,
    memory: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    z = world.latent[ids]
    view = world.memory_view if memory else world.query_view
    return z @ view + NOISE_STD * torch.randn(z.shape[0], OBS_DIM, generator=generator)


def train_encoder(
    model: DualAddressEncoder,
    world: World,
    *,
    steps: int = TRAIN_STEPS,
    seed: int = SEED,
) -> dict[str, float | int]:
    g = torch.Generator().manual_seed(seed + 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    started = time.perf_counter()
    final_loss = float("nan")
    for _ in range(steps):
        ids = torch.randint(0, N_KEYS, (TRAIN_BATCH,), generator=g)
        m = observe(world, ids, memory=True, generator=g)
        q = observe(world, ids, memory=False, generator=g)
        memory_embed = model.memory(m)
        query_embed = model.query(q)
        logits = query_embed @ memory_embed.T / 0.07
        target = torch.arange(TRAIN_BATCH)
        loss = 0.5 * (
            F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    return {
        "steps": steps,
        "batch": TRAIN_BATCH,
        "final_contrastive_loss": final_loss,
        "seconds": time.perf_counter() - started,
    }


def route_bits(n_events: int) -> int:
    # More stored events -> finer routes. Frozen before the exploratory result.
    return max(4, min(10, int(round(math.log2(n_events))) - 2))


def make_planes(seed: int) -> dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + 200)
    out: dict[int, torch.Tensor] = {}
    for bits in range(4, 11):
        p = torch.randn(N_TABLES, bits, EMBED_DIM, generator=g)
        out[bits] = F.normalize(p, dim=-1)
    return out


def _codes(embeddings: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    dots = torch.einsum("nd,tbd->ntb", embeddings, planes)
    bits = (dots > 0).long()
    powers = (2 ** torch.arange(planes.shape[1])).view(1, 1, -1)
    return (bits * powers).sum(-1)


def sparse_candidates(
    memory_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    planes: torch.Tensor,
) -> list[int]:
    """Multi-table exact buckets + low-margin one-bit probes.

    No event id/key id is used for routing.
    """
    memory_codes = _codes(memory_embeddings, planes)
    qdots = torch.einsum("d,tbd->tb", query_embedding, planes)
    qbits = (qdots > 0).long()
    powers = 2 ** torch.arange(planes.shape[1])
    qcodes = (qbits * powers).sum(-1)
    candidates: set[int] = set()

    for table in range(planes.shape[0]):
        probe_codes = [int(qcodes[table])]
        low_margin = torch.argsort(qdots[table].abs())[:PROBES_PER_TABLE]
        for bit in low_margin:
            probe_codes.append(int(qcodes[table] ^ int(powers[bit])))
        for code in probe_codes:
            hits = (memory_codes[:, table] == code).nonzero().flatten().tolist()
            candidates.update(int(i) for i in hits)
    return sorted(candidates)


def latest_semantic_match(
    memory_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    candidates: list[int] | None,
) -> int:
    """Retrieve the latest write among the semantic cluster around best match."""
    if candidates is None:
        idx = torch.arange(memory_embeddings.shape[0])
    else:
        if not candidates:
            return -1
        idx = torch.tensor(candidates)
    query_scores = memory_embeddings[idx] @ query_embedding
    anchor_idx = idx[query_scores.argmax()]
    anchor = memory_embeddings[anchor_idx]
    same_address = idx[(memory_embeddings[idx] @ anchor) >= CLUSTER_THRESHOLD]
    return int(same_address.max())


def _memory_embeddings(
    model: DualAddressEncoder,
    world: World,
    ids: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    return model.memory(observe(world, ids, memory=True, generator=generator))


def _query_embedding(
    model: DualAddressEncoder,
    world: World,
    key: int,
    generator: torch.Generator,
) -> torch.Tensor:
    ids = torch.tensor([key])
    return model.query(observe(world, ids, memory=False, generator=generator))[0]


def make_overwrite_memory(
    n_events: int,
    target: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.randint(0, N_KEYS - 1, (n_events,), generator=generator)
    ids = ids + (ids >= target).long()
    repeats = int(torch.randint(2, 5, (1,), generator=generator))
    positions = torch.randperm(n_events, generator=generator)[:repeats].sort().values
    ids[positions] = target
    return ids, positions


@torch.no_grad()
def evaluate_overwrite(
    model: DualAddressEncoder,
    world: World,
    planes_by_bits: dict[int, torch.Tensor],
    n_events: int,
    trials: int,
) -> dict:
    g = torch.Generator().manual_seed(SEED + 4000 + n_events)
    flat_correct = sparse_correct = route_hits = sparse_reads = 0
    bits = route_bits(n_events)
    planes = planes_by_bits[bits]

    for _ in range(trials):
        target = int(torch.randint(0, N_KEYS, (1,), generator=g))
        ids, positions = make_overwrite_memory(n_events, target, g)
        memory = _memory_embeddings(model, world, ids, g)
        query = _query_embedding(model, world, target, g)
        true_position = int(positions[-1])

        flat = latest_semantic_match(memory, query, None)
        candidates = sparse_candidates(memory, query, planes)
        sparse = latest_semantic_match(memory, query, candidates)

        flat_correct += int(flat == true_position)
        sparse_correct += int(sparse == true_position)
        route_hits += int(true_position in candidates)
        sparse_reads += len(candidates)

    return {
        "flat_accuracy": flat_correct / trials,
        "chm_v1_accuracy": sparse_correct / trials,
        "route_recall": route_hits / trials,
        "avg_candidate_reads": sparse_reads / trials,
        "fraction_flat_reads": (sparse_reads / trials) / n_events,
        "route_bits": bits,
    }


@torch.no_grad()
def evaluate_rare_fact(
    model: DualAddressEncoder,
    world: World,
    planes_by_bits: dict[int, torch.Tensor],
    n_events: int,
    trials: int,
) -> dict:
    g = torch.Generator().manual_seed(SEED + 6000 + n_events)
    flat_correct = sparse_correct = route_hits = sparse_reads = 0
    bits = route_bits(n_events)
    planes = planes_by_bits[bits]

    for _ in range(trials):
        target = int(torch.randint(0, N_KEYS, (1,), generator=g))
        ids = torch.randint(0, N_KEYS - 1, (n_events,), generator=g)
        ids = ids + (ids >= target).long()
        true_position = int(torch.randint(0, n_events, (1,), generator=g))
        ids[true_position] = target

        memory = _memory_embeddings(model, world, ids, g)
        query = _query_embedding(model, world, target, g)
        flat = int((memory @ query).argmax())

        candidates = sparse_candidates(memory, query, planes)
        if candidates:
            idx = torch.tensor(candidates)
            sparse = int(idx[(memory[idx] @ query).argmax()])
        else:
            sparse = -1

        flat_correct += int(flat == true_position)
        sparse_correct += int(sparse == true_position)
        route_hits += int(true_position in candidates)
        sparse_reads += len(candidates)

    return {
        "flat_accuracy": flat_correct / trials,
        "chm_v1_accuracy": sparse_correct / trials,
        "route_recall": route_hits / trials,
        "avg_candidate_reads": sparse_reads / trials,
        "fraction_flat_reads": (sparse_reads / trials) / n_events,
        "route_bits": bits,
    }


@torch.no_grad()
def evaluate_two_hop(
    model: DualAddressEncoder,
    world: World,
    planes_by_bits: dict[int, torch.Tensor],
    n_events: int,
    trials: int,
) -> dict:
    g = torch.Generator().manual_seed(SEED + 5000 + n_events)
    flat_correct = sparse_correct = sparse_reads = 0
    bits = route_bits(n_events)
    planes = planes_by_bits[bits]

    for _ in range(trials):
        key1 = int(torch.randint(0, N_KEYS, (1,), generator=g))
        key2 = (key1 + int(torch.randint(1, N_KEYS, (1,), generator=g))) % N_KEYS
        answer = (key2 + int(torch.randint(1, N_KEYS, (1,), generator=g))) % N_KEYS

        ids = torch.randint(0, N_KEYS, (n_events,), generator=g)
        values = torch.randint(0, N_KEYS, (n_events,), generator=g)
        for key in (key1, key2):
            mask = ids == key
            ids[mask] = (ids[mask] + 1) % N_KEYS

        positions = torch.randperm(n_events, generator=g)[:8].sort().values
        p1 = positions[:4].sort().values
        p2 = positions[4:].sort().values
        ids[p1] = key1
        values[p1] = torch.randint(0, N_KEYS, (4,), generator=g)
        values[p1[-1]] = key2
        ids[p2] = key2
        values[p2] = torch.randint(0, N_KEYS, (4,), generator=g)
        values[p2[-1]] = answer

        memory = _memory_embeddings(model, world, ids, g)

        q1 = _query_embedding(model, world, key1, g)
        flat_i1 = latest_semantic_match(memory, q1, None)
        flat_key2 = int(values[flat_i1])
        q2 = _query_embedding(model, world, flat_key2, g)
        flat_i2 = latest_semantic_match(memory, q2, None)
        flat_answer = int(values[flat_i2])

        c1 = sparse_candidates(memory, q1, planes)
        sparse_i1 = latest_semantic_match(memory, q1, c1)
        sparse_key2 = int(values[sparse_i1]) if sparse_i1 >= 0 else -1
        if sparse_key2 >= 0:
            q2s = _query_embedding(model, world, sparse_key2, g)
            c2 = sparse_candidates(memory, q2s, planes)
            sparse_i2 = latest_semantic_match(memory, q2s, c2)
            sparse_answer = int(values[sparse_i2]) if sparse_i2 >= 0 else -1
        else:
            c2 = []
            sparse_answer = -1

        flat_correct += int(flat_answer == answer)
        sparse_correct += int(sparse_answer == answer)
        sparse_reads += len(c1) + len(c2)

    return {
        "flat_accuracy": flat_correct / trials,
        "chm_v1_accuracy": sparse_correct / trials,
        "avg_candidate_reads_two_hops": sparse_reads / trials,
        "fraction_flat_reads": (sparse_reads / trials) / (2 * n_events),
        "route_bits": bits,
    }


def progression_decision(result: dict) -> str:
    """Frozen exploratory progression rule; not a breakthrough criterion."""
    for task in ("overwrite", "rare_fact"):
        row = result["eval"][task]["4096"]
        if row["chm_v1_accuracy"] + 0.015 < row["flat_accuracy"]:
            return "FAIL_DO_NOT_PROGRESS"
        if row["fraction_flat_reads"] > 0.10:
            return "FAIL_DO_NOT_PROGRESS"
    row = result["eval"]["two_hop"]["4096"]
    if row["chm_v1_accuracy"] + 0.02 < row["flat_accuracy"]:
        return "FAIL_DO_NOT_PROGRESS"
    if row["fraction_flat_reads"] > 0.10:
        return "FAIL_DO_NOT_PROGRESS"
    return "PASS_CPU_MECHANISM_ONLY"


def run(trials: int = 300, train_steps: int = TRAIN_STEPS) -> dict:
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    world = make_world(SEED)
    model = DualAddressEncoder()
    train = train_encoder(model, world, steps=train_steps)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    planes = make_planes(SEED)

    result = {
        "classification": "EXPLORATORY_CPU_FALSIFICATION_ONLY",
        "candidate": "CHM-v1 exact-backed adaptive multi-route retrieval",
        "seed": SEED,
        "seed_status": "consumed_exploratory_never_scientific",
        "train": train,
        "mechanism": {
            "exact_episodic_substrate_preserved": True,
            "routing_uses_event_or_key_ids": False,
            "n_tables": N_TABLES,
            "probes_per_table": PROBES_PER_TABLE,
            "cluster_threshold": CLUSTER_THRESHOLD,
            "noise_std": NOISE_STD,
        },
        "eval": {"overwrite": {}, "rare_fact": {}, "two_hop": {}},
    }
    for n_events in (64, 256, 1024, 4096):
        result["eval"]["overwrite"][str(n_events)] = evaluate_overwrite(
            model, world, planes, n_events, trials
        )
        result["eval"]["rare_fact"][str(n_events)] = evaluate_rare_fact(
            model, world, planes, n_events, trials
        )
    for n_events in (256, 1024, 4096):
        result["eval"]["two_hop"][str(n_events)] = evaluate_two_hop(
            model, world, planes, n_events, trials
        )
    result["decision"] = progression_decision(result)
    result["interpretation_ceiling"] = (
        "Mechanism progression only. No LM, equal-compute, scaling, novelty, or breakthrough claim."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    args = parser.parse_args()
    print(json.dumps(run(args.trials, args.train_steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
