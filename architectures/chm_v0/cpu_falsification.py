from __future__ import annotations

"""Exploratory CPU falsification for CHM-v0.

This is deliberately not a scientific LM run. It tests whether a lossy learned
page directory can preserve exact episodic addressability under overwrite while
reducing candidate reads. Seed 59317 is exploratory and permanently consumed.
"""

import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 59317
N_KEYS = 64
N_VALUES = 64
DIM = 32
PAGE_SIZE = 16
DIRECTORY_SLOTS = 4
HARD_TOP_PAGES = 2
TRAIN_EVENTS = 64
TRAIN_STEPS = 700
TRAIN_BATCH = 64
EVAL_SAMPLES = 1024


class CHMV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory_key = nn.Embedding(N_KEYS, DIM)
        self.query_key = nn.Embedding(N_KEYS, DIM)
        self.value = nn.Embedding(N_VALUES, DIM)
        self.anchors = nn.Parameter(torch.randn(DIRECTORY_SLOTS, DIM) / math.sqrt(DIM))
        self.query_proj = nn.Linear(DIM, DIM, bias=False)
        self.key_proj = nn.Linear(DIM, DIM, bias=False)
        self.out = nn.Linear(DIM, N_VALUES)
        self.recency = nn.Parameter(torch.tensor(1.0))

    def _key_repr(self, keys: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_proj(self.memory_key(keys)), dim=-1)

    def _query_repr(self, query: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_proj(self.query_key(query)), dim=-1)

    def _page_prototypes(self, keys: torch.Tensor) -> torch.Tensor:
        # keys: [batch, pages, events, dim] -> [batch, pages, slots, dim]
        anchors = F.normalize(self.anchors, dim=-1)
        assignment = torch.einsum("bped,sd->bpes", keys, anchors) * 4.0
        weights = assignment.softmax(dim=2)
        prototypes = torch.einsum("bpes,bped->bpsd", weights, keys)
        return F.normalize(prototypes, dim=-1)

    def _scores(self, keys: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_events = keys.shape
        if n_events % PAGE_SIZE:
            raise ValueError("event count must be divisible by PAGE_SIZE")
        n_pages = n_events // PAGE_SIZE
        key_repr = self._key_repr(keys).view(batch, n_pages, PAGE_SIZE, DIM)
        query_repr = self._query_repr(query)
        prototypes = self._page_prototypes(key_repr)
        page_logits = torch.einsum("bd,bpsd->bps", query_repr, prototypes).amax(-1) * 8.0
        event_logits = torch.einsum("bd,bped->bpe", query_repr, key_repr) * 10.0
        position = torch.linspace(0.0, 1.0, n_events, device=keys.device).view(1, n_pages, PAGE_SIZE)
        event_logits = event_logits + F.softplus(self.recency) * position
        return page_logits, event_logits

    def training_loss(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        query: torch.Tensor,
        target: torch.Tensor,
        target_page: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_events = keys.shape
        page_logits, event_logits = self._scores(keys, query)
        soft_page_gate = F.log_softmax(page_logits, dim=-1).unsqueeze(-1)
        event_scores = (event_logits + soft_page_gate).view(batch, n_events)
        read_weights = F.softmax(event_scores, dim=-1)
        context = torch.einsum("bn,bnd->bd", read_weights, self.value(values))
        answer_logits = self.out(context)
        return F.cross_entropy(answer_logits, target) + 0.5 * F.cross_entropy(page_logits, target_page)

    @torch.no_grad()
    def predict(self, keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_events = keys.shape
        n_pages = n_events // PAGE_SIZE
        page_logits, event_logits = self._scores(keys, query)
        top_pages = page_logits.topk(min(HARD_TOP_PAGES, n_pages), dim=-1).indices
        page_mask = torch.zeros(batch, n_pages, dtype=torch.bool, device=keys.device)
        page_mask.scatter_(1, top_pages, True)
        sparse_scores = event_logits.masked_fill(~page_mask.unsqueeze(-1), -1e9).view(batch, n_events)
        read_weights = F.softmax(sparse_scores, dim=-1)
        context = torch.einsum("bn,bnd->bd", read_weights, self.value(values))
        return self.out(context).argmax(-1), top_pages


class FlatRetrieval(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory_key = nn.Embedding(N_KEYS, DIM)
        self.query_key = nn.Embedding(N_KEYS, DIM)
        self.value = nn.Embedding(N_VALUES, DIM)
        self.query_proj = nn.Linear(DIM, DIM, bias=False)
        self.key_proj = nn.Linear(DIM, DIM, bias=False)
        self.out = nn.Linear(DIM, N_VALUES)
        self.recency = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        query: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_repr = F.normalize(self.key_proj(self.memory_key(keys)), dim=-1)
        query_repr = F.normalize(self.query_proj(self.query_key(query)), dim=-1)
        scores = torch.einsum("bd,bnd->bn", query_repr, key_repr) * 10.0
        scores = scores + F.softplus(self.recency) * torch.linspace(
            0.0, 1.0, keys.size(1), device=keys.device
        ).view(1, -1)
        context = torch.einsum("bn,bnd->bd", F.softmax(scores, dim=-1), self.value(values))
        logits = self.out(context)
        if target is None:
            return logits.argmax(-1)
        return F.cross_entropy(logits, target)


def make_batch(batch_size: int, n_events: int, seed: int | None = None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    keys = torch.randint(N_KEYS, (batch_size, n_events), generator=generator)
    values = torch.randint(N_VALUES, (batch_size, n_events), generator=generator)
    query = torch.randint(N_KEYS, (batch_size,), generator=generator)
    target = torch.empty(batch_size, dtype=torch.long)
    target_page = torch.empty(batch_size, dtype=torch.long)
    stale = torch.empty(batch_size, dtype=torch.long)

    for row in range(batch_size):
        repeats = int(torch.randint(2, 5, (1,), generator=generator))
        positions = torch.randperm(n_events, generator=generator)[:repeats].sort().values
        keys[row][keys[row] == query[row]] = (query[row] + 1) % N_KEYS
        for position in positions:
            keys[row, position] = query[row]
            values[row, position] = torch.randint(N_VALUES, (1,), generator=generator)
        last_position = int(positions[-1])
        target[row] = values[row, last_position]
        target_page[row] = last_position // PAGE_SIZE
        stale[row] = values[row, int(positions[-2])]

    return keys, values, query, target, target_page, stale


def train(model: nn.Module, kind: str) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    started = time.perf_counter()
    last_loss = torch.tensor(float("nan"))
    for _ in range(TRAIN_STEPS):
        batch = make_batch(TRAIN_BATCH, TRAIN_EVENTS)
        optimizer.zero_grad(set_to_none=True)
        if kind == "chm":
            last_loss = model.training_loss(*batch[:5])
        else:
            last_loss = model(*batch[:3], target=batch[3])
        last_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return time.perf_counter() - started, float(last_loss.detach())


@torch.no_grad()
def evaluate(model: nn.Module, kind: str, n_events: int) -> dict[str, float | int | None]:
    correct = 0
    stale_errors = 0
    route_hits = 0
    for offset in range(0, EVAL_SAMPLES, 128):
        batch = make_batch(128, n_events, seed=999 + n_events + offset)
        if kind == "chm":
            prediction, top_pages = model.predict(*batch[:3])
            route_hits += int((top_pages == batch[4][:, None]).any(-1).sum())
        else:
            prediction = model(*batch[:3])
        correct += int((prediction == batch[3]).sum())
        stale_errors += int((prediction == batch[5]).sum())

    if kind == "chm":
        pages = n_events // PAGE_SIZE
        comparisons = pages * DIRECTORY_SLOTS + HARD_TOP_PAGES * PAGE_SIZE
        route_recall: float | None = route_hits / EVAL_SAMPLES
    else:
        comparisons = n_events
        route_recall = None

    return {
        "accuracy": correct / EVAL_SAMPLES,
        "stale_error": stale_errors / EVAL_SAMPLES,
        "route_recall": route_recall,
        "comparisons_per_query": comparisons,
        "fraction_flat_comparisons": comparisons / n_events,
    }


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    chm = CHMV0()
    flat = FlatRetrieval()
    chm_seconds, chm_loss = train(chm, "chm")
    flat_seconds, flat_loss = train(flat, "flat")
    result = {
        "classification": "EXPLORATORY_CPU_FALSIFICATION_ONLY",
        "candidate": "CHM-v0",
        "seed": SEED,
        "train": {
            "chm_seconds": chm_seconds,
            "flat_seconds": flat_seconds,
            "chm_last_loss": chm_loss,
            "flat_last_loss": flat_loss,
        },
        "eval": {},
    }
    for n_events in (64, 256, 1024):
        result["eval"][str(n_events)] = {
            "chm": evaluate(chm, "chm", n_events),
            "flat": evaluate(flat, "flat", n_events),
        }
    result["decision"] = "FAIL_DO_NOT_SCALE"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
