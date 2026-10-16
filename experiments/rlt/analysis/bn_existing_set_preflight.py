"""BE/BN SOURCE-FAITHFUL, MODEL-FREE input/counterfactual correctness preflight.

DO NOT import weights, evaluate checkpoints, train, reserve/launch research jobs,
or use these preflight-only inputs for subsequent scientific inference.
This module reads the *frozen BC* input-generator helpers as they appear in the
isolated AX/BE source; it is not a stand-alone reproduction of the source.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from experiments.rlt.cpu_bounded_streaming_aw import balanced_holdout
from experiments.rlt.cpu_lastwrite_probe_ar import (
    QUERY, SET0, SET1, last_write_labels,
)

SEQ_LEN, CHUNK = 128, 8
TRAINED_DIAGNOSTIC_GAPS = (16, 64)
PREFLIGHT_ONLY_SEED = 182818284  # Never use for trained-checkpoint evaluation.


@dataclass(frozen=True)
class Triplets:
    baseline: torch.Tensor
    local_flip: torch.Tensor
    remote_flip: torch.Tensor
    labels: torch.Tensor
    selected_indices: torch.Tensor
    local_positions: torch.Tensor
    remote_positions: torch.Tensor
    population_total: int
    population_per_class: tuple[int, int]


def is_set(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens == SET0) | (tokens == SET1)


def matched_existing_write_triplets(gap: int, *, seed: int) -> Triplets:
    """Condition on naturally occurring local+remote writes with same SET bit.

    Does NOT identify an isolated chunk-phase effect: local/remote positions
    differ in their age as well as their chunk membership. Selected-stratum
    summaries must include the population denominators and target-class mix.
    """
    if gap not in TRAINED_DIAGNOSTIC_GAPS:
        raise ValueError("matched diagnostic only supports BC-trained gaps 16,64")
    if seed in (20671081, 20671082, 20671083, 20681081, 20691081,
                20701081, 314159265, 271828183, 161803399, 173205081,
                58232, 58233):
        raise ValueError("seed reserved/consumed by another purpose")
    x, y = balanced_holdout(gap, seed=seed)
    if x.shape != (256, SEQ_LEN) or y.shape != (256,) or x.dtype != torch.long:
        raise RuntimeError("unexpected BC balanced holdout shape")
    if (int((y == 0).sum()), int((y == 1).sum())) != (128, 128):
        raise RuntimeError("class balance changed")
    write = SEQ_LEN - 1 - gap
    chunk_start = write - write % CHUNK
    if not torch.equal(last_write_labels(x), y):
        raise RuntimeError("natural-input label oracle mismatch")
    if not bool((x[:, write] == SET0 + y).all()):
        raise RuntimeError("forced final SET mismatch")
    if not bool((x[:, -1] == QUERY).all()):
        raise RuntimeError("missing final QUERY")

    rows, locals_, remotes_ = [], [], []
    for i in range(len(y)):
        local = [p for p in range(chunk_start, write)
                 if int(x[i, p]) in (SET0, SET1)]
        remote = [p for p in range(0, chunk_start)
                  if int(x[i, p]) in (SET0, SET1)]
        if not local or not remote:
            continue
        local_pos = local[-1]
        same_bit_remote = [p for p in remote if int(x[i, p]) == int(x[i, local_pos])]
        if not same_bit_remote:
            continue
        rows.append(i)
        locals_.append(local_pos)
        remotes_.append(same_bit_remote[-1])

    chosen = torch.tensor(rows, dtype=torch.long)
    local_pos = torch.tensor(locals_, dtype=torch.long)
    remote_pos = torch.tensor(remotes_, dtype=torch.long)
    a = x[chosen].clone()
    b, c = a.clone(), a.clone()
    if len(chosen):
        r = torch.arange(len(chosen))
        if not torch.equal(a[r, local_pos], a[r, remote_pos]):
            raise RuntimeError("original SET bits not matched")
        b[r, local_pos] = (SET0 + SET1) - a[r, local_pos]
        c[r, remote_pos] = (SET0 + SET1) - a[r, remote_pos]
        truth = y[chosen]
        if not (torch.equal(last_write_labels(a), truth)
                and torch.equal(last_write_labels(b), truth)
                and torch.equal(last_write_labels(c), truth)):
            raise RuntimeError("counterfactual changed answer")
        if not (torch.equal(is_set(a), is_set(b))
                and torch.equal(is_set(a), is_set(c))):
            raise RuntimeError("counterfactual changed write positions")
        if not (torch.equal(a[:, write:], b[:, write:])
                and torch.equal(a[:, write:], c[:, write:])):
            raise RuntimeError("counterfactual changed forced write/suffix")
        if not (bool(((a != b).sum(1) == 1).all())
                and bool(((a != c).sum(1) == 1).all())
                and bool(((b != c).sum(1) == 2).all())):
            raise RuntimeError("counterfactual changed wrong number of tokens")
    return Triplets(a, b, c, y[chosen].clone(), chosen,
                    local_pos, remote_pos, len(y), (128, 128))


def model_free_report(*, seed: int = PREFLIGHT_ONLY_SEED) -> dict:
    """Hash and count selected input examples; no model forward pass."""
    out = {}
    for gap in TRAINED_DIAGNOSTIC_GAPS:
        p = matched_existing_write_triplets(gap, seed=seed + gap)
        out[str(gap)] = {
            "population": p.population_total,
            "selected": len(p.labels),
            "selected_class0": int((p.labels == 0).sum()),
            "selected_class1": int((p.labels == 1).sum()),
            "selected_fraction": len(p.labels) / p.population_total,
            "selected_indices_sha256": hashlib.sha256(
                p.selected_indices.numpy().tobytes()).hexdigest(),
        }
    return {
        "type": "MODEL_FREE_INPUT_PREFLIGHT_ONLY",
        "training": False, "checkpoint_loaded": False,
        "seed_preflight_only": seed,
        "population_class0": 128, "population_class1": 128,
        "trained_gaps": list(TRAINED_DIAGNOSTIC_GAPS),
        "rows": out,
    }
