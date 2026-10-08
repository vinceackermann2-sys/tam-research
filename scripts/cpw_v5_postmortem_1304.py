"""CPW-v5 #1304: CPU-only, postmortem diagnostics on frozen #1290.

These are constructed-feature and initialization controls, not a learned model
evaluation. No scientific seeds are used and no paid compute is requested.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from tam_research.cpw_v5_afm.model import (
    AssociativeFastMemoryPredictor,
    associative_scan_vectorized,
)


def positive_key_collision_probe() -> dict[str, float]:
    torch.manual_seed(1304)
    model = AssociativeFastMemoryPredictor(256, 32)
    with torch.no_grad():
        hidden = torch.randn(256, 256)
        key = F.elu(model.key_proj(hidden)) + 1.0
        sims = F.normalize(key, dim=-1) @ F.normalize(key, dim=-1).T
        idx = torch.triu_indices(256, 256, offset=1)
        offdiag = sims[idx[0], idx[1]]
        # The shared constant component in ELU+1 is not an observation
        # about a trained model. Centering gives a diagnostic contrast.
        centered = key - key.mean(dim=0, keepdim=True)
        centered_sims = F.normalize(centered, dim=-1) @ F.normalize(
            centered, dim=-1
        ).T
        centered_offdiag = centered_sims[idx[0], idx[1]]
        gate = torch.sigmoid(
            model.gate_prev(torch.randn(64, 256))
            + model.gate_cur(torch.randn(64, 256))
        )
        return {
            "initial_key_offdiag_cosine": float(offdiag.mean()),
            "initial_centered_key_offdiag_cosine": float(centered_offdiag.mean()),
            "initial_write_gate_mean": float(gate.mean()),
        }


def constructed_binding_probe(*, filler_gate: float) -> dict[str, float]:
    if not 0.0 <= filler_gate <= 1.0:
        raise ValueError("filler_gate must be in [0, 1]")

    torch.manual_seed(1305)
    t = 320
    rank = 32
    count = 8
    features = torch.ones(1, t, rank)
    # Distinct key identities; positive common component models ELU+1.
    for i in range(t):
        features[0, i, (i * 13 + 17) % rank] += 0.2

    values = torch.zeros(1, t, rank)
    gates = torch.full((1, t, 1), float(filler_gate))
    gates[:, 0] = 0.0
    for pos in range(1, t - 1):
        # Distractor writes encode unrelated value slots.
        values[0, pos, (pos * 7 + 5) % count] = 1.0

    # Eight explicit key -> value entries, each one distinct.
    for i in range(count):
        key_position = 8 + 3 * i
        value_position = key_position + 1
        features[0, key_position] = 1.0
        features[0, key_position, i] += 8.0
        values[0, value_position] = 0.0
        values[0, value_position, i] = 1.0
        gates[0, value_position, 0] = 1.0

    # Query key 6 after long filler; zero query self-write.
    features[0, -1] = features[0, 8 + 3 * 6]
    gates[:, -1] = 0.0
    reads, _ = associative_scan_vectorized(features, values, gates)
    query = reads[0, -1, :count]
    target_score = query[6]
    strongest_wrong = torch.cat((query[:6], query[7:])).max()

    return {
        "filler_gate": float(filler_gate),
        "target_score": float(target_score),
        "strongest_wrong": float(strongest_wrong),
        "target_margin": float(target_score - strongest_wrong),
        "target_rank_one": float(target_score > strongest_wrong),
    }


def inclusive_current_write_probe() -> dict[str, float]:
    """A changing current write can change same-position inclusive read."""
    torch.manual_seed(1306)
    keys = torch.rand(1, 3, 8) + 0.1
    values = torch.zeros(1, 3, 8)
    values[0, 1, 2] = 1.0
    gates = torch.zeros(1, 3, 1)
    gates[0, 1] = 1.0
    gates[0, 2] = 1.0

    before, _ = associative_scan_vectorized(keys, values, gates)
    mutated = values.clone()
    mutated[0, 2, 4] = 100.0
    after, _ = associative_scan_vectorized(keys, mutated, gates)
    return {
        "same_position_read_change": float(
            (after[0, 2] - before[0, 2]).abs().max()
        ),
        "earlier_read_change": float(
            (after[0, :2] - before[0, :2]).abs().max()
        ),
    }


def report() -> dict[str, object]:
    return {
        "issue": 1304,
        "scientific_source_issue": 1290,
        "cpu_only": True,
        "scientific_seeds_consumed": False,
        "trained_checkpoint_loaded": False,
        "key_collision": positive_key_collision_probe(),
        "oracle_binding_no_filler": constructed_binding_probe(filler_gate=0.0),
        "oracle_binding_with_filler": constructed_binding_probe(filler_gate=0.05),
        "inclusive_current_write": inclusive_current_write_probe(),
        "scientific_interpretation": (
            "Mechanistic CPU probes; not an observed effect on trained CPW-v5."
        ),
    }


if __name__ == "__main__":
    print("CPW_V5_POSTMORTEM=" + json.dumps(report(), sort_keys=True))
