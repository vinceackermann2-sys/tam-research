"""AX zero-GPU non-training audit; never reruns or edits frozen AW evidence.

AW has no committed trained checkpoint. Gate traces of a newly initialized
model must NOT be mislabeled as measurements of AW's trained weights. Supply
a separately trained checkpoint/model to diagnose learned retention later.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.rlt.cpu_bounded_streaming_aw import SEQ_LEN, balanced_holdout
from experiments.rlt.cpu_lastwrite_probe_ar import SET0, last_write_labels
from experiments.rlt.model_gated_scan import associative_affine_scan
from experiments.rlt.model_sliding_window_ax import SlidingWindowTransformer
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT

AW = Path("experiments/rlt/cpu/results/rlt-bounded-streaming-vs-reset-20261010-aw.json")
LONG_DISTANCES = (16, 64, 127)


def prove_unreachable_counterfactuals(
    baseline: SlidingWindowTransformer, examples: int = 4,
) -> dict[str, dict[str, int | bool]]:
    """Labels differ yet the entire baseline causal receptive field is identical."""
    baseline.eval()
    maximum = baseline.receptive_distance
    rows = {}
    for distance in LONG_DISTANCES:
        if distance <= maximum:
            raise ValueError("distance must exceed baseline full receptive field")
        x, y = balanced_holdout(distance, seed=20311081 + distance)
        x, y = x[:examples], y[:examples]
        flipped = x.clone()
        flipped[:, SEQ_LEN - 1 - distance] = SET0 + (1 - y)
        assert torch.equal(last_write_labels(flipped), 1 - y)
        suffix = maximum + 1
        assert torch.equal(x[:, -suffix:], flipped[:, -suffix:])
        with torch.no_grad():
            a, b = baseline(x)[:, -1, :], baseline(flipped)[:, -1, :]
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        rows[str(distance)] = {
            "paired_examples": len(y), "identical_final_tokens": suffix,
            "opposite_labels": True, "invariant_transformer_logits": True,
        }
    return rows


@torch.no_grad()
def scan_retention_trace(
    model: BoundedChunkGatedRLT, tokens: torch.Tensor,
    *, last_write_distance: int,
) -> dict[str, object]:
    """Probe actual supplied weights; not automatically the trained AW model.

    Reports the direct affine-scan retention path; this is only one of the
    mechanisms that may influence decoder logits. State width is bounded.
    """
    model.eval()
    if not model.carry_between_chunks or tokens.ndim != 2:
        raise ValueError("requires carry-enabled model and [batch,time] tokens")
    if not 1 <= last_write_distance < tokens.size(1):
        raise ValueError("invalid write distance")
    state = model.init_stream(tokens.size(0)).recurrent
    all_gates, all_norms, all_write_norms = [], [], []
    for lo in range(0, tokens.size(1), model.chunk_size):
        chunk = tokens[:, lo:lo + model.chunk_size]
        memory = model.encode(chunk)
        width = memory.size(-1)
        values = F.linear(memory, model.merge.weight[:, :width])
        gates = F.linear(memory, model.merge.weight[:, width:])
        retention = torch.sigmoid(gates.float() + model.GATE_LOGIT_BIAS)
        writes = (1 - retention) * torch.tanh(values.float())
        states = associative_affine_scan(retention, writes, state.float())
        state = states[:, -1, :]
        all_gates.append(retention.cpu())
        all_norms.append(states.float().norm(dim=-1).cpu())
        all_write_norms.append(writes.float().norm(dim=-1).cpu())
    g = torch.cat(all_gates, dim=1)
    retention_product = g[:, -last_write_distance:, :].prod(dim=1)
    return {
        "trace_type": "weights_supplied_by_caller_not_aw_checkpoint",
        "mean_gate": float(g.mean()),
        "fraction_gate_above_0_99": float((g > 0.99).float().mean()),
        "fraction_gate_below_0_5": float((g < 0.5).float().mean()),
        "mean_direct_retention_over_gap": float(retention_product.mean()),
        "state_norm_by_token": torch.cat(all_norms, dim=1).mean(dim=0).tolist(),
        "write_norm_by_token": torch.cat(all_write_norms, dim=1).mean(dim=0).tolist(),
        "state_shape": list(state.shape),
    }


def frozen_aw_class_bias() -> dict:
    aw = json.loads(AW.read_text())
    if aw.get("job_id") != "rlt-cpu-bounded-streaming-vs-reset-20261010-aw":
        raise ValueError("not the frozen AW evidence")
    if aw.get("status") != "complete":
        raise ValueError("AW not terminal")
    by_name = {r["name"]: r for r in aw["results"]}
    carry = by_name["streaming_carry"]
    out = {}
    for d in LONG_DISTANCES:
        metrics = carry["final"][str(d)]
        out[str(d)] = {
            "accuracy": metrics["accuracy"],
            "class0_accuracy": metrics["class0_accuracy"],
            "class1_accuracy": metrics["class1_accuracy"],
            "class_gap": metrics["class0_accuracy"] - metrics["class1_accuracy"],
            "binary_nll": metrics["binary_nll"],
        }
    return out


def run_readonly_audit() -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(20321081)
    baseline = SlidingWindowTransformer()
    return {
        "status": "cpu_preflight_only", "gpu_requested": False,
        "new_training_attempt": False, "checkpoint_available_for_aw": False,
        "transformer_receptive_distance": baseline.receptive_distance,
        "counterfactuals": prove_unreachable_counterfactuals(baseline),
        "frozen_aw_bias": frozen_aw_class_bias(),
        "interpretation": "No trained AW weights available for gate traces; no new model comparison yet.",
    }


if __name__ == "__main__":
    print(json.dumps(run_readonly_audit(), indent=2, sort_keys=True))
