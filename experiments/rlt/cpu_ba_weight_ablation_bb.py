"""BB: inference-only BA trained-state causal ablation on fresh CPU holdouts.

No optimizer, no training, no model changes. Frozen BA checkpoint is read-only.
Test carries against (a) all-chunk reset and (b) erase state only immediately
before the final chunk. Two new independently balanced heldout seeds. Opposite
labels with identical final chunk provide an information-inaccessibility proof
for both ablations at 16/64/127. This is synthetic-task evidence only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.rlt.cpu_bounded_streaming_aw import balanced_holdout
from experiments.rlt.cpu_carry_intervention_ba import build, counterfactual_pairs, DISTANCES
from experiments.rlt.cpu_lastwrite_probe_ar import SET0, SET1
from experiments.rlt.model_streaming_bounded import (
    BoundedChunkGatedRLT, BoundedStreamState, tiny_streaming_config,
)

JOB_ID = "rlt-cpu-ba-weight-ablation-20261010-bb"
BA_JOB_ID = "rlt-cpu-carry-intervention-20261010-ba"
BA_RESULT = Path("experiments/rlt/cpu/results/rlt-carry-intervention-20261010-ba.json")
CHECKPOINT = Path("experiments/rlt/cpu/checkpoints/rlt-carry-intervention-20261010-ba/carry.pt")
EXPECTED_CHECKPOINT_SHA256 = "74fe2bb1cfb74242aaba3d0eb8ac5df13c1a421a7d50054a0d7dbf503fb4149b"
RESULT = Path("experiments/rlt/cpu/results/rlt-ba-weight-ablation-20261010-bb.json")
SEEDS = (20631081, 20641081)
PAIR_SEED = 20651081
BATCH = 8


class EraseBeforeFinalChunk(BoundedChunkGatedRLT):
    """Exact original model except reset state at t=120, not before."""
    def __init__(self):
        super().__init__(tiny_streaming_config(), carry_between_chunks=True)

    def forward_chunk(self, tokens: torch.Tensor, state: BoundedStreamState):
        if state.total_tokens == 120:
            state = BoundedStreamState(
                recurrent=self.start_state.unsqueeze(0).expand(tokens.size(0), -1),
                total_tokens=state.total_tokens,
            )
        return super().forward_chunk(tokens, state)


def load_and_verify():
    prior = json.loads(BA_RESULT.read_text())
    assert prior["status"] == "complete" and prior["job_id"] == BA_JOB_ID
    assert prior["checkpoint"]["path"] == CHECKPOINT.as_posix()
    assert prior["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
    content = CHECKPOINT.read_bytes()
    if len(content) != prior["checkpoint"]["bytes"] or hashlib.sha256(content).hexdigest() != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("BA checkpoint differs from SHA256-pinned immutable evidence")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert state["job_id"] == BA_JOB_ID and state["model_seed"] == 20541081
    carry = build()
    reset = BoundedChunkGatedRLT(tiny_streaming_config(), carry_between_chunks=False)
    erase = EraseBeforeFinalChunk()
    for model in (carry, reset, erase):
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
    for other in (reset, erase):
        for name, val in carry.state_dict().items():
            torch.testing.assert_close(val, other.state_dict()[name], atol=0, rtol=0)
    return {"carry": carry, "full_reset": reset, "erase_t120": erase}


@torch.no_grad()
def evaluate(model) -> dict:
    out = {}
    for gap in DISTANCES:
        total = 0
        correct, nll = 0, 0.0
        labels = [0, 0]
        correct_classes = [0, 0]
        for base in SEEDS:
            x, y = balanced_holdout(gap, seed=base + gap)
            for lo in range(0, len(y), BATCH):
                xb, truth = x[lo:lo+BATCH], y[lo:lo+BATCH]
                logits = model(xb)[:, -1, SET0:SET1+1]
                pred = logits.argmax(-1)
                total += len(truth)
                correct += int((pred == truth).sum())
                nll += float(F.cross_entropy(logits, truth, reduction="sum"))
                for cls in (0, 1):
                    labels[cls] += int((truth == cls).sum())
                    correct_classes[cls] += int(((pred == cls) & (truth == cls)).sum())
        assert total == 512 and labels == [256, 256]
        out[str(gap)] = {
            "examples": total, "accuracy": correct / total,
            "binary_nll": nll / total,
            "class0_accuracy": correct_classes[0] / labels[0],
            "class1_accuracy": correct_classes[1] / labels[1],
        }
    return out


@torch.no_grad()
def counterfactual_test(models) -> dict:
    out = {}
    for gap in (16, 64, 127):
        a, b, y = counterfactual_pairs(gap, seed=PAIR_SEED+gap)
        if not torch.equal(a[:, -8:], b[:, -8:]):
            raise RuntimeError("suffix mismatch")
        rows = {}
        for name, model in models.items():
            logits_a = model(a)[:, -1, SET0:SET1+1]
            logits_b = model(b)[:, -1, SET0:SET1+1]
            correct_both = float(((logits_a.argmax(-1) == y) & (logits_b.argmax(-1) == 1-y)).float().mean())
            diff = float((logits_a - logits_b).abs().mean())
            same = torch.equal(logits_a, logits_b)
            if name != "carry" and not same:
                raise RuntimeError(f"masked-history leakage in {name} at gap {gap}")
            rows[name] = {
                "paired_examples": len(y),
                "both_correct_fraction": correct_both,
                "mean_absolute_binary_logit_difference": diff,
                "exact_same_logits": same,
            }
        out[str(gap)] = rows
    return out


def run() -> dict:
    if RESULT.exists():
        raise RuntimeError("BB inference already consumed: no duplicate result")
    torch.set_num_threads(2)
    models = load_and_verify()
    scores = {name: evaluate(model) for name, model in models.items()}
    paired = counterfactual_test(models)
    return {
        "schema": 1, "job_id": JOB_ID, "status": "complete",
        "classification": "CPU_INFERENCE_ONLY_TRAINED_STATE_ABLATION_ENGINEERING",
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "source_ba_job_id": BA_JOB_ID,
        "gpu_requested": False, "new_training": False,
        "scientific_execution": False, "breakthrough_claim_supported": False,
        "protocol": {
            "models": list(models), "same_trained_parameters": True,
            "final_window": 8, "chunk_count": 16,
            "last_chunk_erasure_boundary": 120,
            "distances": list(DISTANCES), "eval_seeds": list(SEEDS),
            "counterfactual_seed": PAIR_SEED,
            "balanced_eval_examples_per_class_per_gap": 256,
            "counterfactual_pairs_per_gap": 16,
        },
        "scores": scores, "counterfactuals": paired,
        "interpretation_ceiling": (
            "One fixed BA checkpoint and new inference-only synthetic examples. "
            "Causal state intervention tests whether carry is required for long "
            "distance SET recall under per-chunk local attention. It is not "
            "scientific replication, equal-compute Transformer science, or a "
            "general language-model breakthrough."
        ),
    }


if __name__ == "__main__":
    result = run()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "job_id": JOB_ID,
        "accuracies": {n: {d: x["accuracy"] for d,x in scores.items()}
            for n,scores in result["scores"].items()},
        "counterfactuals": result["counterfactuals"],
    }, sort_keys=True))
