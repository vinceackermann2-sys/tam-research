"""First actual END-TO-END public cross-modal learning pilot for user SI design.

No paid compute, scientific attempt, private heldout or breakthrough claim.
An externally defined task is NOT claimed: this is a SOURCE-AUTHORED
DEVELOPMENT-ONLY compositional task, with five REAL input modality encoders.

Each episode contains 4 independently chosen public symbols, one each
from text, RGB image, audio, and video, followed by a motor query specifying
which past symbol to report. The requested answer is available only in
previous public inputs; the query itself is independent of the answer.
Neither model sees the answer as its observation, action or prior reward.

Compare the exact SIOriginalIntegratedV1 sparse hybrid with a Transformer
backbone replacement retaining EXACTLY the same encoders, adaptive latent
reasoning, action head, memory/fast/replay interfaces and other components.
Both receive the same 96 CPU updates, exactly the same samples/optimizer,
and a common initial parameter state for all matching non-backbone weights.

Limitations: NOT FLOP- or parameter-matched, no persistent memory feedback,
no world-model or search training, and only tiny synthetic sensor patterns.
This is a trainability/representation check, not a full SI scientific test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from tam_research.si_original_hybrid_v0 import DenseBlock
from tam_research.si_original_integrated_v1 import SIOriginalIntegratedV1

INITIALIZATION_FIXTURE = 120312
TRAIN_ID_FIRST = 11001
VALIDATION_ID_FIRST = 21001
UPDATES = 96
BATCH = 16
VALIDATION_BATCH = 32
VALIDATION_BATCHES = 8
MODALITIES = ("text", "image", "audio", "video")
WIDTH = 24
LAYERS = 4
EXPERTS = 4
CPU_THREADS = 2


class SIIntegratedTransformerControl(SIOriginalIntegratedV1):
    """Identical full surrounding system; only recurrent/MoE core replaced."""

    def __init__(self):
        super().__init__(width=WIDTH, layers=LAYERS, experts=EXPERTS,
                         max_steps=3)
        self.blocks = nn.ModuleList([DenseBlock(self.cfg) for _ in range(LAYERS)])

    def _backbone(self, steps):
        seq = self.encode(steps)
        for block in self.blocks:
            seq = block(seq)
        return self.norm(seq[:, -1]), seq.new_zeros(()), []


def _configure_models() -> tuple[SIOriginalIntegratedV1, SIIntegratedTransformerControl]:
    torch.manual_seed(INITIALIZATION_FIXTURE)
    hybrid = SIOriginalIntegratedV1(width=WIDTH, layers=LAYERS, experts=EXPERTS,
                                    max_steps=3)
    torch.manual_seed(INITIALIZATION_FIXTURE + 1)
    transformer = SIIntegratedTransformerControl()
    # Match initial corresponding NON-BACKBONE parameters exactly. Both
    # backbone families retain their own native untrained core parameters.
    original = hybrid.state_dict()
    current = transformer.state_dict()
    reused = {name: weight.clone() for name, weight in original.items()
              if not name.startswith("blocks.") and name in current
              and current[name].shape == weight.shape}
    transformer.load_state_dict({**current, **reused}, strict=True)
    return hybrid, transformer


def _public_batch(seed: int, batch_size: int):
    if (type(seed) is not int or seed < 1
            or type(batch_size) is not int or not 1 <= batch_size <= 64):
        raise ValueError("positive fixed development generator seed/batch required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.randint(0, 4, (batch_size, 4), generator=generator)
    query = torch.randint(0, 4, (batch_size,), generator=generator)
    target = labels.gather(1, query[:, None]).squeeze(1)
    # Four independent encoders; modality values carry ONLY their own label.
    # Query is a legal motor-vector, and NEVER encodes the label.
    text_ids = labels[:, 0] + 20
    image = torch.zeros(batch_size, 3, 8, 8)
    image[:, 0, :, :] = (.10 + .20 * labels[:, 1].float())[:, None, None]
    image[:, 1, :, :] = .30
    sound = torch.ones(batch_size, 32) * (
        .10 + .20 * labels[:, 2].float())[:, None]
    video = torch.zeros(batch_size, 2, 3, 8, 8)
    video[:, :, 0, :, :] = (
        .10 + .20 * labels[:, 3].float())[:, None, None, None]
    video[:, :, 1, :, :] = .20
    motor = F.one_hot(query, 4).float()
    steps = (
        ("text", text_ids),
        ("image", image),
        ("audio", sound),
        ("video", video),
        ("motor", motor),
    )
    return steps, target, query


def _shuffled_cue_batch(steps, *, seed: int):
    """Ablate the stored public cues without changing the motor query/label."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    modified = []
    for kind, sensor in steps:
        if kind == "motor":
            modified.append((kind, sensor.clone()))
        else:
            modified.append((kind, sensor[torch.randperm(
                sensor.size(0), generator=g)].clone()))
    return tuple(modified)


def _policy_logits(model, steps):
    return model(steps, enable_memory=False, enable_fast=False,
                 enable_search=False, reason_budget=3)["logits"]


@dataclass(frozen=True)
class TrainReadout:
    kind: str
    parameters: int
    training_updates: int
    training_samples: int
    initial_training_loss: float
    final_training_loss: float
    validation_loss: float
    validation_accuracy: float
    cue_shuffled_accuracy: float
    accuracy_by_queried_modality: dict[str, float]
    train_wall_seconds: float
    train_flops_matched: bool
    learned_complete_architecture: bool


@dataclass(frozen=True)
class CrossModalPilotReadout:
    status: str
    requirement: str
    backbone_original_source_unchanged: bool
    train_seed_first: int
    validation_seed_first: int
    validation_samples: int
    baseline_chance_accuracy: float
    equal_data_and_optimizer: bool
    results: tuple[TrainReadout, ...]
    scientific_seeds_consumed: int
    scientific_gate_passed: bool
    may_claim_hybrid_superiority: bool


def run_crossmodal_development_pilot() -> CrossModalPilotReadout:
    """One frozen CPU developmental train run per backbone. No restarts."""
    threads = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    try:
        hybrid, transformer = _configure_models()
        results = []
        for kind, model in (("original_hybrid", hybrid),
                            ("transformer_same_supplementary_mechanisms", transformer)):
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003,
                                           weight_decay=0.)
            initial = None
            started = time.perf_counter()
            for update in range(UPDATES):
                steps, target, _ = _public_batch(TRAIN_ID_FIRST+update, BATCH)
                optimizer.zero_grad(set_to_none=True)
                out = model(steps, enable_memory=False, enable_fast=False,
                            enable_search=False, reason_budget=3)
                loss = F.cross_entropy(out["logits"], target)+.001*out["balance"]
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite original-architecture dev loss")
                if initial is None:
                    initial = float(loss.detach())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            seconds = time.perf_counter()-started
            final = float(loss.detach())
            model.eval()
            total = 0
            correct = 0
            shuffled = 0
            validation_loss = 0.
            per_modal_correct = [0]*4
            per_modal_total = [0]*4
            with torch.no_grad():
                for i in range(VALIDATION_BATCHES):
                    steps, targets, queries = _public_batch(
                        VALIDATION_ID_FIRST+i, VALIDATION_BATCH)
                    logits = _policy_logits(model, steps)
                    perturb = _shuffled_cue_batch(
                        steps, seed=VALIDATION_ID_FIRST+200+i)
                    reordered_logits = _policy_logits(model, perturb)
                    predicted = logits.argmax(dim=-1)
                    total += len(targets)
                    correct += int((predicted==targets).sum())
                    shuffled += int((reordered_logits.argmax(-1)==targets).sum())
                    validation_loss += float(F.cross_entropy(logits, targets))*len(targets)
                    for q in range(4):
                        used = queries==q
                        per_modal_total[q] += int(used.sum())
                        per_modal_correct[q] += int((predicted[used]==targets[used]).sum())
            assert all(n>0 for n in per_modal_total)
            results.append(TrainReadout(
                kind=kind, parameters=sum(p.numel() for p in model.parameters()),
                training_updates=UPDATES, training_samples=UPDATES*BATCH,
                initial_training_loss=initial,final_training_loss=final,
                validation_loss=validation_loss/total,
                validation_accuracy=correct/total,
                cue_shuffled_accuracy=shuffled/total,
                accuracy_by_queried_modality={
                    name:per_modal_correct[i]/per_modal_total[i]
                    for i,name in enumerate(MODALITIES)},
                train_wall_seconds=seconds,
                train_flops_matched=False,
                learned_complete_architecture=False,
            ))
        return CrossModalPilotReadout(
            status="SOURCE_AUTHORED_DEVELOPMENT_ONLY",
            requirement="public cross-modal delayed conditional retrieval",
            backbone_original_source_unchanged=True,
            train_seed_first=TRAIN_ID_FIRST,
            validation_seed_first=VALIDATION_ID_FIRST,
            validation_samples=VALIDATION_BATCH*VALIDATION_BATCHES,
            baseline_chance_accuracy=.25,
            equal_data_and_optimizer=True,
            results=tuple(results),
            scientific_seeds_consumed=0,
            scientific_gate_passed=False,
            may_claim_hybrid_superiority=False,
        )
    finally:
        torch.set_num_threads(threads)


def main():
    print("ORIGINAL USER SI — SHARED-SENSOR CROSSMODAL TRAINABILITY PILOT")
    print("DEVELOPMENT ONLY: NOT FAIR ACTIVE-FLOP SCIENCE OR COMPLETE SI",flush=True)
    result = run_crossmodal_development_pilot()
    print(json.dumps(asdict(result), sort_keys=True, indent=2, allow_nan=False))


if __name__=="__main__":
    main()
