# CHM-v1 frozen run manifest

Classification: `ZERO_CREDIT_MANIFEST_COMPLETENESS_ONLY`.

Issue #854 remains the scientific authority. This file adds no launcher, workflow, GPU allocation, paid-compute authority, model/data/optimizer change, scientific threshold, or seed consumption.

## Purpose

The provenance-bound gate already requires exact execution code SHA and exact corpus byte fingerprints. This manifest layer additionally binds any future scientific evidence to the already-preregistered evaluator seeds, evaluation sizes, memory slices, timing workloads, scientific seed order, and planned resource ceiling.

The authoritative entry point for a future authorized result is:

`tam_research.chm_v1_run_manifest.evaluate_scientific_gate_with_manifest`

It validates the frozen run manifest, then delegates to the provenance-bound gate, which in turn delegates to the unchanged scientific gate.

## Frozen evaluation identity

- long-memory evaluator seed: 8,540,911
- validation sampling seed: 8,540,912
- systems timing seed: 8,540,913
- held-out language: 524,288 tokens/model = 64 × 8 × 1,024
- long-memory probes: 24 cases/family, 96 total
- rare/overwrite slice: 512 prior items
- two-hop slice: 1,024 prior items
- timing batch-1: 8 × 1,024-token sessions
- timing throughput: 8 batches × 8 × 1,024 tokens

## Frozen resource plan — not authorization

- NVIDIA L4
- 4 CPU cores
- 8 GiB RAM
- max 3,600 seconds/seed
- max 10,800 GPU-seconds total
- seed order 8611 → 8612 → 8613
- sequential only
- no automatic retries
- max $4.00 aggregate billed compute

These fields record the already-frozen envelope; they do not grant permission to use it.

A non-empty external `authorization_ref` is mandatory in the final run manifest so any paid scientific evidence can be traced to the separate explicit user authorization. Supplying a string to the validator does not itself authorize or launch anything.

## Spend boundary

Scientific seeds 8611/8612/8613 remain reserved and unused. No launcher, workflow, trigger, Modal allocation, or paid resource may be created until the user explicitly authorizes the frozen envelope recorded in issue #854.
