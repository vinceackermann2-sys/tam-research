# CORTEX-S v0 scientific preregistration

Status: **draft to freeze before any paid scientific seed is executed**.

## Separation / provenance

Project namespace: `cortex-s-v0`

Do not reuse TAM/AERA trigger prefixes, result paths, checkpoint names, or scientific seeds. Do not interpret any CORTEX-S result as evidence for TAM/AERA, or vice versa.

Exploratory/consumed seeds: `123`, `124`.

Reserved fresh scientific seeds, untouched as of this preregistration draft:

- seed 1: `48131`
- seed 2: `48132`
- seed 3: `48133`

If any reserved seed is used before the protocol is frozen, mark it consumed and replace it before scientific execution.

## First paid stage: 25M-class seed-1 falsification run

The first paid run is deliberately small. Its purpose is to falsify the architecture cheaply, not to prove a breakthrough.

Matched conditions:

- same tokenized training/validation bytes;
- same tokenizer and sequence boundaries;
- same seed-derived batch stream;
- same optimizer family, LR schedule, warmup, weight decay, gradient clipping, precision, batch and token budget;
- parameter count mismatch <= 0.5%;
- no model receives extra labels or hidden state unavailable to another baseline except the architectural state being tested;
- report both equal-token and equal-wall/GPU-time comparisons;
- checkpoints/results use only `cortex-s-v0/...` paths.

Required baselines:

1. causal Transformer;
2. the current repository recurrent/hybrid baseline that is already scientifically valid at this scale, if reusable without changing its frozen semantics;
3. CORTEX-S v0;
4. CORTEX-S ablation with persistent state reset at every sequence boundary;
5. CORTEX-S ablation with expert recurrence disabled.

The paid seed-1 run must not proceed until true top-k sparse dispatch is implemented or throughput claims are explicitly excluded.

## Metrics

Primary language metrics:

- held-out NLL at equal tokens;
- perplexity;
- tokens/second after warmup;
- peak accelerator memory;
- held-out NLL at equal accelerator time.

Mechanism metrics:

- expert utilization entropy and fraction of experts actually executed;
- retained recurrent-state sensitivity under controlled resets;
- long-horizon state-machine generalization at lengths 2x and 4x the training horizon;
- fast-memory one-shot association accuracy without base-weight mutation;
- calibration/uncertainty metric if the uncertainty head is trained.

Safety kernel tests are reported separately. Passing them is a systems boundary check, **not evidence that a superintelligent system would be safe**.

## Seed-1 progression gate

Seed 1 may authorize seeds 2/3 only if all of the following hold:

- no NaNs/divergence or silent reset/memory leakage;
- parameter mismatch <= 0.5%;
- CORTEX-S equal-token held-out NLL is strictly better than the matched Transformer;
- CORTEX-S equal-time held-out NLL is no worse than the matched Transformer;
- true executed expert fraction is <= 50% for the sparse block being claimed;
- length-4x state-machine final accuracy exceeds Transformer by at least 15 percentage points;
- resetting persistent state removes a measurable portion of the long-horizon gain;
- disabling recurrent expert computation removes a measurable portion of the gain;
- all result artifacts are durably written before the success marker.

Any failure stops paid progression and is recorded as a negative result.

## Multi-seed candidate gate

Only after fresh seeds `48131/48132/48133` all satisfy the seed-1-quality conditions may the project be called a **breakthrough candidate**. Even then, that label additionally requires:

- a 50M-class replication with new fresh seeds;
- a second data mixture/domain;
- at least one modern non-Transformer recurrent/state-space baseline;
- continual-learning experiments measuring new-task gain and old-task forgetting;
- long-context/state tests not structurally trivial for recurrence;
- ablations that isolate which mechanism causes gains;
- cost/throughput accounting that includes routing and memory overhead.

No result in this project alone is sufficient to claim "safe superintelligence".
