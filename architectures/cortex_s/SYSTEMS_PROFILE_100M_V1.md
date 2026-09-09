# CORTEX-S v0 — 100M systems profile v1

Status: **preregistered diagnostic only, before any profile-v1 H100 dispatch**.

## Motivation

The grouped production preflight v3 is consumed. Issue `#802`, workflow `34345899535`, job `102447296828`, source `3a9538228d100a40818030a64b5b9eaf7311f94a`, used engineering seed `2026090905` and measured:

- compiled grouped production throughput: `241,834.8588 tok/s`;
- peak VRAM: `45.4400 GiB`;
- pure 2B training projection: `8,270.1063 s`;
- frozen `1.10x` overhead plus compile envelope: `9,463.8449 s`;
- frozen full gate: `<= 8,500 s` — **FAIL**;
- `full_run_authorized=false`.

No seed-8100 CORTEX 2B training was launched. That v3 decision is final for its frozen protocol and will not be reinterpreted.

The current production graph therefore needs roughly another systems-level improvement before a newly preregistered production preflight could plausibly pass. Instead of guessing at another optimization, this stage profiles the exact compiled grouped graph to identify where time is actually spent.

## Classification and authority

Classification: `ENGINEERING_SYSTEMS_PROFILE_ONLY`.

This stage:

- is not language-quality evidence;
- cannot authorize 2B training;
- cannot authorize a next systems candidate automatically;
- cannot change the v3 gate or reuse the v3 namespace;
- cannot consume paired scientific seed `8100` or reserved fresh seeds `48131/48132/48133`.

Any optimization suggested by the profile requires a separate preregistered candidate stage with a fresh namespace and seed.

## Seed and consumed history

Fresh profile engineering seed: `2026090906`.

Already consumed and forbidden for reuse:

- `910001`;
- `2026090901`;
- `2026090902`;
- `2026090903`;
- `2026090904`;
- `2026090905`.

The profile seed becomes consumed at the first H100 dispatch, not at CPU/static validation.

## Exact production shape

The H100 profile uses the merged production grouped CORTEX-S model:

- `101,778,112` trainable parameters;
- 24 layers;
- 8 experts / top-2 routing;
- logical expert width `338`;
- physical grouped BF16 runtime width `344`;
- context `512`;
- microbatch `64`;
- gradient accumulation `2`;
- `65,536` tokens per optimizer step;
- AdamW and the same production helper used by the v3 calibration;
- compile mode `max-autotune-no-cudagraphs`.

The profile uses the same immutable corpus path and issue-767 hashes.

## Measurements

After one compile-trigger step and three additional warmup steps:

1. measure 10 normal optimizer steps with no profiler active;
2. report throughput, optimizer-step latency, peak VRAM and finite loss;
3. profile two subsequent exact optimizer steps with CPU + CUDA activities;
4. retain the top 30 device-time events and the profiler table;
5. separately measure the current grouped physical-weight materialization path across all 24 MoE blocks: FP32 logical weights are physically padded and then cast to BF16. This probe reports pad/cast milliseconds per full-model forward and an estimated per-optimizer-step fraction at gradient accumulation 2.

The weight-materialization probe is diagnostic. Its time is not automatically assumed removable or additive to end-to-end speed.

## Single-use namespace

- trigger: `[modal-cortex-s-100m-systems-profile-v1]`;
- phase: `systems-profile-v1`;
- app: `cortex-s-v0-100m-systems-profile-v1`;
- result root: `/vol/cortex-s-v0/100m-systems-profile-v1`;
- H100 timeout: 15 minutes.

The Modal root contains `DATA_GATE.json`, `H100_DISPATCH_CONSUMED.json`, and `RESULT.json`. Presence of any corresponding consumed artifact fails closed.

## Zero-GPU gates before dispatch

Before merge/H100:

1. exact grouped 100M parameter count remains `101,778,112`;
2. all 24 MoE blocks are production grouped blocks;
3. logical width remains `338`, physical width `344`;
4. frozen corpus hashes and byte sizes are rechecked by the launcher before H100;
5. static tests prove seed separation, single-use namespace, no full-training function, and `full_training_authorized=false` / `next_stage_authorized=false`;
6. normal CI, CORTEX-S CPU gate, and the existing 100M/2B zero-credit gate must all pass on the exact PR head;
7. live `main` must still match the source used for merge and dispatch.

## Interpretation

The profile may identify a concrete bottleneck such as grouped weight materialization/casting, routing/packing, recurrent scan work, attention, optimizer work, or compiler-generated kernels. It does not establish that any observed self-time is fully removable.

After the result, stop. A candidate optimization is allowed only after its mechanism, semantic equivalence criteria, speed threshold, seed, namespace and stop conditions are separately frozen.
