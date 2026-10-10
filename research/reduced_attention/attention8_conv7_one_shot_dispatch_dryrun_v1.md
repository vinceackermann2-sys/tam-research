# Conv7 one-shot engineering dispatch: CPU dry-run v1

**No engineering run is authorized.** This is an in-memory state-machine rehearsal, not a Modal execution harness. It contains no Modal client or trigger, no persistent writes/reservations, no training, no GPU allocation and no consumed seed. A completed rehearsal is not scientific or engineering evidence.

## Grounded prior design

The completed attention-8 2B scientific runner reserved its attempt on a durable Modal volume **before** any paid H100 call, required first-attempt-only dispatch, set retries=0, and ran matched models in order. An error stopped subsequent model dispatch. That behavior is the governance reference for this separate proposed engineering experiment; no old runner, workflow, marker or result is changed.

## Unapproved three-model engineering order

1. Fresh 101,803,520-parameter Transformer (baseline).
2. Original 101,795,328-parameter attention-8 v2 (parent control).
3. New 101,803,536-parameter attention-8 + causal Conv7 (candidate).

The draft pilot specifies 3,052 optimizer steps per model (200,015,872 exposures) with the first 3,052 steps of the 30,518-step full-2B cosine. It uses the primary Modal account only. The engineering seed, hard cost ceiling and paid authorization are **unset**, so the preflight must report that real dispatch is blocked.

## Synthetic one-shot lifecycle

The module accepts explicitly labeled synthetic events only:
- Exactly one synthetic reservation must precede any model start.
- Each model has exactly one synthetic start, then either complete or fail, with no concurrency and no skipped comparator.
- A failure is terminal; a retry, resumption, second reservation, duplicate start, wrong model order, or event after terminal state is rejected.
- A successful complete-three-model synthetic trace is still labeled NOT_EVIDENCE and retains all paid-run / GPU / real-seed flags false.

Unit tests cover clean transitions, incomplete traces, interruption after each of the three models, duplicate and out-of-order actions, non-synthetic events and bypass attempts. The runtime also calls the already merged source-lock readiness checker, which validates exact Git blobs and the draft proposal before reporting the blocked planning state.

## What a future real execution harness must still do

This simulation is **not** sufficient for paid work. A separate reviewed runner and exact workflow must add actual immutable source/tree/harness binding; primary-account authentication; externally verifiable unique issue/job ID and seed; durable exclusive reservation with concurrency control before paid GPU allocation; hard per-model timeouts and conservative all-in spend-accounting; durable GPU start/error/result snapshots; strict stop-on-error and no-retry/no-resume; training/evaluation code and dataset fingerprint checks. Test failure conditions without GPU where possible.

Because Modal GPU billing and compile overhead are variable, a modeled budget alone is **not** a guaranteed spending cap. Any real run additionally needs explicit user authorization, an unconsumed engineering seed, approved maximum spend, and a verified external cost/timeout policy. A hypothetical engineering pass does not authorize 2B science, replication, or 250M/5B scaling. Previous scientific seed 60232 remains permanently consumed.
