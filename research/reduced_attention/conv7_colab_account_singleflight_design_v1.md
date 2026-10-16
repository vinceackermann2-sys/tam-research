# Conv7 Colab Pro account-wide single-flight: safety design (zero GPU)

Status: **DESIGN ONLY; NOT IMPLEMENTED OR AUTHORIZED FOR EXECUTION**. This note is additive and does not change the source-locked Conv7 experiment, the pending Colab worker PR #1446, or any scientific attempts.

## Confirmed source-level race

In PR #1446, `scripts/colab_pro_issue_worker.py` reserves a **per-job** global Git ref (`refs/tags/colab-attempts/<identity>`) and a **per-machine** `fcntl.flock`. Two workers on different machines with *different* job identities can both pass `account_snapshot()` (zero active assignments), each create its own global job tag, and then independently call `colab run`. The second `account_snapshot()` at lines 453-456 is still a read, not an account-wide atomic exclusion. A simultaneous allocation is therefore not ruled out by source code. Whether Colab itself permits overlapping sessions is **not** established by this CPU-only audit.

A distinct issue exists when two workers share a predictable `chatgpt-issue-<number>` session name: a losing worker may run `colab stop` in `finally` even if it never invoked `colab run`. PR #1446's successful historical CI does not cover these proposed fixes.

## Safety invariants for a future reviewed implementation

1. **No allocation without ownership.** All Colab research jobs for the same actual Google account must first acquire the *same* globally atomic account claim, regardless of GitHub issue, model, machine, worker state directory, or script. Per-job tags remain permanent, separately.
2. **Claim before allocation, fail closed.** Acquire the account claim before any command that might allocate. If acquisition fails, do not invoke `run`, `stop`, or artifact collection for another worker's session. A worker may stop only a session it actually attempted to start and whose unique session ID is durably recorded.
3. **No automatic time-based takeover.** A lease timeout, stale heartbeat, machine crash, lost network, or missing local receipt does *not* prove a Colab session is gone. Because the Colab CLI lacks a demonstrated server-side fencing token, auto-expiring a lease could allow overlapping allocations.
4. **Recovery requires external proof.** A stuck claim must block new jobs until an authorized operator verifies zero assignments and all associated session IDs stopped, reconciles receipts and permanent attempt tags, and performs a separately logged release. Unknown account state means no release.
5. **No unsafe release by loser.** A process that never acquired the account claim must not release it. Even the owner must not release it before successful teardown, artifact readback where required, and fresh account-state verification. An ambiguous teardown leaves the claim held.
6. **Crash durability.** Record unique owner token, account scope, session ID, attempt identity, and immutable claim evidence before allocation; recovery must never reuse a consumed job ID or seed.
7. **No paid authority.** This design does not approve Colab allocation, GPU spending, replication, a scientific seed, source mutation, or experimental scaling.

## Deterministic CPU test contract (future successor PR)

- Two workers / two distinct valid jobs / same account / both see idle: at most one may reach `colab run`.
- Two workers / same job: permanent per-job tag rejects the second independently of the account claim.
- Worker loses account claim: zero `run` and zero `stop` commands.
- Claim holder crashes before allocation, during allocation, during artifact export, or during teardown: no other worker allocates.
- Unknown or nonzero active assignment, missing account OAuth, or account mismatch: no allocation and no claim release.
- Manual release attempted with stale evidence, wrong owner token, or unverified session: rejected.
- Successful verified teardown and authorized release: a **different** approved job may subsequently acquire the account claim; the prior attempt tag remains immutable.
- Failure or cancellation of first-attempt CI freezes that exact PR head; never rerun or amend it.

## Implementation boundary and practical limitation

A create-once account tag is a simple conservative *safety gate* but permanently blocks further work until an explicitly reviewed recovery/release mechanism exists. GitHub per-job tags are **not** an account mutex. A time-based distributed lock is unsafe without Colab-side fencing. Before implementation, select and audit a durable atomic claim backend and an authenticated account identity; do not assume a GitHub issue comment, local `flock`, or two read-only session checks provide atomic exclusion.

## Scientific continuation

The original three-model Conv7 proposal remains zero-GPU review only. Preserve the source-locked matched Transformer, Attention-8 and Conv7 comparisons, heldout NLL, measured throughput, unique approved seed/job/budget, and independent replication. A synthetic concurrency check is engineering evidence, **not** model-quality evidence.
