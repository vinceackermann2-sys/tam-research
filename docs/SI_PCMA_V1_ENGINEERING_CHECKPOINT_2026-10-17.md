# Original SI PCMA-v1 engineering checkpoint — 2026-10-17

**Status: NON-CONFIRMATORY SCAFFOLD / DRAFT. No positive experimental result.**

## Provenance and isolation
- Stacked on the original SI v26 draft branch, PR #1499, canonical original hybrid source blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`.
- v26's stop/no-go for the *unchanged* 64,614-parameter tiny hybrid remains in effect. We are evaluating a different proposed mechanism, not reopening v26.
- Original v25 seed `117214` was consumed and subsequently duplicated in error. The duplicate remains quarantined under audit-only PR #1500; neither result may be interpreted as valid new evidence.
- V25/26 attempts and other research branches untouched. `main` is read-only. No Modal/Colab/GPU attempts, seed reservations, confirmation trials, automation, model scale, or merge.

## What this change actually implements
- Deterministic 16-hidden-state, four-public-symbol, four-action *engineering* world with an optional unannounced regime shift, noisy public observations, paid observation probe, and continuous world clock across sessions.
- Only a public `PublicObservation` and an executed-action `FactualTransition` are intended for policy consumption. The world object owns latent state/regime, and no counterfactual method is exposed. **This is an API-level invariant, not a security boundary**: Python callers holding the world object can inspect private attributes; evaluation must pass only public records to controllers.
- Capacity-limited factual memory (default 128), action-conditioned retrieval, monotonic event-time/provenance checks, and window reset separate from persistent memory reset.
- Rank-2 clipped online reward adapter, trained only from delivered *chosen-action* reward, per-session update ceiling and snapshot/restore. Adapter is *not* the canonical original hybrid's pre-trained fast adapter.
- Identical external PCMA state interface on the canonical original hybrid and the repository's Transformer control. Both neural backbones are **randomly initialized and untrained** in this engineering slice.
- Tests check session isolation, exact record schemas, probe cost, factual-action budget, duplicate rejection, independent ablations, deterministic replay and forward tensor sanity. These are correctness tests, not independent scientific seeds.

## IMPORTANT missing work before any scientific run
1. **Benchmark validity.** Current `long` family is a placeholder and needs a distinct preregistered horizon setting. The current task can be solved from observed symbol transitions; add varied generators, exact-match-friendly and compositional families, and an empirical test of whether cheap controls trivialize it. Do not use this as a confirmatory dataset.
2. **Learned control.** Train outcome-supervised hybrid and Transformer policies using the same visible inputs, memory, update budget and tool/probe costs. Design and test a learned uncertainty/probe controller; the current PCMA value method is only a simple heuristic.
3. **Fair competitors.** Add classical value lookup, online ridge/TD, change-detection, GRU, memory-only/adaptation-only/no-persistence, no-probe, shuffled-memory, action-blind retrieval and privileged labeled upper bound. Fairly measure active compute rather than claiming parity from parameter counts.
4. **No-leakage hardening.** Separate environment/private state from policy process or function boundary, deny arbitrary policy access to `PersistentWorld`, enforce factual event authenticity in the runner, fix/train/test generator partitions, and measure intervention effects only on delivered factual trajectories.
5. **Falsifiable protocol.** Freeze generator, reward normalization, exact code blob hashes, attempt ledger, heldout splits, thresholds, seeds, device/CPU budget and failure procedure in an independently reviewable commit *before* a new scientific attempt. Never reuse consumed seeds or rerun interrupted attempts. B/C/D gates remain provisional.
6. **Evidence.** Report original v26 negatives beside new results. A positive small-CPU test would at most establish a narrow mechanism worth independent replication, not an SI breakthrough or scaling authorization.

## Explicit engineering checklist (not a results table)
- [x] Original hybrid file not edited
- [x] Factual-only public feedback schema and no counterfactual method
- [x] Cross-session world persistence and independent memory/adapter resets
- [x] Unit tests for invariants and two shared neural interfaces
- [ ] Adequate OOD environment / generator partitions
- [ ] Trained matched neural and strong classical full comparison panel
- [ ] Learned confidence/probe controller
- [ ] Compute profiling / leakage threat model / frozen preregistration
- [ ] Any valid independent scientific confirmation

This document is intentionally not a preregistration or a breakthrough claim.
