# PCMA-v1 randomized factual acquisition — engineering checkpoint

**Status 2026-10-17:** DRAFT / NONCONFIRMATORY. NO scientific attempts, training seeds, paid GPU/Modal/Colab, scaling, merges, or autonomous loops.

## Research lineage and frozen negatives

This branch is stacked on original SI PCMA probe-value draft PR #1505, then fair controls #1504, OOD worlds #1503, factual learning #1502, PCMA foundation #1501 and canonical original SI v26 draft #1499. The original source `tam_research/si_original_hybrid_v0.py` blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` stays unchanged. The unchanged 64k SI architecture has a v26 scientific NO-GO. Consumed v25 seed 117214 and its invalid duplicate remain quarantined under audit PR #1500. No results or thresholds are retroactively revised.

## New randomized factual acquisition infrastructure

The new module `tam_research/si_pcma_v1_randomized_acquisition.py` separates an acquisition decision from its later factual continuation:

- Use ONLY `development` rule configurations. All `heldout` configs are rejected **before** creating the worlds or calling a continuation policy. Repeated scenario identifiers are rejected.
- At each two-action boundary, randomize the first action uniformly among three guesses and the paid probe, logging known assignment probability 1/4 BEFORE feedback. Actions occur in the real sequential simulator; this is not a counterfactual reward oracle.
- Execute precisely one fixed, nonadaptive **public** continuation policy in the next step for every first action. It consumes ONLY the real next observation and actually executed first transition; it is not passed latent context, true mapping, or a simulator instance. No second probes are allowed within these two-action pairs.
- Store both factual transitions, before context, first arm, assignment propensity, sessions, total observed two-action net reward, first/second logical times. Validate that second action immediately follows first in the same session and that actual fees/rewards exactly reconcile.
- Provide per-arm support counts and descriptive raw and propensity-weighted aggregates. These are **NOT confidence intervals, hypothesis tests, or confirmatory estimates**. Randomization permits causal first-action inference under a *frozen common continuation*, with caveats about within-world dependence, observed state distributions, finite action horizons and potentially heterogeneous rules.
- The default `FrozenSymbolContinuation` is only an engineering fixture. It maps public symbol 0→0, 1→1, 2→2. It has not been validated as optimal or fair under heldout mappings. A real study must choose/fit/freeze continuation on development-only data independently **before** test seed reservations.

## Why this matters

The prior two-step online controller in PR #1505 fitted return means after adaptively choosing the first action. Those observational means could reflect selection bias and changing downstream behavior. Random first-arm assignment and the same continuation policy isolate the first acquisition choice more rigorously. With probe cost 0.10, the experiment must include both the explicit fee AND the reward opportunity foregone on the probe step. Under some distributions buying information may have **negative value**, and a correct result must report that.

## Boundaries / unresolved issues

1. **No training or confirmatory evidence.** Unit tests use disposable development fixture integers. All heldout configurations remain out of the development logger. No scientific reserve/seed ledger has been written or executed. Repeated pairs within one world are not independent world-level replications.
2. **Causal interpretation requires locking the continuation.** A frozen public continuation must be identical across randomized first arms and cannot peek at latent mapping or change its weights in response to newly collected training data. A post-hoc optimized second action would bias apparent probe gains.
3. **No all-architecture comparison yet.** Need outcome-supervised development-only fitting, frozen checkpoints and identical data/model inputs for SI hybrid vs matched Transformer vs compact unmatched GRU, plus high-compute Transformer and strong classical controls.
4. **Fair resource measurement:** preregister active-forward/backward compute proxy, total trainable/frozen parameters, CPU latency including bought probes, storage/update cost, per-world action budget, and memory/adapter ablations.
5. **Independent heldout protocol:** fix scenario distributions/independent world seeds, exact source hash, train budgets, propensities, alpha/error bars by WORLD (cluster), absolute competence and architecture-uniqueness gates vs cheapest nonprivileged competitors. No post-hoc threshold/seed substitution. Confirm consumed-seed history across original SI, PCMA and other tracks before any fresh scientific reservation. Retries=0.
6. **Threat model:** this Python evaluator constructs a `OODRuleWorld` and passes only public dataclasses to the frozen continuation, but Python process-local private fields are not a security boundary against hostile arbitrary callbacks. For trustworthy adversarial leakage claims, isolate the evaluator in a distinct process and give the policy only validated serialized public data.
7. **Missing external capability:** synthetic rules cannot validate SI, lifelong multimodality, tool use or general reasoning. Even clean small-task gains justify only replication, not GPU scale or breakthrough.

## Engineering completion gate

- Review source contract for causal first-choice randomization, truthfully delivered factual transition and fee, same frozen continuation, no cross-session pairing, no heldout training collection.
- Pass exact-head original GitHub CI and verify unchanged original blob and `main`.
- **Stop at reviewable engineering if any guard fails.** A separate, prospectively frozen scientific study should be considered only after its complete control panel and independent test design exist. Unit-test pass is not a research verdict.
