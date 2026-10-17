# Original SI successor PCMA-v1 — Neural two-step probe-learning checkpoint

**Status 2026-10-17: ENGINEERING ONLY. Not preregistered. No science seeds consumed. No experimental result.**

## Exact lineage and negative evidence
This draft branch descends only from original screenshot-inspired SI PCMA probe-value draft PR #1505, following draft #1504 → #1503 → #1502 → #1501 → original SI v26 draft #1499. The canonical original SI hybrid `tam_research/si_original_hybrid_v0.py` MUST retain blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`. The old **unchanged** 64,614-parameter SI model's v26 no-go is FINAL for that toy benchmark line. The v25 duplicate consumed seed 117214 remains quarantined in audit PR #1500. Neither experimental negative is overwritten. `main` and other AI architectures are not modified.

## New mechanism and why it matters
Previous neural PCMA actor predicts *one-step* reward. A probe has immediate reward 0 minus fee 0.1, so greedy action selection will tend to reject it even when that information would improve a future action. Conversely, fee-only value calculations ignore the guess sacrificed by making a probe.

This checkpoint introduces `tam_research/si_pcma_v1_neural_probe.py`:
- `randomized_pairs_to_causal_samples` reconstructs in-session public history from actual randomized, nonoverlapping two-step factual pairs. Input tensors contain only previous chosen actions/rewards plus current **unlabeled** public observation/candidate first action. The current two-step net becomes a separate supervised target, never an input. World/session/pair IDs are used for continuity checks, never model features.
- `fit_development_two_step_predictors` uses the **same factual developer data** and **identical number of bounded gradient updates** for original SI hybrid, Transformer and compact unmatched GRU. It rejects heldout world configurations before collecting data. No model is fitted on heldout worlds. Up to **32** tiny CPU updates are permitted solely as a correctness/development utility; this is NOT a scientific training budget or replication. The GRU's unmatched size must be declared.
- `FrozenNeuralTwoStepPolicy` predicts the **two-action net return** for each candidate first action (guesses 0,1,2 or paid probe 3) and follows the SAME immutable, public-symbol-based second-action continuation for every architecture. Both real actions and their fees count. The original hybrid/Transformer/GRU backbones freeze during deployment.
- `TwoStepPCMAState` shares the SAME bounded pairwise factual memory (max 128 records) and rank-2 online residual adapter (8 updates per 16-action session), including actual probe-first outcomes. This is needed because the old one-step `ReversibleLowRankRewardAdapter` skips probes and cannot update their value. Memory-only, adapter-only, neither, no-probe, and restart/erased-memory ablations are possible. No hidden state, latent rule, true unchosen arm, oracle second action or fabricated rewards are ever passed into this mechanism.
- The existing `MeasuredFactualPolicy` fair accounting now recognizes the pairwise memory deque and updates, and reports per-policy action count, paid probes, fees, net reward, total parameters, decision/update timings and retained memory/adapter counts.
- CI tests check timelike ordering, no current/future target in model inputs, truly identical causal tensors, fixed public continuation, prohibition on unchosen outcomes, two-real-step feedback requirement, replay prevention independent of memory eviction, frozen neural weights, ablations, and development-only fitting.

## What this has NOT shown
1. **No trained performance evaluation or breakthrough.** Tiny CI train steps are API/gradient validation and must never be treated as scientific evidence, successful optimization, reward improvement or generalization.
2. **No statistically valid causal information advantage.** Randomized first actions and fixed public continuation make a future trial *possible*; current engineering fixtures are not a registered trial. One-step and two-step actor policies do not infer true counterfactuals. A learned first-stage policy needs robust out-of-sample heldout causal evaluation and a no-probe control with the same action budget.
3. **Not architecture-native differentiable retrieval.** This is a shared external memory and adaptation mechanism bolted onto all neural backbones, deliberately making architectural comparison falsifiable. It is not a uniquely original SI cognitive memory or general intelligence.
4. **No matched active compute.** SI hybrid and Transformer may have near-equal *parameter count*, but there is no verified active-FLOP equivalence. Compact GRU is intentionally unmatched. CPU timing diagnostics are not proof of equal compute. Must add an equal/greater compute Transformer baseline.
5. **No process-isolated policy sandbox.** The Python evaluator passes only public records by API, but private attributes can be inspected by malicious code if allowed to run in the same interpreter. A hardened separate process / restricted RPC is required for adversarial policy testing.
6. **No scientific seed registration.** All test fixture integers are developer constants only. Do not reuse old consumed seeds, including failed/canceled ones. Do not launch a paid run or scientific CPU trial from this branch.

## Proposed preregistration target (NOT FROZEN; must be reviewed/approved before an experiment)
- **Primary estimand:** mean net reward PER EXECUTED ACTION over independent heldout worlds, including paid sensor fee and opportunity cost.
- **Critical comparisons:** SI hybrid + PCMA against matched-input Transformer + same PCMA and higher-compute Transformer; best classical development-locked factual sliding, change detector and probe-value learners; GRU + PCMA and no-probe; persist-vs-reset; memory-only/adapter-only; shuffled memory.
- **Family robustness:** separately report stable, rule-shift, high-noise and 16-action horizon worlds; post-restart reward and shift recovery; no hidden rule/seed in public features; no selection/tuning on heldout.
- **Independent sampling:** use a world/training-seed as one unit of uncertainty, not each action; preregister clustered confidence intervals and genuinely disjoint preregistered new scientific seeds, with a durable unique-attempt reservation and retries=0 **before any scientific CPU or GPU**.
- **Stop:** absent an independently supported improvement over the strongest classical baseline AND matched Transformer under explicit cost/resource constraints, do not promote the mechanism, scale, or claim an SI breakthrough. Sample sizes, thresholds, wall-time caps, seed allocations and failure handling all remain **UNSET**.

## Immediate next blockers
Implement a scientifically separate, seed-locked heldout runner with immutable manifest and SHA provenance; audit historical consumed seed ledger; add strong classical controls trained under the same randomized dev logging; profile active FLOPs and proper training budget; implement end-to-end policy isolation and staged approval. **Until those exist, stop creating endless toy metric variants; this draft only delivers the previously missing trained two-step neural path.**

**Scientific verdict: PCMA-v1 UNKNOWN; unchanged original SI v26 NO-GO.**
