# PCMA-v1 factual learning and calibration — NONCONFIRMATORY engineering checkpoint

**Date:** 2026-10-17. **Not a frozen protocol. No confirmation seeds or performance claims.**

## Lineage
Stacked on PCMA draft PR #1501, which itself descends from original screenshot-inspired SI v26 draft PR #1499. Canonical `tam_research/si_original_hybrid_v0.py` remains unchanged with Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`. V26 remains strict NO-GO for its unchanged tiny hybrid; v25 duplicate seed 117214 is consumed and quarantined. No other model track or `main` files are modified. No paid compute, scientific attempt, background job or merge.

## This checkpoint actually implements
- Causal reward examples from **executed** sequential actions. Input contains only public pre-action observation, chosen action and earlier *observed* in-session rewards. Each current query reward channel stays zero and the target is returned separately; logical-time validation rejects future rows.
- An original SI hybrid and a matched Transformer reward forecaster with identical observation/action/modality encoding and reward heads. The original canonical architecture is imported unmodified.
- A bounded `fit_factual_predictor` developer utility (at most 32 updates), operating exclusively on factual rollouts and requiring no oracle/latent labels. **No parameter-count and active-FLOP equivalence is claimed** beyond shared wrapper shapes.
- Frozen deployed forecasters: their pretrained weights stop updating. A shared, bounded online PCMA memory and adapter can persist across session restarts. All outcomes for *unchosen* candidate actions remain unobserved.
- Cheap nonprivileged classical comparators: Beta-smoothed exact-match value lookup and online TD(0). These are necessary controls but **not yet a sufficient strong baseline panel**, especially without exploration and change detection.
- A feedback reliability gate that learns empirical neural-vs-memory trust from the squared errors of **executed chosen actions**. This is a low-compute factual-error controller, **not** a learned neural probe-purchase controller or proof of causal generalization.
- A sequential runner giving each agent its **own** world. Identical initial world seeds do not produce fictitious action-wise counterfactual results when agents choose different actions.

## Limits of current engineering validation
1. The original world generator is still a deliberately simple **toy**. The `long` family remains a placeholder, and current reward/observations permit cheap classical methods to dominate. Disjoint train/test generator laws, 4–8-step/longer OOD suites, and process-isolated leakage security have not been completed.
2. The reward-predictor developer routine is **not** a registered scientific training run, benchmark, or full-capacity training plan. CI unit tests may use tiny fixed numbers for interface verification; they cannot be promoted to evidence. The measured ability of any agent on novel worlds remains **UNKNOWN**.
3. Fixed neural/memory fusion and feedback error gating are engineering designs. No calibrated uncertainty-triggered, information-value-aware, learned probe controller is present. The toy probe may offer little utility for the reward function and needs an independent test of causal usefulness.
4. New models may differ in active compute even at roughly similar parameter count. Profile gradients, FLOPs proxy, inference/update latency, storage and actual probe cost before freezing a comparison.
5. Baselines must additionally include recent-window and change-detection policies, fair exploration/control budgets, GRU+PCMA, retrieval-only and adapter-only, shuffled/action-blind memories, no-probe, privileged *clearly-labeled* upper bound. A control with extra compute must be included if needed.
6. Next scientific protocol requires new, verified **unused** seeds reserved before execution, immutable code hash and thresholds, OOD partitions, and no retries/resumes. Do not reuse any v25/v26 seed or quarantined result; verify all existing GitHub results and consumed ledgers before scientific execution. No GPU without explicit authorization.

## What counts as progress
This change can satisfy code correctness and governance gates only. The next experiment must show value above cheapest factual-memory/TD alternatives AND hybrid-specific gains over a similarly capable Transformer equipped with the same PCMA. A positive narrow result would justify *independent replication* rather than a breakthrough declaration.
