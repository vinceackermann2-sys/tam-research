# SI Original — Causal Surprise Router v2, frozen CPU engineering protocol

Date: 2026-10-16. Dedicated branch `experiment/si-original-surprise-router-v2-20261016` stacked on PR #1467 -> original screenshot-inspired hybrid SI PR #1459; **not EHW**. Zero paid GPU/Modal/Colab work, no other research branches or main modified.

## Context and purpose

Prior original SI hybrid and fixed-mask probes showed **long attention better in stable worlds**, but **short local attention better after abrupt rule changes**. A trainable soft mixture failed to exhibit meaningful change-dependent gating and cost more compute. A no-history control confirmed this simulator does reward attention to past observations.

This pilot tests a distinct and falsifiable attention-horizon mechanism: **strictly causal surprise routing with ONE attention operation per layer**, not a two-path blending mechanism. This is an isolated world-model submechanism, not the full original hybrid stack.

## Method

`scripts/si_original_surprise_router_v2.py`: same shared 2-block width32, four-head causal attention + dense FFN world predictor across all four settings. Four candidate masks: `no_history`, `local` (last 6 timesteps), `global` (full prefix), and `surprise` (detector-selected local/full per query and per episode). **Same initial weights, stream, AdamW settings and nominal parameter count**. The surprise model has a parameter-free ridge-based causal prediction-error detector. For timestep t>=12, fit a regularized state/action -> next-state estimator on *earlier completed* transitions [max(0,t-18):t-5], and evaluate its squared-error increase on the most recent five completed transitions [t-5:t]. If new-minus-old MSE exceeds the **fixed** threshold 0.09, query t uses 6-token local attention, otherwise full-prefix causal attention. Targets used to compute routing at t are past observed states up to **x_t**, NEVER next-state x_(t+1). No actual change time, regime label, or future query target enters the gate. Ridge detector is **not differentiable**, has no extra trainable parameters and requires prefix solves per timestep; counts/complexity and CPU runtime must be disclosed. Single SDPA call per layer but **not proven end-to-end compute efficient**.

World #1: random action-conditioned dynamics with per-episode unknown 3x3 transitions and 4x3 action-effect matrices, stable/switch at randomly sampled t=8..18; training includes 50% stable and 50% switching, 24 steps/episode. World #2: unseen `delayed` simulator where the **previous action** is a dominant effect, tested at stable and switch without training on this family. Never use simulator labels to route.

## Development and frozen confirmation

Pre-freeze engineering smoke used seed **96300** (3 updates, batch3), finite backward passes, strict causal-prefix perturbation test and a detector routing sanity check. The development detector fired on approx 10.7% of stable late observations versus 72.0% of switching late observations; **diagnostic, not confirmatory evidence**. Detector threshold 0.09, cutoff t>=12 and window6 are frozen before held-out tests. Local code SHA256: `188421d89c3d38edda443bf6b2e98d2d9550ee843392d8d86373ec59bdf28561` before confirmation; verify GitHub source identity before interpreting run outputs.

**Preselected fresh CPU engineering seeds: 96311, 96312, 96313**. One execution per seed; no rerunning consumed attempts. All four candidates trained for **150 steps**, batch16, length24, 2 layers, 32-wide, 4 heads, AdamW LR .001, weight decay .01, grad clip1, 2 CPU threads. Identical generated data streams/initial weights for all four modes. Test 3×32 independent held-out worlds per (base/delayed) × (stable/switch) condition and seed, evaluation base seed=seed*1,000,000+94,000 with +7 per batch.

**Primary dual endpoint:** (1) held-out `base_stable` full-sequence MSE and (2) `base_switch` late-half MSE, each compared against best fixed candidate and no-history. Go for further **CPU** research only if surprise beats BOTH of the best fixed models on both endpoints in mean across 3 seeds and wins >=2/3 paired seeds per endpoint, without >1.5× median training time or routing collapse. Strong no-history and persistence controls required. No claim of breakthrough even if passed.

**Secondary required:** unseen delayed-family switch and stable MSE, routing fraction stable vs switch, number of instantiated trainable params, CPU wall time; transparent report of all seeds. Detector must fire meaningfully more in switch than stable; this condition alone is never sufficient for success. If the score fails, STOP/REDESIGN THIS ROUTER as-is; do not rationalize or spend Modal dollars. Do not infer any general intelligence/SI from synthetic dynamics.

This gate does not establish equal effective compute because the surprise detector solves small linear systems. Record all failures including infrastructure. Source/seed/budget frozen, no launchers, no automatic retries, no paid scientific reservation or authorization.
