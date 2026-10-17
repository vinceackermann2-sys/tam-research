# PCMA-v1 prospective first-arm randomization — DEVELOPMENT-ONLY DRAFT

**2026-10-17; new work on PCMA draft PR #1505. This is NOT a scientific preregistration or confirmation.**

## Why two-step data alone are insufficient
A policy that adaptively chooses probes in states it considers difficult cannot naively compare realized probe returns to realized non-probe returns. The state/action histories are selected differently; the cheap online `OnlineFactualTwoStepPolicy` explicitly provides only *observational* averages. Moreover, a probe uses a full step and costs 0.1; comparing it solely against the fee is invalid. Correct reward units for the simple engineering world must include **two real action opportunities**: the return of a chosen first action and its chosen follow-up.

## Additional instrument (development only)
`tam_research/si_pcma_v1_probe_randomization.py` draws the *first* action uniformly from four arms (guesses 0,1,2; probe 3), with publicly registered mathematical first-arm propensity **0.25 each**, independent of hidden context conditional on pre-action history. After the actual first transition, the continuation is **frozen** `FrozenPublicContinuation((0,1,2))` (or a separately specified immutable public-symbol permutation); it makes one **real** guess based only on its next public symbol. Both actual rewards are logged and the probe receives fee 0.10 and gives up its first opportunity to guess. Disjoint nonoverlapping two-step pairs remain inside session boundaries.

All returned `RandomizedFactualPair` objects expose a stable world/session/pair grouping key, the public pre-action observation, chosen arm, known assignment probability, first/second factual transitions and fixed continuation mapping. Duplicate grouping keys are rejected when summarizing. Grouping keys are for later cluster-level inference; a row-per-step confidence interval would be pseudoreplication. The evaluator internally owns the hidden OOD world object. No unexecuted action reward is written, and no latent rule family is returned to the learner.

The collector is **development-only** and refuses `heldout` configurations *before* world construction; unit tests use deterministic engineering fixture integers, not scientific seeds. A simple arm-summary helper reports counts and observed means per public symbol/action, leaving missing arm values as `None`. It intentionally computes no p-values, confidence intervals, fitted effect thresholds, or architecture rankings.

## What this DOES and DOES NOT identify
Random first-arm assignment with known propensity and a fixed continuation *can support* a causal intention-to-treat comparison of two-step returns for that **specific continuation and sampling distribution** if a separate predeclared independent-world study controls confounders, trajectory dependence, sample counts, interference and estimation uncertainty. It **cannot** prove that a probe is universally useful, that PCMA learns to use probe data optimally, or that unseen rule families generalize. Repeated decisions within a world are not independent trials; analyze uncertainty at the randomized-world/training-seed level. Some action-dependent observation noise streams differ by policy, so exact paired counterfactual world equality is not promised.

The default public-symbol continuation is merely a cheap fixed rule and may fit some rule permutations more than others. Predeclare all allowed continuation policies, randomization, causal estimand and test world generation before spending scientific seeds; do not use heldout performance to select the continuation.

## Remaining prerequisites
1. Independently auditable frozen source/train budget and registry of newly unused science seeds across all past branches and failed/timeout attempts.
2. Scientifically credible heldout evaluation method and process-isolated policy access, not arbitrary Python attribute access in the world process.
3. Same-information outcome-supervised hybrid/Transformer/GRU controls, matched compute including higher-budget Transformer; classical fixed, memory-only, change-detector and two-step/no-probe baselines.
4. Fixed credible OOD generators, uncertainty and memory necessity ablations, sample efficiency, probe value attributable to evidence, stable long-horizon performance, energy/runtime accounting.
5. Predeclared stop/no-go gates (including a margin versus **strongest** cheap control and matched Transformer) and a complete attempt ledger; no retries, GPU, scaling or merge authority from CI.

**V26 unchanged hybrid scientific stop stands. PCMA performance remains unknown. Passing engineering CI is not a scientific result.**
