# PCMA-v1: factual two-step acquisition — engineering checkpoint

**2026-10-17 · DRAFT · NONCONFIRMATORY · No new scientific seeds / paid runs**

## Authority and lineage
This is a separate draft successor branch stacked on PCMA fair-comparison draft PR #1504, then #1503/#1502/#1501 and the original SI v26 NO-GO #1499. Do not alter the original screenshot-inspired 298-line hybrid source blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`. The v26 STOP for unchanged 64k hybrid remains authoritative; consumed v25 seed 117214 and its invalid repeated output remain quarantined in the v25 audit #1500. `main` remains read-only and no Modal/Colab/GPU or new scientific CPU confirmation is authorized by this checkpoint.

## Specific scientific flaw addressed
Previous greedy neural PCMA chooses among immediate predicted rewards. Its probe action (3) costs 0.1 and returns no immediate reward, but creates a trusted observation that may improve a **later** guess. Comparing 0.1 against a single guessing reward ignores the larger opportunity cost: probing also sacrifices one of a finite number of decisions. Conversely, a probe has no guaranteed benefit until the agent learns to use that information.

New `tam_research/si_pcma_v1_probe_value.py` supplies:
- **Nonoverlapping factual two-action outcomes**: observed chosen first action plus next chosen action, total return `(r_first - fee_first) + (r_second - fee_second)`. Terminal singletons never borrow next-session feedback.
- **`OnlineFactualTwoStepPolicy`**: a deliberately cheap, nonprivileged online control with Beta-like shrinkage priors, configurable first-action epsilon exploration, a learned actual two-step value per public observation/action, minimum factual-probe sample requirement, and a nonnegative margin before purchasing information. It chooses a *guess* for every second action and never probes on the last remaining step.
- **No-probe ablation**: exact same policy with `allow_probe=False`, requiring zero information purchase. `erase_persistent=True` clears retained factual values; ordinary restart keeps them across sessions.
- Tests of real fee + lost action, no two-step update before **both** executed rewards arrive, probe never forced by missing data, last-turn no-probe, strict pending feedback validation, no privileged rule/world imports and no crossing session boundaries.

## STRICT interpretation and remaining bias
The empirical two-step estimates are **behavior-policy-dependent observed returns**, NOT unbiased causal values of information. The first action is selected adaptively, so the logged first-arm groups can differ in context/history; random exploration contributes known stochastic coverage but the current online policy does not expose or use logged propensities. Its follow-up guessing policy also changes during training. Therefore a higher score for the learned probe arm is *not* evidence a probe caused an advantage over the unchosen guess.

A genuine scientific first-probe comparison must instead predefine a common continuation policy or uniformly/randomly assign first actions with logged propensities and use an unbiased/predeclared estimator with sufficient per-arm coverage; report confidence intervals by independent *world/training seed*, not individual actions. The true reward of an unchosen action must NEVER be passed to the controller, though an offline, clearly privileged oracle may compute upper bounds. Cost accounting must include fees **and** sacrificed action budget.

## Next scientific entry prerequisites — still NOT met
1. Complete a development-only randomized first-action acquisition data logger with locked continuation policy and explicit propensities; adequate first-stage arm support in every public context; a heldout split inaccessible to the trainer.
2. Train *all three* neural models on the same development-only factual observations under a declared fixed budget; freeze neural checkpoints; ensure the compact GRU is explicitly unmatched; fairly measure actual compute and storage. Add higher-compute Transformer control.
3. Implement ablations: persistent versus reset, memory-only, adaptation-only, shuffled memory, factually matched probe-only vs no-probe, action-blind memory, zero information/purchase and privileged exact-rule Bayesian upper bound.
4. Explicitly freeze an independently reviewable protocol: source and test blob hashes, test-world distributions, fresh independent unused seeds checked against full existing attempt ledger, fixed results schema, minimum effects versus *strongest* cheap and matched controls, confidence intervals, failure/termination criteria and retries=0.
5. Strengthen world/policy isolation to protect against malicious code; the current Python private fields and narrow call arguments are API-level checks, not a security sandbox.

Until a fully registered study succeeds, outcome remains **UNKNOWN for PCMA**, **NO-GO for original unchanged v26**. No claim of self-improvement, long-horizon planning, general SI or breakthrough.

## Stopping rule for this engineering branch
Do not create an endless succession of small benchmark tweaks or consume scientific seeds merely because engineering CI passes. After reviewing this PR, proceed either to a preregistered scientific-ready comparison with genuinely matched controls or record a clear feasibility block; do not reinterpret passing unit tests as evidence of superior performance.
