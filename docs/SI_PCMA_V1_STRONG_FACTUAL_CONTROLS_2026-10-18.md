# Original SI / PCMA-v1 — strong factual cheap controls (2026-10-18)

**Status: DEVELOPER ENGINEERING ONLY. No heldout experiment, no new science seeds, no paid compute, no breakthrough.**

## Root and preexisting negative evidence
This draft successor stacks on PCMA IPC PR #1511, originally rooted in the user's screenshot-inspired full SI hybrid. Immutable original source `tam_research/si_original_hybrid_v0.py` Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` remains unchanged. Original unchanged v26 hybrid remains **NO-GO**; the v25 invalid duplicate of consumed seed117214 remains quarantined. None of the stacked research branches has been merged to `main`.

The original v25 scientific result had already shown that a **simple same-data per-action constant-value control captured nearly all apparent learned first-probe improvement**. That is a major warning: merely fitting a neural reward head or bolting on public memory is not a plausible uniquely SI architecture breakthrough unless it defeats these cheap comparators. A new family of cheap controls is therefore a *falsification requirement*, not a weaker reference.

## Implemented code and exact information parity
`tam_research/si_pcma_v1_strong_controls.py` introduces three independently trained, frozen factual value predictors:

| Baseline | Fixed inputs | Offline fitting |
| --- | --- | --- |
| `constant` | Chosen first action only | One smoothed mean of **real two-step net** per action (4 values) |
| `symbol` | Current public observation symbol + chosen first action | 3×4 smoothed conditional means, with explicit nonprivileged priors |
| `ridge` | Current public symbol, first action, 16-step-normalized remaining budget, and recent prior executed feedback (up to last 7 transitions) | Deterministic low-dimensional L2 ridge fit on **real paired net returns** |

All three derive supervised targets from the **same randomized first-arm development logs** as the SI hybrid, matched-input Transformer, and compact GRU. A first action is one of guess0/1/2/probe3, the second action uses the same precommitted public-symbol-only `FrozenPublicContinuation((0,1,2))`. Observed target is actual `reward_first-fee_first + reward_second-fee_second`; probing also sacrifices a guess and does not make future guessing free.

The ridge observation/history features receive only public **past factual** feedback; the current/future outcome is not encoded. Model access to factual past history is limited to the last 7 transitions, exactly like `causal_tokens(max_context=8)` for the neural query. No private rule, latent context, unexecuted action reward, known optimal action, Bayesian teacher or `heldout` family enters fit. The fit-once model is immutable; deployment controls may update only a shared bounded `TwoStepPCMAState` rank-2 residual and paired factual memory.

`FrozenClassicalTwoStepPolicy` uses the same frozen continuation, two real actions per decision, actual paid-probe opportunity cost, bounded persistent memory, rank-2 online update budget, optional no-probe ablation and restart/reset control as `FrozenNeuralTwoStepPolicy`. Each gets the same one-action-at-a-time public IPC/feedback interface and net-reward accounting; no architecture is given an unobserved-arm outcome.

The three controls are **cheaper and not parameter/FLOP-matched**; that is intentional. If cheap controls achieve the relevant behavior, the novel architecture is unnecessary. Do not require cheap models to spend wasted equal compute. Conversely the hybrid/Transformer parity claim requires a *separate complete active-compute audit*; equal nominal parameters and number of updates are insufficient.

## Validation, boundaries, and fair comparison
Contract tests exercise: identical training `CausalRewardSample` records between the classical and neural dev collection path, target nonleakage, causal-history/future-feedback guard, contextual lookup with missing-arm smoothing, deterministic ridge fit, actual IPC subprocess action costs and action counts, shared memory/update accounting, same immutable second-stage choice, no-probe/memory-off/adaptation-off ablation, bogus action/feedback rejection and heldout training refusal.

**Not a scientific result:** tests intentionally do not publish outcome means, p-values, architecture rankings, causal effect claims or heldout metrics. Any development-world reward is nonconfirmatory. The dev logger and fitter are in the same Python process during training, so a malicious caller could supply fabricated `CausalRewardSample` objects; only the restricted development collection entry point constrains provenance by ordinary API contract. The IPC deployment worker similarly is not a hostile-code security sandbox.

## Remaining decisive work, prioritized
1. **Frozen fair experiment preregistration**: Strong classical baseline suite above plus recency/change-detector/TD/no-probe, matched-PCMA Transformer and a higher-compute Transformer, compact GRU, fixed/random references, persistent-vs-reset, memory-only/adapter-only and shuffled factual retrieval.
2. **Real development training**: substantial frozen budget and independent heldout environments designed so trivial 3×4 lookup cannot easily solve them; otherwise STOP and classify task as non-discriminative. Existing 6/16-step 3-hidden-symbol six-rule toy is only an engineering fixture.
3. **Full compute cost and validity**: actual train+inference FLOPs, memory and timing under controlled hardware, not the current partial profiler estimate.
4. **Independent evaluation, not toy leakage**: separate protected heldout generator, authenticated evaluator, matched policy feedback and decision budgets, independent world/training seeds with clustered uncertainty and no retrospective tuning. No seed is safe without full cross-project attempt-ledger reconciliation and a durable one-time reservation.
5. **Real, precommitted STOP/GO**: PCMA requires a prespecified meaningful margin over the strongest cheap control **and** matched-compute Transformer, without family collapse. Failure is a no-go for the successor under those conditions; it does not disprove general superintelligence.

**Verdict remains PCMA-v1 UNKNOWN. Original unchanged SI v26 remains NO-GO. Neither engineering CI nor identical synthetic trajectories establish an SI breakthrough.**
