# Original screenshot-inspired SI / PCMA-v1 — Published bandit action-dependent triage

**Date: 2026-10-19 | Status: Engineering/development only | No neural evaluation, scientific run, paid compute or approval.**

## Provenance and binding evidence
This draft starts at CI-passed original SI/PCMA external memory-positive-controls PR #1516 (which descends from #1515, #1514, #1513, #1511, #1510, #1509, #1508, #1507, #1505, #1504, #1503, #1502, #1501 and the original v26 branch). Canonical original `tam_research/si_original_hybrid_v0.py` Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` must stay byte-identical; `main` must not change. The 64,614-parameter original SI v26 scientific **NO-GO** remains untouched. The consumed v25 117214 repeated attempt is still quarantined. The custom PCMA history generator's weak dev signal remains **NO-SCALE**.

External POPGym 1.0.7 `RepeatFirstEasy`, `RepeatPreviousEasy(k=4)`, and `CountRecallEasy` positive controls showed that exact cheap public history already reaches +1.000 average return on all tasks. These are useful elementary memory QA, not evidence for novel hybrid superiority; perfect classical controllers make an architectural advantage impossible on those task reward metrics.

## Why this published task is a more relevant falsifier
POPGym's independently authored `MultiarmedBanditEasy` has **10 actions, 200 decisions per episode, hidden arm payout probabilities resampled each episode, and publicly observed actual binary payouts**. Agents must learn which arms pay and balance exploration/exploitation. Unlike `RepeatFirst`, merely saving a single public value does not solve the episode. A *cheap* classical Bayesian or upper-confidence-bound learner may still dominate expensive neural architectures; that is the intended scientific falsifier.

The original PCMA 3-symbol/4-action fixed-6/16-step reward predictor is NOT directly compatible with this 10-arm/200-step benchmark. This branch makes **no fake claims of neural PCMA training or transfer**; a separate externally reviewed, frozen common adapter and training protocol is required before neural comparison.

## Fixed nonprivileged classical benchmarks
`tam_research/si_pcma_v1_external_bandit.py` contains four nontrainable, fixed-in-code online controllers:

| Control | Uses only |
| --- | --- |
| `fixed` | Always choose action 0 (no learning) |
| `round_robin` | Deterministic action cycle, no reward learning |
| `ucb1` | Own executed arm's empirical binary payout + `sqrt(2 log(t)/count)` |
| `beta_thompson` | Own arm's successes/failures with Beta(1,1) posterior and separately seeded sampling |

The policy callback receives **only binary public observation (0 or 1)**, and updates from **one actual executed chosen-arm event** with action, public payout, time and reward. It never receives the environment object, `get_state()`, `bandits`, oracle optimum, unexecuted-arm rewards, or Gymnasium `reset/step info`. **Critically, this published environment returns all true hidden arm probabilities in its `info` dictionary. The runner deliberately discards those privileged fields.** Any later evaluator must maintain that same explicit public-only boundary. The code checks real payout/reward compatibility, step ordering, one pending action, complete horizon and no hidden-field entry.

The development-screen episode IDs are fixed prospectively in code at **41–48**, eight independent world episodes per policy. Each gets exactly 200 chosen-action/feedback opportunities. Report only each arm's observed factual history to its own policy, not a reconstructed optimal action. Benchmark reward is used as returned, summed over 200 steps, and averaged over **independent worlds**, not 200 correlated actions. Resetting each environment with the same episode ID fixes initialization but action-dependent RNG consumption means two divergent policies do NOT observe identical counterfactual payout noise. Do not interpret these as paired causal treatment outcomes or quote statistical certainty.

The isolated PR CI installs `popgym==1.0.7`, runs the exact unmodified published task and four fixed controls once as a bounded CPU development printout, then runs full repository pytest. No tuning on that printout, no retries as new evidence, and no scientific seed/attempt reservation. The numbers will be recorded verbatim in PR discussion after CI. Strong classical outcomes are **negative evidence for a uniquely hybrid-superior information-acquisition claim**, not a score to beat via retrospective benchmark changes.

## Research decision gate
**Stage A (this PR):** does a published action-dependent partially observable environment execute reproducibly under a strictly factual/public feedback boundary? Compare very cheap fixed, round-robin, UCB and Bayesian controllers descriptively. Engineering PASS if scripts and full suite pass, irrespective of which policy wins. No neural PCMA verdict.

**Stage B (not yet authorized):** explicitly adapt hybrid, same-PCMA Transformer, larger-compute Transformer and GRU to identical 10-arm/200-step public histories, bounded memory/state and fair frozen feedback/optimizer budget; include these cheap online-learning controls. Freeze evaluation seeds and independent world-level uncertainty outside all code and development choices. If cheap classical controllers explain the effect or an original hybrid provides no practical gain over matched Transformer, record **scientific NO-GO** for PCMA under that design.

**Honest status**: original v26 NO-GO; handcrafted PCMA world NO-SCALE; published simple memory control PASS; published bandit controller diagnostic in development; **PCMA neural advantage UNKNOWN**. All work isolated, unmerged, without paid GPU, new scientific seed or main changes.
