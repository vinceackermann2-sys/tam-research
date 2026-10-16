# Original SI v12 — Causal hypothesis-belief augmentation results (2026-10-16)

**Scientific decision: NO-GO for the frozen belief-augmentation mechanism as a robust transferable solution; NO-GO for original SI architectural advantage.** Giving all models a hand-specified, causal posterior over the polarity and change of a hidden sensor helped when the family was in the posterior hypothesis set, but impaired performance on unseen types of change. This is *not* general change-point discovery, a breakthrough, or authority for paid GPU scaling.

## Identity, protocol, and attempt accounting

- Isolated v12 branch `experiment/si-original-hypothesis-belief-v12-20261016`, stacked on original SI v11 PR #1483 → original screenshot-inspired hybrid PR #1459; no changes to `main`, other research tracks, GPU/Modal/Colab or automations.
- **Protocol COMMITTED before all confirmation runs:** `docs/SI_ORIGINAL_HYPOTHESIS_BELIEF_V12_PROTOCOL_2026-10-16.md`. Exploratory smoke seed **104000** used 2+2 steps and a 50+60/512-world small sample, excluded from confirmation; filter diagnostic seed **104002** also excluded. The unfavorable double-change results were observed in development and disclosed before the freeze, with no subsequent tuning.
- New frozen executable `scripts/si_original_belief_filter_v12.py` locally executed **Git blob `fc160944420fe35ec9cf2f626def07c75a86656a`**, SHA256 `3e457137c1f5cf655acdc665d0a847df778ee6349b2a99198c77d03e672b0703`, verified byte-identical to GitHub. Original canonical 298-line SI model `tam_research/si_original_hybrid_v0.py` blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`; v8 environment `233dff4d660da142fbb57a91cd06b0c4ec17a0d0`, fixed v10 contextual critic `c8ade2177f495e0448638cc80d98dc6adde4fb51` and corrected v11 drift evaluator `1eea8c3ce72952388b9c79b8d299c23b75ca3351` verified locally. All tests in `tests/test_si_original_belief_filter_v12.py` **6/6 passed** locally, including future-probe/hidden-state isolation, model parameter parity, posterior normalization, unknown change detection, true logged-arm gradient, and double-flip benchmark finite checks; branch exact-head CI separate.
- **Fresh, frozen confirmation seeds 104211, 104212, 104213 each completed exactly ONCE**, each exit0 / stderr 0 bytes / complete raw JSON of 13,146 / 13,167 / 13,153 bytes; **zero failures or replacements**, zero retries. CPU PyTorch 2.10+cpu, 2 threads. Per seed six models (hybrid, Transformer, smaller GRU × raw, belief), each 180 actor +200 critic output-head updates, batch96; same seeded training episodes/initialization for within-backbone raw/belief. Evaluations **2,048 previously unseen worlds per family per seed**, same worlds/goals across models and references. Mean model training seconds reported below; no equal FLOP or GPU claim.

## Task, causal mechanism, comparisons and limitations

Six calibration observations pair noisy target and reference signals; current query and optional paid trusted probe precede final binary action. Reward is correct final action minus probe cost. Bayes hypothesis filter explicitly knows candidate patterns {stationary, one polarity change before calibration index 2,3,4}, each class having fixed training prior P(stationary)=0.5 and P(change)=0.5. It **integrates out six private binary calibration truths and sums over all possible patterns**, never accesses the actual sensor polarity, true query mode, unpurchased probe observation, or private change time. Its features are P(z=+) and P(any sensor change), injected *at query positions only* into two otherwise-unused existing 7-channel token slots, equally for all model classes. Neural models learn executed binary outcomes from action+correctness feedback and separate contextual expected success of bought/skipped probes using **randomized observed arm only**. Hidden simulator labels/Bayesian probe choices are never targets of the neural optimizer. This explicit analytic filter is **hand-designed known-family prior structure**, not self-discovered epistemic uncertainty.

Models: exact OriginalSIHybrid **64,614 params** vs matched TransformerControl **64,637**; GRU **10,283**, not parity matched. Original hybrid attention, recurrence, MoE, refiner active; persistent external episodic store, continuous learning, real modalities absent. Both raw and belief variants instantiate exactly the same per-architecture params. Extra Bayesian feature inference has CPU cost and may be badly misspecified.

Train on stationary `base` and one-flip `one_flip` environments only (alternating minibatches), then evaluate previously unseen world instances of those plus three OOD families:
- `early_flip`: hidden single flip at previously unseen calibration index 1 or5.
- `double_flip`: two opposite polarity changes, a **model-class violation** for the frozen Bayes filter (the real current polarity differs from hypotheses).
- `weak_calibration`: less informative sensor signals.
Strong comparisons include a mixture-model Bayes reference on each family (knows prior hypothesis family but **not actual world/switch time**), stationary-assumption Bayes, always and never buy. The mixture-model Bayes is not a universal oracle under unseen patterns; in particular its double-flip model is wrong.

## Main 3-seed mean realized net reward (higher better)

| Model and input | Stationary base | Single flip | Unseen early flip | Unseen double flip | Weak calibration |
|---|---:|---:|---:|---:|---:|
| Original hybrid raw | 0.768069 | 0.766328 | 0.752508 | **0.720862** | 0.760606 |
| Original hybrid + belief | **0.791887** | **0.792517** | 0.732285 | 0.635630 | 0.761803 |
| Transformer raw | 0.768284 | 0.758228 | **0.754277** | **0.741455** | 0.764212 |
| Transformer + same belief | **0.793237** | **0.798646** | 0.730576 | 0.618185 | **0.765937** |
| GRU raw | 0.753824 | 0.761475 | **0.758961** | **0.758583** | **0.763074** |
| GRU + same belief | **0.785251** | **0.787224** | 0.739278 | 0.643861 | 0.757163 |
| Bayesian mixture filter reference | **0.822991** | **0.822830** | 0.750794 | **0.584399** | **0.786353** |

**Paired effect of giving the engineered feature, 3-seed mean**:

| Architecture | Stationary | One flip | Early flip | Two flips | Weak calibration |
|---|---:|---:|---:|---:|---:|
| Hybrid belief - raw | **+0.023818** | **+0.026189** | -0.020224 | **-0.085232** | +0.001197 |
| Transformer belief - raw | **+0.024953** | **+0.040418** | -0.023701 | **-0.123269** | +0.001725 |
| GRU belief - raw | **+0.031428** | **+0.025749** | -0.019683 | **-0.114722** | -0.005911 |

**One-flip paired per-seed belief gain (preregistered 104211/104212/104213):**
- Hybrid **[+0.0293,+0.0211,+0.0281]**, **3/3 wins**, mean +0.02619.
- Transformer **[+0.0465,+0.0383,+0.0365]**, **3/3 wins**, mean +0.04042.
- GRU **[+0.0187,+0.0259,+0.0327]**, **3/3 wins**, mean +0.02575.

**Normal/probe behavior and Brier**: Base mean purchase fractions hybrid raw 0.95+ vs belief **0.697**, Transformer raw ~0.95 vs belief **0.669**, GRU raw 100% vs belief **0.672**. One-flip belief buy fractions hybrid **0.729**, Transformer **0.707**, GRU **0.678**, all interior; full-model base logged-arm conditional-Q Brier hybrid/Transformer **~0.128**, one-flip **~0.130/~0.129**; meets 0.16 base/one-flip criterion. Under early flip Brier worsens to hybrid **~0.170**, Transformer **~0.170**; under double flip to **0.239 hybrid**, **0.244 Transformer** (GRU 0.224), exposing reward-belief miscalibration.

Mean CPU train times: original hybrid raw **5.086s**, belief **5.609s**; Transformer raw **4.132s**, belief **4.476s**; GRU raw **1.141s**, belief **1.633s**. Inference filter overhead not separately profiled; identical train parameter counts do NOT mean matched active FLOPs. Hybrid belief ~1.25x Transformer belief training wall-time.

## Predeclared gates — no post-hoc promotion

**Mechanism usefulness full gate: FAIL.** On in-family one_flip, both full models exceed +0.025 net reward vs same-seed raw, win 3/3 seeds, preserve base and have good base/oneflip probe fractions/Brier. But mandatory OOD criterion is **at least 2 of 3** unseen family outcomes no worse than -0.02; for the hybrid early flip -0.020224 (*marginal but still below locked threshold*) and double flip -0.085232 both FAIL, only weak_calibration passes. For Transformer early flip -0.023701 and double flip -0.123269 FAIL. GRU likewise harmed on double flip. **Do not round -0.020224 into a pass or ignore the two-flip family.**

**Original hybrid advantage gate: FAIL.** Hybrid+belief base 0.791887 vs Transformer+belief 0.793237, and one_flip 0.792517 vs Transformer 0.798646. Does not win by required +0.015, and loses familiar-world mean with little advantage; is not reliably superior overall. Hybrid's smaller degradation under double flip is a relative *robustness observation*, not success because its own absolute reward falls 0.085 and raw/GRU outperform it.

**New-law transfer: FAIL.** Filter's hypothesis set excludes true double flip; its polarity inference can be confidently wrong, and the corresponding Bayesian reference gets only ~0.5844 reward, below raw neural policies. Prior diagnostic showed double-flip polarity inference accuracy around 31% despite apparently high change probability. This reveals **model-class misspecification**; providing an explicit belief feature can be *worse* than letting a neural model interpret raw history.

## Next bounded research question

Do not scale v12 or claim SI. The next zero-GPU test, if desired, should compare **unknown-model change-point filtering / model-class uncertainty** that explicitly keeps probability mass on an **unknown-other model**, with a conservative fallback to raw observations or direct raw neural action whenever the observations do not fit learned hypotheses. Critically, train/test over independent multiple-switch distributions and compare exact Bayesian multi-switch/particle filtering, matched Transformer/GRU and cost-aware always/never controls. A mechanism must *detect unsupported observations* before trusting confidence over a closed hypothesis set; naive Bayesian probabilities do not cover unmodeled worlds. Original full-stack hardware/compute claims and multimodal/continual learning remain untested.

**$0 paid GPU/Modal/Colab credits, no background automation, no PR merge/main changes.** Preserve all raw per-seed files and original frozen protocol. Require GitHub CI exact PR head and independent review; negative result is part of the record.
