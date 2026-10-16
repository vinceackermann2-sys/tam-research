# Original SI v15 — three-channel fault isolation, three-seed CPU results (2026-10-16)

**Verdict: narrow PASS for a hand-specified classical Bayesian single-fault control; FAIL for neural guarded repair and original SI architectural advantage.** No superintelligence breakthrough, no broad learned uncertainty, no GPU/Modal/Colab spending or automation, no merge or direct main change.

## Preregistration, reproducibility, attempts
- Frozen protocol `docs/SI_ORIGINAL_THREE_SENSOR_FAULT_V15_PROTOCOL_2026-10-16.md` committed **before confirmation**. Local full protocol SHA256 `9fb115194cb3e3b3418e00f6f3ad1f3a1ecd4f55aab25036717228ab33219ac1`. The *original canonical* full 298-line SI hybrid source Git blob **`575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`** is unchanged.
- Executed v15 source `scripts/si_original_sensor_fault_v15.py` Git blob **`34aadc62c36dd207169733001f67e6fc75ca83d5`**, SHA256 `2a87023cac58814de9b02778aeec8c80734fb8e4f299e846f71144762f9377ef`; tests `tests/test_si_original_sensor_fault_v15.py` blob **`9121e29a7dbcbff535c5cadfe5b7c47ba7b09d41`**. Both exactly locally byte-verified against GitHub; **six local integrity tests passed**, including no hidden fault-label or future-query leakage. A float32 probability assertion was relaxed from <=1 to <=1+1e-6 **before the protocol and confirmation freeze**, only to accommodate numerical rounding, no model/threshold change.
- CPU PyTorch2.10, two threads. Excluded development seed `107000` (2+2 updates) and tests `107010–107014`. **Fresh 107211,107212,107213 each completed exactly once**, return code 0, empty stderr, fully parseable result JSON (about 43KB each), no failures/retries/replacements. Each seed trains six models from identical matched streams: canonical OriginalSIHybrid 64,614 params, TransformerControl 64,637, smaller GRU 10,283 each raw+belief with unmodified v12 logged feedback algorithm, **180 actor +200 chosen-arm critic optimizer steps, batch96**. Evaluation uses **512 independent unseen worlds per family per seed**, same worlds for all conditions, 9 families, pooled n=1,536/family but only 3 independent model seeds. Low-powered engineering evidence; not formal scientific replication.
- A **third independently sampled noisy calibration sensor** provides actual additional information; its simulator knows hidden calibration truth only for generation. Fault detector uses only observed x/y/anchor and nominal strengths. Classical posterior integrates unknown per-calibration truth over **120 hand-specified** possible target/fault/polarity configurations (5 target histories ×4 reference states ×3 anchor states ×2 target polarities) with **fixed product priors**. Methods that use the third sensor pay fixed synthetic **0.025 reward per episode**, regardless of whether they substitute a reading; original neural raw/belief policies do not pay the added fee. The finite Bayesian model is **privileged by its known Gaussian likelihoods and priors** and is not learned. Neural guarded rule substitutes anchor into reference only if P(reference fault)>.55 **and** P(reference fault)-P(anchor fault)>.20. Neural policy never gets hidden labels or unpurchased probe.
- Classical cheap baseline uses last-three target observations and sign of normalized nominal-reference plus anchor consensus; both classical options pay same sensor fee. Neural comparison includes no-fee raw and belief, fee-charged v14 fixed disagreement repair and always-replace-anchor. Two trained raw+belief models and analytic posterior have extra parameter/FLOP/latency costs; active equal-FLOP matching not provided.

## Three-seed mean *realized net reward*, higher better

| Family | Hybrid raw | Hybrid belief | Hybrid v14 repair | Hybrid v15 guard | Transformer v15 guard | Majority control | Bayesian fault model |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stationary | **0.769** | **0.802** | 0.778 | 0.777 | 0.771 | 0.802 | **0.803** |
| One target flip | 0.774 | **0.799** | 0.773 | 0.773 | 0.771 | 0.801 | 0.782 |
| Two target flips | **0.718** | 0.610 | 0.584 | 0.585 | 0.582 | 0.564 | **0.760** |
| Reference flips late | 0.711 | 0.583 | **0.757** | 0.747 | 0.743 | 0.676 | **0.799** |
| Reference flips early | 0.762 | **0.789** | 0.772 | 0.773 | 0.772 | 0.816 | 0.810 |
| Third sensor flips | 0.770 | **0.798** | 0.559 | 0.769 | 0.765 | 0.676 | 0.796 |
| Reference noisy dropout | 0.734 | 0.696 | 0.754 | 0.765 | 0.768 | 0.706 | 0.799 |
| **Reference and third both flip** | **0.727** | 0.586 | 0.559 | 0.562 | 0.557 | 0.448 | 0.563 |
| **Target and third both change** | **0.722** | 0.628 | 0.664 | 0.610 | 0.598 | 0.671 | **0.766** |

Within three-seed means the baseline classical estimator's fault posteriors behave differently by family: reference late P(ref faulty) **0.694**, P(anchor faulty) 0.294; anchor late P(ref faulty) **0.342**, P(anchor faulty) **0.649**. The y-fault prior is asymmetric even in healthy episodes, so probability ordering is **not** evidence of perfectly calibrated fault identity. The learned guarded policy replaces the reference in about **69.2%** of reference-late, only **9.0%** of anchor-late, but fails simultaneous faults. Under simultaneous reference+anchor reversal, mean P(ref faulty) **0.098** and P(anchor faulty) **0.091**: both are understated and the model is confidently misspecified.

## Frozen gate decisions

**Gate A: classical specified fault-model usefulness — PASS narrowly.**
- Fee-adjusted Bayesian minus majority on late-reference reversal **+0.123**, on anchor reversal **+0.120** (required >=+0.015 each); stationary roughly tied.
- Bayesian minus *unpriced* raw hybrid on reference reversal **+0.088** and anchor reversal **+0.026** (required >=+0.015).
- Reconstructed per-world posterior ordering: **86.1%** of late-reference cases P(y_bad)>P(anchor_bad), **76.1%** of anchor-fault cases P(anchor_bad)>P(y_bad) (required 65%). **The criterion does not cover simultaneous-fault calibration**; simultaneously faulty reference+anchor gives near-zero marginal fault mass and low reward. **NO claim** of general safe fault localization.

**Gate B: guarded neural mechanism — FAIL for both full architectures.**
- Versus v14 same-fee rule, on anchor reversal hybrid improves **~+0.210** and Transformer **~+0.209**, easily meeting +0.08.
- But on late-reference reversal hybrid guarded **0.747** vs v14 **0.757**, Transformer **0.743** vs **0.754** (required new guard +0.015; FAIL).
- On simultaneous reference+anchor reversal, guarded hybrid **0.562 vs raw 0.727**, Transformer **0.557 vs raw 0.723** (max allowed raw gap .03: FAIL). On simultaneous target+anchor failure hybrid **0.610 vs raw 0.722**, Transformer **0.598 vs raw 0.726** (FAIL). Improvements in selected single-fault families cannot supersede simultaneous-fault failures.

**Gate C: original SI backbone superiority — FAIL.** Guarded hybrid-Transformer reward difference is only about **+0.006 stationary, +0.004 reference reversal, +0.004 anchor reversal**, each below prereg required +0.015. Mean CPU train time two-network original hybrid **10.505 seconds**, two-network Transformer **8.341 seconds** (ratio ~1.26x; temporal constraint passes but no measured efficiency win). Smaller GRU is a strong additional no-novelty control. Source and nominal per-network parameter parity do not establish comparable active FLOPs.

## Decision, hard limits, next gate
This is classical **Bayesian finite-hypothesis inference with manually supplied fault families/likelihoods**, not an emergent original SI capability. The observation model's simultaneous reference+anchor failure causes strong, misleading confidence, and a raw control is safer. Robust fault isolation of common-mode simultaneous failures is not guaranteed from three correlated readings without new identifiability assumptions or independent observations. The next bounded CPU question should test **posterior predictive checks/abstention and a truly held-out independent fourth diagnostic**, or determine impossibility cases analytically first, against explicit classical majority, known-model Bayes, cost-aware raw control, and matched Transformer/GRU. Stop unchanged v15 for GPU advancement. **No breakthrough, main merge, scaling, automation or paid compute authority.** Preserve the seed-level JSON, precise failure evidence, GitHub exact-head CI and code/protocol hashes. 
