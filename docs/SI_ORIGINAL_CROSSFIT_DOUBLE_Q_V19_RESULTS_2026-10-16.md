# Original SI v19 — Cross-fitted double-Q from logged action outcomes (three-seed CPU results, 2026-10-16)

**Locked verdict: task/control validity PASS; narrow crossfit prediction/reward improvements PASS as subsidiary endpoints; overall learned sequential-decision gate FAIL; original SI hybrid advantage FAIL.** No SI breakthrough, no GPU/Modal/Colab spending, no automation or main merge.

## Frozen source, protocol and attempt accounting

- Protocol `docs/SI_ORIGINAL_CROSSFIT_DOUBLE_Q_V19_PROTOCOL_2026-10-16.md` committed **before any new confirmation attempts**, including unfavorable development seed `111010` and fresh attempt seeds. Isolated branch stacked directly on CI-passed original v18 PR #1491 and canonical original SI hybrid PR #1459, not any separate architecture project. No main changes, no other research tracks affected.
- Full original 298-line canonical model `tam_research/si_original_hybrid_v0.py` **Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`** unchanged. Original exact v18 same-Q trainer/critic `scripts/si_original_logged_value_v18.py` Git blob `9d6155a27dac3c98502114f4f905dda647ca2172`; v17 task `scripts/si_original_active_experiment_v17.py` blob `aa60524483117237d9929a0e44da87a3d10dd8c5` unchanged.
- **V19 locally executed executable** `scripts/si_original_crossfit_value_v19.py` **Git blob `933512894318a00afd6ee33136cc477116a129cb`**, SHA256 `ada041074f1f304ec8a75a308ca4ddb5cf7b99241f6190875821f9f70792a815`; **tests** `tests/test_si_original_crossfit_value_v19.py` **Git blob `4c422b40db34dc7ec8c9631f8359beaa0760c48b`**, SHA256 `1ce2389fd83aa011260b097df7d39201863a5ee1ebfa7bd52a2337672905b594`. Both GitHub blobs byte-identical to local file, **7/7 local tests pass**, covering disjoint actual logged-arm gradient, independent A/B selection/evaluation, exact source and full actor/backbone equality to v18, no future/hidden-probe input, OOD cost-only world mutation and no teacher purchase labels. One *test-only* assertion typo repaired before preregistration; no model change, no successful confirmation before correction.
- Development ONLY, excluded: smoke `111000`, integrity tests `111001–111005`, **full-budget dev `111010`** (160+240 updates, 256 twin pairs per family): crossfit base reward ~.6981 hybrid/.6578 Transformer versus same-Q .6779/.6453, reduced Q MSE. Both crossfit full models strong-first ~0%, while known-model Bayes strong-first 97.3% under discounted-strong OOD. **No hyperparameter or threshold tuning after seeing development**.
- **FRESH, PREREGISTERED, ONE-TIME** CPU engineering seeds **`111211,111212,111213`**, each attempted once, unique reserved seed directory created before compute, completed with **exit0**, empty stderr, complete ~18KB JSON; **no re-run, replacement, resume or retry**. CPU PyTorch2.10+cpu, two threads. Per seed 6 learned conditions: 3 backbones × frozen v18 same-Q vs v19 disjoint double-Q; **160 full-actor +240 head-only critic AdamW updates, batch96** per condition. Evaluations exact same **512 coupled opposite-label twin PAIRS/family/seed** =1024 held-out worlds for each of 4 families, same worlds and costs across all conditions.
- Models: original SI hybrid original backbone 64,614 params; matched Transformer original backbone 64,637; GRU smaller unmatched. v18 task-head totals **64,638 / 64,661 / 10,307**, v19 each adds **8 additional Q head params** (64,646 / 64,669 / 10,315). The v19 extra head is constructed *after* actor training so unchanged v18 and new v19 are actor-byte identical at equal seeds. This is NOT equal-effective-Q-samples, inference FLOP or energy parity: each v19 Q head learns on half of available second-arm logged examples. Extra fitted estimate and joint readout have cost.
- V19's two Q1 heads are updated on disjoint subsets of actual logged second-arm outcomes (selected buy/skip only); first-stage bootstrap selects using critic A but evaluates using B, with detached target minus paid first cost. At deployment Q1 outputs average A/B. Simulator calculates correctness only after agent executes final action; hidden labels and unpurchased probes aren't neural inputs and analytic Bayes action labels NEVER enter this learner. However bootstrapped Q0 is NOT an observed counterfactual reward and can remain biased. This is feedback-supervised known task learning, NOT emergent general reasoning.

## Exact three-seed mean realized net reward (higher better)

| Policy | Base | Higher costs OOD | Weaker sensors OOD | Strong-probe discounted OOD |
|---|---:|---:|---:|---:|
| Analytic two-stage Bayes | **0.747880** | **0.636165** | **0.628440** | **0.847829** |
| Original hybrid — v18 single-Q | 0.63036 | 0.47346 | 0.54237 | 0.71685 |
| **Original hybrid — v19 crossfit** | **0.65558** | 0.48432 | 0.55529 | 0.76426 |
| Transformer — v18 single-Q | 0.61791 | 0.46936 | 0.52604 | 0.68800 |
| **Transformer — v19 crossfit** | **0.66474** | 0.48406 | 0.54536 | 0.73848 |
| GRU — v18 single-Q | **0.67430** | **0.58220** | **0.58023** | **0.69046** |
| GRU — v19 crossfit | 0.65979 | 0.54497 | 0.57121 | 0.68777 |

Simple fixed one-probe normal control: fixed cheap ~**0.6903**; fixed strong ~**0.6998**. The crossfit hybrid **0.65558** and Transformer **0.66474** remain BELOW a simple fixed strategy, despite Q calibration improvement. In new strong-discount OOD, Bayes favors strong-first **97.8%**, fixed strong-only scores ~**0.8464** versus fixed cheap-only ~**0.6385**. V19 hybrid still chooses strong first only **28.6%**; Transformer **~0.0%**, and both lag fixed strong by large margins.

**Base heldout randomized *actually observed arm* Q1 MSE (lower better):** hybrid v18 **0.117** -> v19 **0.107** (improvement **0.01037**), Transformer v18 **0.169** -> v19 **0.152** (improvement **0.01725**). This improvement is real within three paired engineering seeds but inconsistent per seed: first seed Q MSE **WORSENS** for hybrid and Transformer. GRU v18 ~0.110 vs v19 ~0.107 (small gain), but GRU reward decreases by ~0.0145.

## Per-seed compulsory reporting and locked gates

**Normal reward delta v19−v18 per seed (111211,111212,111213):**
- Hybrid **[−0.004196, +0.078997, +0.000879]**, mean **+0.025226**, wins **2/3**.
- Transformer **[+0.000000,+0.123653,+0.016827]**, mean **+0.046827**, wins **2/3** (first is a tie, not a win).
- GRU **[−0.062232,−0.009563,+0.028263]**, mean **−0.014510**, wins **1/3**.
- Full-model crossfit hybrid normal rewards **[0.64661,0.69468,0.62545]**, Transformer **[0.64468,0.67214,0.67739]**. Hybrid wins **2/3** but mean crossfit hybrid **0.65558** < matched Transformer **0.66474**, a **−0.00916** margin.

**Normal first strong / second purchased rate per seed (crossfit):**
- Hybrid first strong **[0.8%,89.5%,0.0%]**, second purchased **[98.9%,0%,99.6%]**. Hybrid's high-first mean ~30.1% is entirely driven by ONE outlying model seed and does not meet per-seed nondegeneracy.
- Transformer first strong **[0.2%,0%,0%]**, second bought **[100%,75.4%,75.8%]**. All second-purchase rates above allowed 65%.
- Mean CPU wall training time full actor+critic: hybrid same-Q **8.63s**, crossfit **8.72s**; Transformer same-Q **7.04s**, crossfit **7.04s** (reference CPU timing only). Hybrid/Transformer ~1.24×, no proven equal-FLOP efficiency or SI resource benefit.

**Gate A task/distribution validity PASS:** original Bayes base reward exceeds best fixed ~.6998 by >.025; strong-discount Bayes chooses strong first 97.8% (>80% required), and fixed strong-only outperforms fixed cheap by ~0.208 (>0.10). Twin/private-probe invariance tests pass.

**Gate B useful independent double-Q acquisition FAIL despite partial numeric improvements:** both full backbones clear base mean gain +.015, win2/3 paired seeds, average heldout Q1 MSE gains >=.01 and no mean OOD losses. **But every-seed first strong >=10% FAILS** (two hybrid seeds, all Transformer seeds); every-seed second-buy rate 5–65% FAILS (nearly 0 or nearly 100%); strong-discount strong-first >=50% FAILS; strong-discount reward within0.03 of fixed strong-only FAILS; beat best fixed normal by+.015 FAILS; within0.045 of Bayes normal FAILS. The method remains unable to reliably value which FIRST experiment to perform or whether a second is worth paying for.

**Gate C original SI competitive superiority FAIL:** hybrid v19 0.65558 < Transformer 0.66474 normal mean, far below required hybrid+0.015; despite 2/3 seed wins, average is negative. OOD differences vary and hardware-efficiency unproven.

**Gate D model identity and causal feedback integrity PASS** within synthetic task restrictions. Both source files exact checked, all seven tests pass, no direct Bayes purchase-label learning or unpurchased probe leakage.

## Conclusion / CPU-only research direction
The double-Q split reduced average second-stage MSE in two ~65k-parameter networks, but **value estimates for the FIRST acquisition decision and heldout cost sensitivity remain catastrophically unreliable**. A near-perfect second-stage estimator would still fail if first-stage Q0 is biased by insufficient logged coverage/bootstrapping or the frozen action policy differs by first-probe type. Future bounded CPU tests should first **measure separate first-arm Q0 calibration against randomized observed rollouts without privileged counterfactual labels in the optimizer**, inspect per-arm coverage, then try fitted Q evaluation/cross-fitting or direct randomized first-stage logged reward targets versus Q0 bootstrap on exactly paired worlds. Compare truly simple linear contextual value estimators and matched Transformer/GRU, use more independent seeds, report active compute. The original full hybrid has not been justified for paid GPU scaling. No main merge, autonomy automation or superintelligence breakthrough is authorized. Keep draft; require exact-head CI and review.
