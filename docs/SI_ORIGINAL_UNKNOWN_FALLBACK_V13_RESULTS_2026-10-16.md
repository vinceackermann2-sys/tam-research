# Original SI v13 — causal known/alternative model-class fallback: three-seed results (2026-10-16)

**VERDICT: NO-GO for robust fallback, NO-GO for original SI hybrid superiority, NO-GO for broad unknown-environment discovery.** All three fresh frozen CPU engineering seeds 105211/105212/105213 completed successfully exactly once; no failed attempts or retries; no paid GPU/Modal/Colab use, no automation, no main or other project modifications. The fixed fallback recovers reward vs an incorrect Bayesian feature on explicitly represented two-switch worlds, but fails a strict randomized-gate test, loses to a simple raw policy, and breaks on a truly unmodeled *reference sensor* change.

## Provenance / preregistration
- Frozen protocol: `docs/SI_ORIGINAL_UNKNOWN_FALLBACK_V13_PROTOCOL_2026-10-16.md` committed **before** confirmation runs. Full local pre-confirmation protocol SHA256 `a015cd5f37825753e316f0180a259b55a48cd47bd0607a9f0504589b9c391b85`.
- Source `scripts/si_original_unknown_fallback_v13.py`: exact GitHub/local Git blob **`51e76032747bcdf90f755164216a923f310dbed3`**, local SHA256 **`0192935e1144e7797d2f15fef84ba012d32507dd2afe773dfdc4798fbf7ce9d6`**. Tests `tests/test_si_original_unknown_fallback_v13.py` Git blob **`381d536e44d88be0695d24bed24cc7ec38995abd`**, byte-identical GitHub/local, **6/6 targeted pytest passed**. Source-verified full 298-line original hybrid Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`; frozen v12 trainer `fc160944420fe35ec9cf2f626def07c75a86656a`. v13 is stacked on **CI-passed v12 repair PR #1485**, not on the frozen failed CI v12 head.
- CPU PyTorch2.10, 2 threads, **three new engineering seeds 105211,105212,105213**, each exact single successful execution, stderr empty. Per seed six trained models: hybrid, Transformer, small GRU each raw and belief-augmented, **180 actor +200 critic updates, batch96**, alternating stationary/one-flip worlds only; all model conditions share identical seeded training streams and evaluation worlds. **2,048 heldout worlds per family per seed**, six families, 6,144 examples/family over three seeds; model-seed n=3.
- The raw vs belief choice is based solely on **observed** six calibration target/reference pairs, costs and known signal strengths. Posterior explicitly enumerates 32 target-sensor sign patterns (stationary+one flip at time2/3/4 = 4 known; 28 other patterns given alternative prior). Choose raw if P(alternative | calibration) >0.5, otherwise separate belief network. **Two full neural models** are used per architecture, plus Bayesian likelihood enumeration: hybrid full fallback **129,228 instantiated neural params** vs 64,614 for single model; matched Transformer fallback 129,274 vs 64,637. Original hybrid retains recurrence, local attention, sparse MoE/refinement but no activated persistent multimodal/continual memory. This is a hand-designed finite pattern expansion, **not autonomous discovery**.
- Negative control: **shuffled gate preserves exact fallback percentage per evaluation family** but randomly assigns it to different episodes; evaluator family metadata is used only to construct this offline negative control, *never* the deployed detector. Strong references: expanded-pattern Bayes posterior/value-of-information, stationary-model Bayesian planner, Bayesian always-buy and never-buy. Classically informed Bayes knows the likelihood family; neural models do not have hidden simulator variables or future unpurchased observations.
- All raw JSON outputs and failed/error logs (all empty for these seeds) retained in user evidence archive with machine-readable aggregate, tests, original model source and complete local README. No reruns or retrospective seed changes.

## 3-seed mean realized net reward (higher better)

| World family | Hybrid raw | Hybrid belief | Hybrid fallback | Hybrid shuffled | Transformer raw | Transformer belief | Transformer fallback | Transformer shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stationary base | 0.78131 | **0.80308** | 0.80087 | 0.79582 | 0.77630 | **0.80109** | 0.79996 | 0.79498 |
| Known one-flip | 0.76780 | **0.79262** | 0.78501 | 0.78425 | 0.75534 | **0.79368** | 0.78249 | 0.78258 |
| Early single flip | **0.73357** | 0.72397 | 0.72644 | 0.72707 | **0.74650** | 0.72916 | 0.73182 | 0.73878 |
| Previously unseen double flip | **0.70516** | 0.61224 | 0.68078 | 0.67753 | **0.73822** | 0.62448 | 0.70592 | 0.70351 |
| Weak calibration | 0.73973 | 0.74994 | 0.74885 | 0.74800 | 0.75151 | 0.75084 | **0.75581** | 0.75127 |
| Previously unseen *reference* drift | **0.65928** | 0.56022 | 0.58332 | 0.59364 | **0.71937** | 0.57136 | 0.60855 | 0.61853 |

Smaller GRU provides a strong additional control: base raw/belief/fallback **0.76148/0.79324/0.79267**; double_flip **0.75856/0.62504/0.71582**; reference_drift **0.76217/0.57179/0.62353**. The fallback still fails to beat raw.

**Posterior alternative detection rate** (same across architectures, because hand-coded): stationary 0.275; one flip 0.352; early flip 0.509; double flip **0.713**; weak calibration 0.488; changed reference sensor **0.334**. As designed, it is better at detecting represented target-sensor changes than reference-sensor law failure.

Expanded-model informed Bayesian reference rewards stationary 0.82595, one flip 0.81348, early flip 0.78560, double flip 0.77005, weak calibration 0.77398, **reference drift only 0.55229** (vs stationary Bayes 0.67467 and stationary always-buy 0.73108). Thus *the hand-coded expanded Bayesian reference also fails on truly unmodeled reference drift*. OOD logged-arm Q Brier under deployed fallback increases from ~0.125 stationary to ~0.205 hybrid/0.191 Transformer on double and **0.267 hybrid/0.252 Transformer** on changed reference, showing uncalibrated confidence.

**Mean CPU training wall-time raw + belief:** original hybrid **10.93s**, Transformer **8.85s**, GRU **2.86s** per seed. The fallback requires two models plus pattern enumeration. No total inference FLOP/latency measurement was instrumented, so it cannot be claimed to save compute; parameter parity holds only per model, not fallback-vs-single-model.

## Frozen gates — no after-the-fact relaxation

### Gate A: Robust fallback — FAIL for both full architectures
- **Partial recovery:** hybrid double-flip fallback minus belief = **+0.06854**, Transformer **+0.08144**, each wins 3/3 paired seeds; >.04 target. Fallback rate double0.713 >.60 and stationary0.275 <.40; base and one-flip losses vs belief less than .015. These subconditions PASS.
- **But still worse than raw:** double-flip hybrid fallback **0.02438 below raw**, Transformer **0.03230 below raw**, exceeding allowed .015 gap. FAIL.
- **Not detector-specific:** double-flip fallback minus shuffled = **+0.00325 hybrid**, **+0.00241 Transformer** (required +.015). FAIL. This suggests the effect may be explained largely by fallback mixing percentage, not useful per-example uncertainty sorting.
- **Truly unmodeled reference failure:** reference_drift hybrid fallback **0.07596 below raw** and Transformer **0.11082 below raw**, exceeding .02 maximum harm. FAIL. Detector falls back in only 33.4% of episodes.
- Each mandatory subgate had to pass. **Overall FAIL**.

### Gate B: Original hybrid competitive advantage — FAIL
Hybrid fallback base **0.80087** versus Transformer **0.79996** (+0.00091, not required +0.015), one-flip **0.78501** versus Transformer **0.78249** (+0.00252). Hybrid double-flip **0.68078** < Transformer **0.70592** and reference-drift **0.58332** < Transformer **0.60855**, both by >.015. Approx training time hybrid/Transformer ratio ~1.24, no GPU or active FLOP benefit.

### Gate C: Novel model-class discovery — FAIL
The broader 32-pattern model **already enumerates** possible two flips; outside the enumerated target-sensor class, reference-sensor drift is poorly detected and the expanded Bayes model also fails. None of this demonstrates discovery of new dynamical laws.

## Next research question
The mechanism's missing element is an *online predictive consistency check across multiple sensors*, **without presupposing which device might fail**, and a **conservative information strategy** that can distrust its own model and choose robust actions. A meaningful next CPU preregistration should compare out-of-model posterior predictive checking vs a no-Bayes simpler alternative, multi-sensor failure families withheld from training, classical online estimators and matched Transformer/GRU, while charging for actual active FLOPs and preserving the simple raw action baseline. **No paid GPU/Modal/Colab, no breakthrough or merge authorization.** Keep this draft; exact-head GitHub CI and independent review required.
