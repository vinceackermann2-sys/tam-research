# Original SI v18 — Logged-feedback-only sequential acquisition (three-seed CPU results, 2026-10-16)

**Predeclared gate verdict: reference/task validity PASS; independent reward-only sequential acquisition FAIL; relative original hybrid-over-Transformer comparison PASS as a numerical gate, but is NOT an architecture breakthrough or permission to scale.** Full learned policies score below a simple fixed single-probe strategy. The hybrid's apparent competitive margin is driven by two favorable, seed-sensitive comparisons of inadequate decision-makers; it requires independent replication and stronger absolute competence before attributing a useful advantage to original hybrid recurrence/attention/MoE.

## Source and attempt integrity
- Isolated branch `experiment/si-original-logged-sequential-v18-20261016` directly stacked on CI-successful v17 PR #1490, ultimately full original SI PR #1459. No direct main changes, other research track edits, GPU/Modal/Colab charges, background automations, or merges.
- **Preregistered protocol committed BEFORE new confirmation attempts**: `docs/SI_ORIGINAL_LOGGED_SEQUENTIAL_V18_PROTOCOL_2026-10-16.md`. It explicitly disclosed the failed full-budget development seed **110010**, the precise frozen source/training settings, and strict A/B/C/D gates before touching fresh seeds.
- Canonical unchanged original full 298-line model module `tam_research/si_original_hybrid_v0.py` Git blob **`575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`**. V17 source unmodified blob **`aa60524483117237d9929a0e44da87a3d10dd8c5`**. NEW *executed/GitHub byte-identical* source `scripts/si_original_logged_value_v18.py` Git blob **`9d6155a27dac3c98502114f4f905dda647ca2172`**, SHA256 `cdf3c8786a6cea27672e34050d5187cb7a7bebf778e94b1b2d44f99cc003acb2`. NEW tests `tests/test_si_original_logged_value_v18.py` Git blob **`da7cfd0d34f6c2281f252c90a3d9f1036d85572f`**, SHA256 `fccf668e22220cdac654aa0ea1d45dc1256be87d0b6b21436999810d8d7c0ce1`. **6/6 local tests pass** before/after confirmation (model hash, parameter parity, future/private signal isolation, executed-action+correctness feedback, selected-arm gradient only, and no oracle purchase labels inside train).
- Excluded development: smoke **110000**, integrity tests **110001–110006**, full-budget negative development **110010** (160 actor+240 critic steps, batch96, 256 twin pairs/family). Its base hybrid/Transformer logged reward was **0.679305/0.669539**, both bought cheap 100% of worlds, strong/second 0%. No change to experiment source or hyperparameters after seeing this negative.
- **Exactly three FRESH preregistered CPU confirmation seeds `110211,110212,110213`** each launched once with a durable attempt marker, finished process **exit0**, had **empty stderr** and complete JSON, no retry or replacement. CPU PyTorch2.10, 2 threads. Each seed trained three v18 logged-Q models plus three independently retrained v17 Bayes-teacher models; evaluated 512 exactly coupled opposite-label twin PAIRS per family =1,024 test worlds × three families, shared inputs and costs. Only **three independent training seeds**, not high-powered scientific replication. Individual raw JSON, exit markers, source hashes, and a machine-readable aggregate saved in the user evidence ZIP.

## Frozen training mechanisms and computational fairness

v18 uses exact full original hybrid **64,614** nominal backbone params and matched Transformer **64,637**, plus smaller unmatched GRU **10,283**; each receives identical *24 trainable task-head parameters*: hybrid **64,638**, Transformer **64,661**, GRU **10,307**. The full original attention/recurrence/top1 MoE/refiner remains, but persistent episodic memory, fast adapter, and multimodal/continual abilities are NOT activated.

- **No analytic Bayes purchase labels in v18 optimizer.** Random logged training interventions purchase actual available probes; final randomly executed action and correctness feedback reconstruct a training binary outcome. Train the final-action head and original backbone for **160 AdamW steps**, batch96. This is *outcome-supervised learning*, not fully unsupervised; the environment privately knows the simulated answer to generate correctness.
- Freeze backbone and final classifier. Train two-stage `Q1[skip/buy]` with **240 critic-head-only updates** from randomly selected and actually observed second-stage arms, actual terminal action correctness and actual second-probe costs. Train `Q0[stop]` from a randomly executed and observed no-probe outcome; train `Q0[first cheap/strong]` by bootstrapping the model's best predicted Q1 after purchasing that first probe, minus its actual cost. This is logged-action fitted Q regression, NOT counterfactual second-probe reward supervision. **However bootstrapping predicts values for unobserved second alternatives and can be overoptimistic/biased; it does not independently verify their true outcomes.**
- At deployment buy first argmax Q0, buy second iff Q1[buy]>Q1[skip], execute final action only with purchased signals. The known-simulator Bayes planner is used **only in evaluation**.
- Separate v17 control trains **240 full-backbone AdamW steps** per seed to imitate analytic Bayes purchase decisions and receives teacher labels that v18 does not. The teacher is an **explicitly advantaged training-signal control**, not fair evidence of an architecture's independent planning ability. Update counts (160 full +240 head-only vs240 full), training interventions/label type and active FLOPs are **not matched**. Report actual timing; no compute-efficiency conclusion from raw speed.

## Three-seed mean net reward (higher better)

| Method | Base | High probe cost OOD | Weak sensors OOD |
|---|---:|---:|---:|
| Known-model full two-stage Bayes | **0.73742** | **0.63264** | **0.62600** |
| Original SI hybrid — reward-only Q | **0.67164** | 0.58054 | 0.57851 |
| Matched Transformer — reward-only Q | **0.64825** | 0.55516 | 0.55906 |
| Smaller GRU — reward-only Q | 0.66937 | 0.60773 | 0.57518 |
| Original hybrid — independently retrained v17 oracle imitation | 0.72613 | 0.61454 | 0.59466 |
| Matched Transformer — v17 oracle imitation | 0.72711 | 0.61474 | 0.60444 |
| GRU — v17 oracle imitation | 0.72658 | 0.61192 | 0.59220 |
| Best fixed one-probe (choose best single baseline on each family) | 0.70445 | 0.62070 | 0.58791 |
| Never probe | 0.50000 | 0.50000 | 0.50000 |

**BASE, one-seed learned net reward, frozen seed order 110211/110212/110213:**
- Hybrid **[0.655396, 0.672747, 0.686763]**; Transformer **[0.672754, 0.641807, 0.630180]**; GRU **[0.669996, 0.675685, 0.662428]**.
- Paired hybrid minus Transformer **[-0.017357,+0.030940,+0.056583]**, mean **+0.023389**; hybrid wins **2 of 3**. This is unstable across seeds: the second-purchase policy in the last seed is wildly different from the first two.
- Hybrid base first cheap fractions **[0.94336,0.98438,0.98438]**, strong **[0,0,0.01562]**, second-purchase fractions **[0.00098,0,0.70996]**.
- Transformer first cheap **[0.94141,1.0,1.0]**, strong **[0.05859,0,0]**, second-buy **[0,0.07617,0.87598]**.
- GRU first cheap **[0.96875,1.0,0.84961]**, strong **[0.00781,0,0.15039]**, second-buy **[0.10352,0.00195,0]**.
- Means: hybrid base first cheap **0.97070**, strong **0.00521**, second purchased **0.23698**; Transformer cheap **0.98047**, strong **0.01953**, second **0.31738**. Reporting only mean second-purchase rates would **hide** the extreme seed collapse. Mean base normal Bayes gap hybrid ~0.06578, Transformer ~0.08917. Both are inferior to fixed single-probe strategy, so *better than the Transformer* is not evidence of good planning.

## Preregistered gates (no post-hoc threshold changes)

**A Task and controls PASS:** exact Bayes base reward 0.73742 > never by0.23742, > best fixed0.70445 by0.03297 and > always both by >0.04, with selective second acquisition. The twin/no-future tests pass and known-model Bayes is a strong reference.

**B Independent logged-feedback sequential experiment learning FAIL for BOTH full architectures.** Every-seed first high >=10% requirement fails badly (mean hybrid0.52%, Transformer1.95%; neither ever chooses high in most seeds). Every-seed second purchase between5%-65% fails for both: near 0% in one/two seeds, then 71% hybrid/88% Transformer in seed110213. Hybrid mean base 0.67164 is **0.03281 below** best fixed high-only0.70445, Transformer **0.05620 below**; both miss required +0.015 above fixed. Both miss required Bayes margin within0.045. No reward-only mechanism advancement or GPU authority.

**C RELATIVE original-hybrid superiority gate PASS as a narrow numerical gate only:** hybrid mean base 0.67164 > Transformer 0.64825 by **+0.02339** (required +0.015), paired seed wins 2/3, hybrid higher on BOTH OOD family mean rewards, and mean hybrid total CPU time <1.5× Transformer. **Crucially this does not rescue Gate B:** two weak models and a highly seed-unstable fitted critic may differ significantly without the hybrid being a competitively useful general architecture. The prior v3–v17 track usually failed original superiority; this one benchmark's relative pass is worth independent CPU replication but NOT a robust breakthrough, not general model superiority and NOT scale approval. The frozen experiment gives only three independent model seeds and unequal active FLOPs.

**D Causal integrity PASS (bounded):** original full model hash, future-probe mutation tests, logged chosen-arm-only regression, separate teacher control and executable Git blob verified. The optimizer uses observed correctness and a bootstrapped estimated unchosen Q1 value rather than actual counterfactual outcome; don't call it full online causal discovery.

## Final decision
**NO-GO for unchanged v18 fitted-Q reward-only sequential learning** because Gate B failed despite relative Gate C passing. Preserved useful hypothesis: original hybrid may be less susceptible than Transformer to this specific critic's third-seed overpurchase, but full synthetic controls and independent new seeds are required to establish any stable inductive-bias benefit. The next *CPU-only* attempt should isolate critic overestimation with double-Q/cross-fit targets, per-arm Q calibration and strong linear/GRU contextual value baselines; test on distinct hidden-rule worlds where a fixed cheap probe cannot trivially dominate. No paid GPU, scaling, merger, automation or superintelligence claim. Keep draft and check exact PR-head CI.
