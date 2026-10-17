# Original SI v24 — Frozen-acquisition terminal-action bottleneck (2026-10-17)

**Decision:** A source/task validity **PASS**; B FINAL-CLASSIFIER BOTTLENECK **FAIL** for original full hybrid AND Transformer; C INFORMATION-ACQUISITION BOTTLENECK DIAGNOSTIC **PASS** for both (not a trained fix). No original hybrid architectural advantage, autonomous discovery, breakthrough or GPU scaling. Strong classical full two-stage Bayes and fixed strong-only policies still outperform all frozen-acquisition neural variants. The source and model were not post-hoc tuned on confirmation outcomes.

## Strict attempt / source / provenance
- Separate original SI research branch stacked on exact-head CI-successful **v23 draft PR #1496**, descendant of the *user's original screenshot-inspired* SI hybrid PR #1459. `main` untouched, other models/tracks untouched, no paid GPU/Modal/Colab, automation or merge.
- Original preregistration `docs/SI_ORIGINAL_TERMINAL_BOTTLENECK_V24_PROTOCOL_2026-10-17.md` **committed BEFORE original confirmation seed116211**, specifying fixed simulator, training and strict gates. **Original full 298-line `tam_research/si_original_hybrid_v0.py` Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`** unchanged, runtime hash-attested. Hybrid canonical backbone64,614 parameters versus Transformer64,637, GRU10,283 unmatched; task-head totals in this v21/v23 setup hybrid**64,674**, Transformer**64,697**, GRU**10,343**. Full external episodic memory, fast adapter and multimodality remain inactive. Params are not equal active FLOPs.
- **116211 consumed by wall-clock timeout, 0-byte stdout/stderr, no captured child exit, no science result**, NEVER rerun. Original 116212/116213 were NOT attempted. Documented prospectively in `docs/SI_ORIGINAL_TERMINAL_BOTTLENECK_V24_INFRA_ADDENDUM_2026-10-17.md` BEFORE more confirmation compute. New original-vs-cached purchase-scoring parity test passes; purchase caching changes performance only and preserves exactly the same Q0/Q1 actions/reward. Updated executed source `scripts/si_original_terminal_bottleneck_v24.py` **Git blob `fac6a47a1575d87ba5bfea9bc39bc8741b900139`**, SHA256 **`718af254a7673b743f5c0313e4ba5d6255b37cd279d666fd32d17c04409903a7`**; revised tests `tests/test_si_original_terminal_bottleneck_v24.py` **Git blob `21c56b6049be7d89d2757170895c5fa19364aa15`**, SHA256 **`f758ed7400814fb5ad5e9d87bf8dc33173db5b3c0874c96e9ec8f428f70cae0f`**. **9/9 targeted local tests passed**, including exact source, no purchased data leaks, factual feedback, identical cache/uncached scoring, fixed purchase and price parity, forced-probe cost accounting. Both GitHub blobs exactly match their locally executed file bytes.
- Excluded dev 116000 smoke, 116010, 116011, optimized implementation-only 116021; recorded before the new seed panel; no tuning after outcomes. In particular 116021 hybrid original policy collapsed to no-probe 100%, while final classifier performed well on forced strong; informed diagnostic but not confirmation.
- **Separate prospective engineering extension seeds exactly 116214,116215,116216** reserved durably BEFORE CPU and each launched ONCE with no retry/resume/replacement, **all child process exits0 and stderr empty**, full individual JSON ~34KB. Never call them original 116211–116213 completed confirmation. CPU torch2.10, two threads, **same original exact v23 training**: actor80+v19 critic120+v21 Q1 success80+v21 factual Q080, batch64, **4000 randomized observed second-purchase outcomes**, plus **4000 SEPARATE randomized factual terminal feedback examples** for raw and engineered-feature ridge heads. Each seed tests **256 exactly opposite-label twin PAIRS/family** =512 heldout worlds/family, across base/high_cost_ood/weak_sensors_ood/strong_discount_ood. Three independent model-training seeds only, a bounded **engineering screen not statistically powered replication**.

## What the experiment actually controls
Keep same trained original SI/Transformer/GRU Q0 initial-stop/cheap/strong and Q1 conditional second-probe decisions for all four terminal policies: frozen original neural final-action head, 7-feature raw-observation ridge, 9-feature engineered Gaussian-product ridge, and explicitly hand-specified known-Gaussian signed likelihood terminal. Purchases and actual prices are **identical per architecture, seed, family and terminal variant**, asserted in the executable for each result. Terminal ridge heads train on executed random final actions and returned correctness (recovering query answer after feedback), not hidden direct oracle training labels and not any unpurchased probe. The engineered products `mu*reading` and exact Gaussian terminal explicitly **know structural observation laws**; they are privileged inductive biases, not fair equal-knowledge novel architecture claims. Because Q0/Q1 remain trained for the old neural final actor, replacing it creates an **evaluation-only policy mismatch**; this cannot prove a fully joint alternative trainable agent would or would not work.

Under the paired opposite-answer twins, an agent receiving no probe cannot tell labels apart (exact 50% accuracy). Forced strong/cheap/both diagnostics additionally test the **same frozen final classifier with fixed purchased information**; they do NOT represent prospective changes to first/second buy decisions. Full known-model Bayes obtains optimal purchasing **and** its own terminal decision; this is a different stronger control than the same-purchases Bayes-terminal ablation.

## Means of 3 separately registered COMPLETED extension seeds

### REAL net reward after probe costs (higher better)
| Model/terminal | BASE | High-cost OOD | Weak sensor OOD | Discounted strong OOD |
|---|---:|---:|---:|---:|
| Original SI hybrid, neural final | **0.57763** | 0.52714 | 0.52752 | 0.62801 |
| Original SI hybrid, raw learned ridge final | 0.58609 | — | — | — |
| Original SI hybrid, engineered learned likelihood ridge | **0.58675** | **0.53951** | 0.52883 | 0.63191 |
| Original SI hybrid, known-law Bayes terminal ON SAME purchases | **0.58609** | 0.53951 | 0.52818 | 0.63322 |
| Matched Transformer, neural final | **0.63198** | 0.55543 | 0.56141 | 0.69311 |
| Transformer, engineered learned likelihood ridge | **0.64305** | 0.56520 | 0.56727 | 0.70027 |
| Transformer, known-law Bayes terminal ON SAME purchases | **0.64435** | 0.56389 | 0.56727 | 0.70027 |
| Small GRU, original neural final | 0.66929 | 0.57004 | 0.56613 | 0.73147 |
| Small GRU, engineered learned likelihood ridge | 0.67059 | 0.57134 | 0.56288 | 0.73538 |
| Best fixed high-only separate purchase policy | **0.71390** | — | — | — |
| Best full known two-stage Bayes (purchases AND final) | **0.75604** | — | — | — |

Every original neural method scores below the best fixed high-only control on normal base. Raw learned ridge values and all four fam/seed data remain in raw JSON/aggregated evidence; `—` is presentation-only omission, not missing execution.

### Exact base seed-level reward + no-probe behavior
| Extension seed | SI neural | SI learned likelihood | SI no-probe rate | Transformer neural | Transformer learned likelihood | Transformer no-probe |
|---|---:|---:|---:|---:|---:|---:|
| **116214** | 0.5517 | 0.5537 | **79.3%** | 0.7216 | 0.7294 | 0.4% |
| **116215** | 0.6812 | 0.7065 | **0%** | 0.6135 | 0.6349 | 35.5% |
| **116216** | **0.5000** | **0.5000** | **100%** | 0.5609 | 0.5648 | 77.7% |

The original SI hybrid has a very different purchase policy across seeds. In seed116216 its Q0 decides to buy no information in ALL test worlds, making **every terminal classifier equivalent (net0.5)** by observational symmetry. In seed116215 it buys at least one probe in all worlds and learned final classifier modestly helps. Such extreme variation with tiny n=3 is decisive evidence against a reliable active experimenter, not something to average away.

Mean BASE no-probe fractions: original hybrid **59.8%**, Transformer **37.9%**, GRU **26.0%**. Fraction choosing strong first (base) SI5.9%, Transformer5.2%, GRU11.5%, lower than a strong policy should on many worlds.

### Forced sensor (diagnostic only) final classification accuracy
| Force purchased | Original hybrid neural | Transformer neural | GRU neural | Engineered ridge | Known-Gaussian Bayes terminal |
|---|---:|---:|---:|---:|---:|
| Cheap only | **75.85%** | 76.11% | 75.78% | 77.54% | 78.06% |
| Strong only | **93.49%** | **93.62%** | **93.23%** | 94.34% | 94.27% |
| Both probes | **95.25%** | 94.60% | 95.90% | 96.48% | 96.55% |

The original neural final classifier is already highly effective **conditional on actually buying strong/both sensors**. Terminal-only improvements are <1 percentage point forced-strong and ~1–2 points forced-both. This explains the limited same-purchase reward lift despite a large gap to optimal information acquisition.

Measured mean per-seed wall for 3 trained models + their v23 calibration and v24 scoring: original SI hybrid **11.49s**, Transformer **7.66s**, GRU **2.69s**, ratio original/Transformer ≈**1.500**; do not claim better efficiency or a clean ratio pass, since this includes evaluation differences/overheads and no active FLOP/energy parity.

## Frozen preregistered A/B/C/D outcomes

**A integrity/task PASS on separate completed extension:** exact source/9 tests, same purchase schedule/price across 4 final methods each model/family/seed, preprobe symmetry50%, no unpurchased signal leak, source attestation. Known Bayesian full two-stage base0.75604 > best fixed single0.71390 by +0.04214 (threshold≥+.025).

**B TERMINAL CLASSIFIER BOTTLENECK FAIL for BOTH full backbones:**
- Hybrid engineered learned terminal base improvement **+0.00911**, below required **+0.020**, although positive 2/3 seeds (the third is a zero-gain no-probe collapse). Known Gaussian terminal same-purchase uplift **+0.00846**, below required +0.025. Conditional forced accuracy already high.
- Transformer learned base improvement **+0.01107**, below +.020 (positive 3/3), known-law terminal **+0.01237**, below +.025. Both have minor OOD benefits but no primary gate pass.
- Small GRU learned uplift only +.00130, including weak-sensor negative. There is no uniquely deficient SI final classifier and no architectural discovery.

**C ACQUISITION BOTTLENECK DIAGNOSTIC PASS for both full backbones:** Mean neural forced-strong accuracy hybrid **0.9349**, Transformer **0.9362** exceeds required .85; original hybrid BASE no-probe fraction .598 and Transformer .379 exceed .20. Same-purchase Bayes-terminal uplift SI .00846 / Transformer .01237 ≤ .015; no-probe paired-world accuracy is intrinsically .5. **This identifies a critical issue, NOT a trained fix.** A fully different purchase policy with retrained terminal actor could behave differently.

**D original SI advantage NOT DEMONSTRATED:** Original hybrid neural BASE0.5776 < Transformer0.6320 < small GRU0.6693; with engineered linear terminal SI0.5867 < Transformer0.6430 < GRU0.6706. Synthetic task and 3 independent training seeds insufficient for broad superintelligence claims. Full original persistent memory and multimodal abilities not trained/tested. No GPU, main merge or research automation authorized.

## No-go and next CPU-only research question
V24 isolates the dominant measured loss: **the model does not reliably acquire observations it can already classify well**. Next bounded falsifier should hold final classifier FIXED and learn/compare Q0 first-action **known-price contextual value** from randomized real net-return logs, including a simple *fixed high* and a cheap linear contextual policy, and evaluate whether the apparently collapsed no-probe seeds can be avoided prospectively. Critically require >fixed high and >same-budget Transformer/GRU in net reward, multiple independent training seeds and explicit training/inference FLOPs. Do not keep layering heads without proving first-probe selection competence. Archive original timed-out 116211 as consumed and the separate complete extension, require exact-head GitHub CI and independent review before any main integration.
