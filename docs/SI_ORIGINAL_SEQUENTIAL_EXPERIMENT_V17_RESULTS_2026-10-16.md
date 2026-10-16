# Original SI v17 — Two-stage active sensing, frozen three-seed CPU results (2026-10-16)

**Decision: task validity PASS, learned conditional probe imitation PASS, original SI architectural advantage FAIL.** No superintelligence breakthrough, GPU/Modal/Colab credits, main write or merge, automation, or scaling authority. The full report and raw evidence are separately archived in the user-facing v17 CPU ZIP; this is the repository's reviewable negative/positive result record.

## Exact frozen source and attempt integrity

Frozen protocol docs/SI_ORIGINAL_SEQUENTIAL_EXPERIMENT_V17_PROTOCOL_2026-10-16.md **COMMITTED before all confirmation seeds**. Full frozen local protocol SHA256 bb2d2d67e5f0c8a99c699f6557f257fac6b0a923bf9c44524486728d15c039d5. Complete original 298-line hybrid model Git blob 575c17dc2f39a29d930d82179aa0c96dcbdc7d0f. Exact executed/GitHub script scripts/si_original_active_experiment_v17.py **Git blob aa60524483117237d9929a0e44da87a3d10dd8c5**, SHA256 f70db56c096fd9ee7cb68f67eabdd01d6bd675173319b797260ac14a013ca4c1; tests/test_si_original_active_experiment_v17.py **Git blob e0cc960fb7298c126711aa7473d034dcbd6f156d**, SHA256 2d69d0a260fc68b77deb81a351c212a5a642db16c6841ef46cb8284d0d493b33. Both files byte-identical to local execution/test. Initial source upload differed only by location of a nonfunctional comment; reconciled to exact attested blob without changing any executed behavior or frozen result. **Seven targeted pytest tests pass** locally (pre/post confirmation).

Excluded development 109000,109001–109006,109010,109011,109019. Fresh locked **109211,109212,109213** each launched exactly once under durable attempt marker; exit code 0, empty stderr and complete JSON, no seed retries/substitution. Train **240 AdamW steps × batch96** on base worlds per model/seed (2 CPU threads, LR .001, wd .01). 3 matched training architectures: original full hybrid 64,614 params, TransformerControl 64,637, smaller unmatched GRU 10,283. Evaluation each seed: **1,024 exactly observationally indistinguishable twin PAIRS** (2,048 opposite-label episodes) in each base/high-cost/weak-sensor family, paired inputs across all methods. n=3 independent model seeds; strictly low-powered synthetic engineering evidence.

Important training qualifier: **original/Transformer/GRU models explicitly imitate purchase decisions made by an analytic known-simulator Bayesian dynamic-programming teacher**. Terminal labels can be reconstructed from executed random action and correctness feedback. No hidden truth or unpurchased probe value is a neural inference input, but training purchase labels ARE privileged Bayesian demonstrations. Not autonomous experiment-discovery, learned causal world laws, generalized self-improvement or model-free RL. Full original SI external memory, fast continual adaptation, multimodality, tools and real long-range planning remain undemonstrated.

## Same-world 3-seed mean net rewards, higher better

| Control or policy | Base | Higher-cost OOD | Weaker-sensor OOD | Base second buy rate | Base high-first |
|---|---:|---:|---:|---:|---:|
| Known-model **two-stage Bayes** | **0.749662** | **0.636542** | **0.629204** | 0.178 | 0.482 |
| Greedy first probe then optimal second | 0.744826 | 0.636597 | 0.627137 | 0.126 | 0.643 |
| **Original SI hybrid** | **0.730411** | 0.601901 | 0.610246 | 0.217 | 0.560 |
| **Matched Transformer** | **0.732836** | 0.613128 | 0.606135 | 0.242 | 0.387 |
| Smaller GRU | 0.720455 | 0.609615 | 0.592510 | 0.189 | 0.356 |
| Fixed cheaper probe only | 0.676697 | 0.613397 | 0.581570 | 0 | 0 |
| Fixed stronger probe only | 0.712424 | 0.566976 | 0.592184 | 0 | 1 |
| Always both (cheap then strong) | 0.643841 | 0.437046 | 0.535082 | 1 | 0 |
| Never buy new information | 0.500000 | 0.500000 | 0.500000 | 0 | 0 |

**Base per-seed reward in frozen seed order [109211,109212,109213]:**
- Bayes [0.758739,0.748092,0.742156], greedy [0.751571,0.743724,0.739182].
- Hybrid [0.740357,0.730300,0.720576].
- Transformer [0.742883,0.727042,0.728584].
- Hybrid−Transformer [−0.002526,+0.003258,−0.008008], mean **−0.002425**, wins **1/3** seeds.
- Mean CPU train wall: hybrid **10.953s**, Transformer **9.148s**, GRU **~2.69s**; hybrid 1.20x Transformer. No active-FLOP parity demonstrated.

## Locked gates — no posthoc relaxation

**A sequential task validity PASS:** Bayes normal base net 0.749662 beats no probe by +0.249662, fixed low by +0.072965, fixed high by +0.037238, always-both by +0.105821 (all exceed prereg thresholds); second purchase ~17.8% (5%–60% interior), initial cheap/strong each ~51.8/48.2%. Bayes first-stage 24-vs48 quadrature choice disagreed ~1.05% in excluded development integrity test (max3%). First-stage regret versus greedy is small: **+0.004836** base on three seeds, while greedy was actually slightly ahead at high-cost OOD (**0.636597 vs 0.636542**). Do not overstate Bayesian gain over greedy; numerical quadrature and finite sampled test worlds make small differences uncertain.

**B learned conditional sequential acquisition PASS in base family:** Both neural full backbones choose cheap AND strong first in >=10% of worlds in each seed, second purchase between 5% and65% in every seed. Hybrid buys second [0.273,0.219,0.159], Transformer [0.272,0.284,0.169]. Hybrid and Transformer base net rewards are within **0.019251** and **0.016826** of privileged Bayes (<.045 allowed), respectively, and exceed better fixed single-probe method 0.712424 by **+0.017987** and **+0.020412** (>=.015 threshold). This is supervised teacher distillation success, not independent optimal experimental design.

**C unique original hybrid advantage FAIL:** Hybrid base score 0.730411 is **below Transformer 0.732836**, fails required +0.015 and >=2/3 seed wins, despite meeting CPU ratio <1.5 and OOD loss tolerance (hybrid under high cost **−0.011227** vs Transformer; with weaker sensors **+0.004111**). Weak-sensor single-family win cannot substitute for paired normal gate. Hybrid costs more CPU time and has no demonstrated hardware advantage.

**D identifiability integrity PASS:** Twin preprobe calibration/query/quality/price exactly identical with opposite ground truth; no-probe deterministic classifiers cannot beat paired 50%; stage0 tokens contain neither unpurchased probe, stage1 excludes remaining unpurchased probe. Seven tests cover private-label/probe mutation, phase token causal behavior, posterior math, second-stage decision and finite network gradients.

## Decision and next bounded research
Next CPU-only question: remove teacher-provided Bayesian purchase labels and train **sequential, conditional value estimates from logged actual chosen actions/outcomes**, with sufficient randomized exploration and conditional cost awareness, against a true two-stage Bayes/greedy policy and matched Transformer/GRU. Make the causal information boundaries and matched world seeds strict, include out-of-distribution sensor/cost laws and genuine regret/cost curves. No paid GPU, automated research loop, main merge or SI breakthrough based on this small known-family supervised test. Require CI on exact PR head and independent scientific review.
