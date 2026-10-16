# Original SI v11 — paired v9 vs v10 training and hidden sensor-polarity drift (2026-10-16)

**Verdict: V10 contextual-value method PASSES the preregistered paired base-task method gate; original SI hybrid relative advantage FAILS; unseen drift robustness FAILS. No SI breakthrough, GPU authorization, or claim of general online world inference.** Report all comparisons, failure attempts and constraints without promoting favorable endpoints alone.

## Provenance, preregistration and consumed attempts

- Isolated `experiment/si-original-paired-drift-v11-20261016`, based on v10 draft PR #1482 and ultimately original screenshot-inspired full-hybrid PR #1459; main and all other research tracks untouched.
- Frozen initial `docs/SI_ORIGINAL_PAIRED_DRIFT_V11_PROTOCOL_2026-10-16.md`, written **before** CPU confirmation. Exact complete original 298-line `tam_research/si_original_hybrid_v0.py` Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`. Imported **unchanged** v9 trainer blob `9ff12b978ebc4f6554fa8443c13f1c545d4df6ca`, unchanged v10 trainer blob `c8ade2177f495e0448638cc80d98dc6adde4fb51`, original environment blob `233dff4d660da142fbb57a91cd06b0c4ec17a0d0`; all local copies Git-blob-verified to repository.
- First new benchmark file `scripts/si_original_paired_drift_v11.py` blob `c24c217bee994e5f98153f23bc6f54aee340dda1` underwent one **isolated numerical control correction BEFORE successful confirmation**. Corrected final GitHub/local byte-identical benchmark blob **`1eea8c3ce72952388b9c79b8d299c23b75ca3351`**, SHA256 `633f2f3462e3b8f663995cf11efc0f957ca351e0a0cc4114b5cc20b6fcb0bbd8`, six tests blob **`9cd8071002bd8551ba33f95f6337e892c50c7987`**. **All 6 tests pass locally**, plus 3x4,096-world corrected Bayesian development-only stress checks.
- **Exact attempts:** Original seed `103211` failed immediately on a wrong `python scripts/...` import path, producing 0-byte JSON and `ModuleNotFoundError: No module named 'scripts'`. Second `103212` got past imports but failed **before model training** while evaluating an always-probe Bayesian control: for very confident float32 posterior, VOI rounds to 0 and a cost=0 surrogate did not force a probe. JSON was again 0 bytes. **Both seeds consumed; neither retried; neither supplied performance evidence.** Full error messages preserved. See `docs/SI_ORIGINAL_PAIRED_DRIFT_V11_INFRA_ADDENDUM_2026-10-16.md` and `docs/SI_ORIGINAL_PAIRED_DRIFT_V11_NUMERICAL_FIX_ADDENDUM_2026-10-16.md`. Corrected Bayes forced-probe method passes explicit stress test and changes **no trained model, training sample, reward, threshold, latent information channel, or Bayesian selective method**.
- Before replacement starts, addenda registered `103214` (for `103211`) and `103215` (for `103212`). **Complete valid engineering seeds: `103213` (original), `103214` and `103215` (replacements)**, each launched **once**, full JSON and process exit0, zero stderr; no retries. This is a corrected, transparently substituted **CPU engineering screen, not unamended independent scientific replication**.
- CPU-only PyTorch **2.10.0+cpu**, 2 threads. Per seed, **9 models** (3 original backbones ×3 frozen training regimes). Training worlds stationary base only and independently seeded evaluation worlds: 3,072 per environment/family/seed, identical paired evaluations across nine models and references (9,216 episodes/family pooled but only 3 model-training seeds). No paid GPU, Modal/Colab, automation or main merge.
- Existing **original SI hybrid 64,614 params** and **TransformerControl 64,637 params**, smaller unmatched GRU **10,283 params**. Original hybrid retains the canonical local attention + gated recurrence + sparse MoE + refiner; persistent episodic memory store and fast adapter are instantiated/dormant, NOT functioning general lifelong memory. No equal measured FLOPs across methods. Comparisons are toy binary sensor actions, not real multi-step agent capability.

## Protocol and simulator

- `v9_original`: original scalar REINFORCE + observed-correlation learning, **240 AdamW updates**, batch128.
- `v9_500steps`: **same unchanged v9 algorithm**, 500 AdamW updates, batch128; matches V10 in *total update count* only.
- `v10_contextual`: **unchanged** two-stage method, 240 actor updates then 260 logged-bandit contextual value-output updates, batch128. The critic regresses only the **randomly observed** buy/skip arm's actual binary correctness. No hidden truth/polarity labels, unpurchased future sensor, model-specific oracle labels, or hidden change-point enter training. V9 vs V10 initialization seeds and early training streams match, while V10's critic stream intentionally differs (fresh observations); therefore matched-seed evaluation is rigorous, not perfectly equal-compute/data training.
- Base family `base`: stationary sensor polarity from original v8. Holdout `weak_calibration`: weaker evidence as in v8. New previously unseen `polarity_flip` holds true query polarity at z, but flips noisy target calibration polarity at private change index c∈{2,3,4}, sampled independently each episode. Agent only sees six paired sensor signals, known strengths and costs; never receives private c/z, query truth or unpurchased probe.
- Drift Bayes known-family reference **sums likelihoods over all three possible change times**, never receives actual c, then optimally buys information by expected gain; it *does know* that the family undergoes one switch, making it an **advantaged diagnostic reference**. A mis-specified stationary Bayes policy plus forced always/never probe Bayes controls use the same inputs but differing assumptions. Models trained **only on stationary worlds**, so drift score is out-of-distribution testing, NOT online adaptation learned by training.

## Three-seed averaged realized net reward (higher is better)

| Architecture and method | Normal/base | Hidden polarity flip | Weaker calibration |
|---|---:|---:|---:|
| Original hybrid — V9 240 steps | 0.776753 | 0.758918 | 0.761414 |
| Original hybrid — V9 500 steps | 0.781419 | 0.754577 | 0.763042 |
| **Original hybrid — V10 contextual** | **0.801272** | **0.741365** | **0.752878** |
| Transformer — V9 240 steps | 0.780117 | 0.734286 | 0.760654 |
| Transformer — V9 500 steps | 0.792596 | 0.733423 | 0.767127 |
| **Transformer — V10 contextual** | **0.813579** | **0.682569** | **0.761200** |
| GRU — V9 240 steps | 0.770785 | 0.752516 | 0.763910 |
| GRU — V9 500 steps | 0.785716 | 0.730489 | 0.767299 |
| GRU — V10 contextual | 0.766714 | 0.754651 | 0.760782 |
| Correct Bayesian (knows respective family law) | **0.853714** | **0.846131** | **0.802454** |
| Bayesian always buy (knows family law) | 0.791293 | **0.782572** | 0.773350 |
| Bayesian never buy (knows family law) | 0.777018 | **0.767036** | 0.667318 |
| Misspecified stationary Bayes | Same as normal correct | **0.667369** | Same as weak correct |

**Base mean probe fractions:** original hybrid V9-240 **1.000**, V9-500 **1.000**, V10 **0.67556**; Transformer V9-240 **1.000**, V9-500 **0.93587**, V10 **0.64985**; GRU V10 **0.93327**. Full hybrid V10 per-seed purchases [0.75488,0.73014,0.54167], Transformer [0.76270,0.63737,0.54948]; all interior 0.1–0.9. **Normal Q-arm Brier**: hybrid V10 **0.12092**, Transformer V10 **0.11778** (both pass <=.16); GRU V10 0.15269.

**Hidden polarity flip:** V10 hybrid probe fraction **0.68110**, randomized-arm Brier **0.16323**; V10 Transformer probe fraction **0.66461**, Brier **0.19925**; GRU probe fraction **0.94195**, Brier **0.16172**. Both full models' decision-value calibration degrades under previously unseen drift. Because some decisions do not buy, a critic may systematically over/under-estimate counterfactual success without access to drift inference; a tiny calibration loss is not evidence of calibrated uncertainty in real tasks.

## Primary paired gains and explicit gate outcomes

**V10 vs V9-500 on normal reward (paired per seed, in prereg seed order 103213/103214/103215)**:
- Hybrid V10 - V9-500 = **[+0.010753,+0.023658,+0.025145]**, mean **+0.019852**, wins **3/3**.
- Transformer V10 - V9-500 = **[+0.012856,+0.026375,+0.023719]**, mean **+0.020983**, wins **3/3**.
- V10 vs original V9-240 mean improvements: hybrid **+0.024518**, Transformer **+0.033462**, both 3/3 seed wins.
- V10 normal reward **0.801272 hybrid vs 0.813579 Transformer**, delta hybrid - Transformer **-0.012308**, and Transformer wins 3/3 seed comparisons.

**Predeclared contextual method gate on familiar worlds: PASS as narrow CPU engineering evidence.** Both full neural architectures outperform original v9 and 500-step v9 by >=+0.015, win >=2/3 paired seeds, use nondegenerate measurement-purchase rates and achieve base Brier <=.16. This is a valid relative within-task method improvement. It does **not** pass the previous stricter v10 absolute Bayesian reward-gap gate, does NOT imply general transfer and is not GPU authorization.

**Predeclared original SI backbone superiority gate: FAIL.** Original hybrid V10 is worse than matched Transformer in every base seed and both OOD families, despite matched nominal parameter counts. Hybrid/Transformer training wall: V10 hybrid **8.318s**, Transformer **6.414s** (~1.30× slower); no demonstrated active-compute savings. Mean CPU wall V9-500 hybrid **18.557s**, Transformer **15.299s** (500 full-model steps vs 240 full-model +260 output-only updates; update count parity is NOT true FLOP parity).

**Predeclared drift robustness gate: FAIL badly.** V10 hybrid drops **0.059907** absolute from 0.801272 normal reward to 0.741365 drift (max allowed 0.05). Its drift reward is **0.041207 below** corrected Bayesian always-buy 0.782572 and **0.025671 below** corrected never-buy 0.767036, and **0.104766 below** correct change-aware Bayes 0.846131. V10 Transformer drops **0.131010** from base to drift, even more sharply. On drift, hybrid V9-500 **0.754577** beats hybrid V10 **0.741365**; Transformer V9-500 **0.733423** beats Transformer V10 **0.682569**. A learned value-function improvement can harm robustness, even when it reduces average costly probe decisions in distribution.

**Method-specific GRU negative control:** smaller GRU V10 does not beat its own V9-500 normal reward (0.766714 vs 0.785716) and near-always purchases information. Thus v10 is NOT a universally superior training method.

## Scientific interpretation / next step

This resolves the previous seed/budget confound more carefully than v9–v10 historical comparisons: with identical paired evaluation worlds and same **total optimizer update counts** against V9-500, contextual logged-arm valuation gives clear positive gains on the training-family task for both 65k-param neural backbones. Yet it **does not robustly adapt** when the target sensor's hidden observation law changes during a calibration episode; the critic's value estimates become unreliable, resulting in poor action/outcome reward relative to even always/never-buy Bayesian controls.

Do not declare the original hybrid architecture superior or make a scaling claim. A useful **new CPU-only hypothesis** would learn a causal latent change-point/sensor-reliability posterior from calibration observations, propagate uncertainty into the **conditional value of information** (possibly conservative expected regret under model uncertainty), and evaluate on *multiple unseen drift distributions* versus exact Bayesian, particle-filter, Transformer/GRU and always/never-buy controls. Because the known-model Bayesian reference knows the shift family, further controls should deliberately withhold the shift prior from learned/planning baselines and assess genuine online discovery rather than copying a known simulator. Require more independent model seeds and rigorous compute parity before paid H100/GPU promotion. 

**Zero paid GPU/Modal/Colab credits, zero research task automation, zero merge to main.** Keep draft branch, verify exact-head GitHub CI, preserve all interrupted attempts and raw seed outputs. No SI breakthrough confirmed.
