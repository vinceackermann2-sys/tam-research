# Original SI v7 — CPU task-validity screen, results (2026-10-16)

**Gate: PASS for benchmark usefulness; NOT A PASS for the original SI architecture.** A simple partially observed binary-mode active-sensing task yields a genuine reward benefit from choosing *when* to purchase information. A compact recurrent policy trained by Gaussian Bayes-policy imitation performs close to the correct reference in distribution but does NOT adjust sufficiently under corrupted sensors. The actual original SI hybrid and the matched Transformer were **not** trained on this task. Do not interpret these results as evidence that SI beats a Transformer.

## Exact source and execution

- Parent: source-attested original SI v6 PR #1478 (CI success); this standalone mechanism-validation branch `experiment/si-original-belief-pomdp-v7-20261016` is stacked on it. Main and other architecture branches unchanged.
- Protocol frozen before confirmation: `docs/SI_ORIGINAL_BELIEF_POMDP_V7_PROTOCOL_2026-10-16.md`.
- Executed benchmark `scripts/si_original_belief_task_v7.py` Git blob **`c5531695e7c4eaee6e3974d80f540a067dbaea4b`**; local and GitHub blobs verified identical; SHA256 `91755b40591f949643779ee9aed62f4dc55a5787e2aedf57f8b6e8a5a8e5848a`. Repo-layout test file `tests/test_si_original_belief_task_v7.py`, exact Git blob `3e4c44b3fb30b54a312c33e60fdc3ee4fffca8de`, also matches the locally executed tests. **Four unit tests passed locally**.
- Training: supervised Gaussian optimal-probe imitation labels (known synthetic posterior), synthetic hidden-rule action labels, 240 AdamW updates/model/seed, batch 256, LR=.002, WD=.01; two CPU threads. Dense **338 parameters** vs GRU recurrent **2,722 parameters**; parameter and FLOP parity NOT enforced. They are *different simple task controls*, not original SI or Transformer models. No RL, no real-world data.
- Evaluation: 8,192 seeded episodes × 3 separate completed seeds **per family**. Same held-out worlds and costs across all candidates on each seed. Sensor flip of 20% and higher-cost probe distributions are unseen during training. Reward = P(correct mode action) minus information-purchase cost, not raw accuracy.
- Engineering seed history: originally planned **99611,99612,99613**. First `99611` was interrupted by local outer-shell timeout before a complete result (0-byte file), not rerun or interpreted. A *preexecution* infra addendum `docs/SI_ORIGINAL_BELIEF_POMDP_V7_INFRA_ADDENDUM_2026-10-16.md` declared replacement **99614**. Valid completed attempts: **99612,99613,99614** (each one attempt, all success); no retries of consumed seeds. This is engineering evidence with an interrupted-seed replacement, **not strict unamended scientific replication**.
- Two-token recurrent input has a hard causal information mask; probe observation is unavailable before probe decision, even when probe data were generated for possible counterfactuals. Analytic Bayesian decision calculates value of information for a *specified known generative model*; the 20%-corruption reference is given the correct corruption rate, an advantage not available to the trained neural models.

## Three-seed mean held-out realized reward, higher better

| Controller | In-distribution Gaussian | OOD: 20% initial-cue sign flip | OOD: higher probe cost |
|---|---:|---:|---:|
| Bayesian posterior and optimal information purchase, correct model | **0.841957** | **0.768108** | **0.816078** |
| GRU learned via Bayes imitation / hidden mode targets | 0.834600 | 0.747081 | 0.802434 |
| Dense learned via Bayes imitation / hidden mode targets | 0.826578 | 0.739548 | 0.794366 |
| Gaussian Bayes (misspecified under sign flip) | 0.841957 | 0.754544 | 0.816078 |
| Never purchase probe | 0.769775 | 0.661255 | 0.769775 |
| Always purchase probe | 0.769295 | 0.745003 | 0.678997 |
| Random final action, never probe | 0.504801 | 0.504801 | 0.504801 |

In-distribution per-seed rewards: exact Bayesian [0.83881,0.84061,0.84646]; recurrent [0.83056,0.83576,0.83749]; dense [0.82435,0.82510,0.83029]; never [0.76184,0.77637,0.77112]; always [0.76866,0.76600,0.77323].

OOD 20% sign-flip per seed: correctly informed Bayesian [0.77011,0.76415,0.77006]; misspecified Gaussian Bayesian [0.75676,0.74815,0.75873]; recurrent [0.74486,0.74269,0.75369]; dense [0.73965,0.73459,0.74441].

Higher cost per seed correct Bayesian [0.80693,0.81859,0.82271]; recurrent [0.80023,0.80068,0.80639]; dense [0.79119,0.79448,0.79743].

## Information-acquisition behavior

| Controller | In-distribution probe rate | 20%-corrupted initial sensor | Higher-cost sensor |
|---|---:|---:|---:|
| Correct Bayesian | **0.46358** | **0.72392** | **0.31132** |
| Recurrent controller | 0.46139 | 0.46330 | 0.40942 |
| Dense controller | 0.46460 | 0.46545 | 0.40934 |
| Never / always | 0 / 1 | 0 / 1 | 0 / 1 |

The exact Bayesian controller beats never-probe by **+0.072182** and always-probe by **+0.072662** in normal Gaussian worlds, in each seed and with an interior probe fraction. **All predeclared task-validity conditions passed.** Under sign-flipped cues, knowledge of the correct observation model improves the Bayesian controller by **+0.013564** reward over misspecified Gaussian Bayes. The trained recurrent controller **does not increase probe rate** in the corrupted regime (~46.3%), whereas the informed Bayes controller does (~72.4%). Higher probe costs elicit a lower purchase rate in all models, but neural controls still buy probes too frequently.

In-distribution stage2 correctness: recurrent 0.91191, correct Bayesian 0.91268. Mean training CPU seconds: recurrent ~2.33 s per seed, dense ~1.21 s, excluding model import/runner overhead. This confirms the benchmark can teach compact models simple active sensing but does **not** demonstrate general belief updating, environment discovery or strong OOD transfer.

## Decision and next architecture gate

**Advance the benchmark, not the original architecture.** It has a strong Bayesian reference and meaningful counterfactual information-purchase outcome. The recurrent controller's small normal-case gap is expected for **supervised imitation of an analytic solver**, not an architectural discovery. The 20% corrupted-cue OOD failure exposes a falsifiable hypothesis: a learnable belief/hypothesis *uncertainty over the observation model itself* should adapt probe frequency when recent evidence contradicts assumptions.

Next compare **fully source-verified OriginalSIHybrid** vs **TransformerControl** on this same task, with the **same input representation and external inference/memory mechanisms** and Bayesian, MLP/GRU controls. More meaningful later tasks need **unknown continuous dynamics, stochastic observations, delayed effects, multiple stages, and actual model identification**. No GPU scale, no 250M/5B, no breakthrough or paid authorization. Branch remains draft until CI and review.
