# Original SI v11 — paired v9/v10 replication and hidden sensor polarity drift (CPU preregistration)

**Frozen 2026-10-16 BEFORE any confirmation-seed execution.** Branch `experiment/si-original-paired-drift-v11-20261016` stacked on v10 draft PR #1482 -> original screenshot-inspired SI hybrid PR #1459. Distinct from EHW, PGW, CPW, CHM, RLT and all other tracks. Do not merge, modify `main`, spend paid Modal/Colab/GPU credits, or schedule automations.

## Motivation and scope

V9 reward-policy learning collapsed to almost-always buying information; v10 learned two separate contextual expected-correctness scores from randomized *observed* buy/skip outcomes, reducing the collapse in the full hybrid and matched Transformer but not passing original architecture advancement. **Prior v9 and v10 used different scientific seeds, evaluation worlds and optimization budgets**, so prior cross-version differences were not paired causal evidence.

**v11 resolves two concrete uncertainties** without inventing a new large architecture: 
1. Re-execute the exact **unaltered v9 and v10 trainers** with fresh **identical training initialization seeds**, identical *base-world training sampler* and the *same heldout episode IDs* in the same turn. Include an additional **v9-500-updates** control to account for v10's 240 actor + 260 critic updates. These are matched *update counts*, **NOT equal active FLOPs or even equal training observations**, since v10 critic uses a fresh logged bandit dataset and v9 updates all backbone weights throughout. V9-240 is the literal historical method and v9-500 is a newly designated longer-run control, not a retroactive v9 replicate.
2. Test whether both policies fail if the noisy target sensor's **hidden polarity reverses partway through the six calibration readings**, and benchmark against **Bayes that knows the shift family but NOT the realized shift time**, and a misspecified Bayesian controller that assumes no change.

This is STILL a small binary latent-sensor contextual active-measurement benchmark, not multimodality, general intelligence or true persistent agent memory.

## Data / causal integrity

Exact original `tam_research/si_original_hybrid_v0.py` (298 lines) Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` locally verified. Existing **unchanged** frozen trainers: v9 blob `9ff12b978ebc4f6554fa8443c13f1c545d4df6ca`, v10 blob `c8ade2177f495e0448638cc80d98dc6adde4fb51`; v8 environment blob `233dff4d660da142fbb57a91cd06b0c4ec17a0d0`. v11 source `scripts/si_original_paired_drift_v11.py` Git blob **`c24c217bee994e5f98153f23bc6f54aee340dda1`** / SHA256 **`637502b1c9251b2e8ff0dc9e60790486afdf8512047a0336afe4a76f2e09403c`**; test `tests/test_si_original_paired_drift_v11.py` blob **`1dccc7708c657dd4d5f308afa66dc17a67cb4012`**. Five local tests passed (source hash, deterministic hidden switch world, hidden truth/future/transition private metadata isolation, exact Bayesian marginal drift model, and gradient smoke). A float32-boundary unit assertion was corrected **before freezing this protocol** (sigmoid may round to exactly 1), no model/evaluation code change and no confirmation attempted before correction.

Evaluation families:
- **base**: originally frozen Gaussian stationary sensor from v8, no drift, same training-family dynamics but previously unseen episodes.
- **polarity_flip (unseen)**: each episode has private random hidden change time c uniform among calibration indices {2,3,4}; noisy target sensor reads **opposite polarity** before c and **current polarity** thereafter and for the new query. All six reference-sensor readings unchanged. The true hidden switch c is never fed to learned policies, and the Bayesian "known drift family" reference **marginalizes all three possible c** rather than receiving the realized switch. Also score a **misspecified stationary Bayesian** reference, and known-family Bayes with forced always/never-buy.
- **weak_calibration (unseen)**: frozen v8 weak-calibration signal, a separate sensor-evidence robustness control.

Agents use six paired target/reference calibration readings, new query cue, optional purchased probe if policy chooses to pay, and subsequent correctness feedback. Hidden z, switch time c, true query theta and unpurchased probe are evaluator-private. Both models trained on **stationary base only**, no drift data. Reward = fraction correct final action minus the actual per-episode probe price. The exact Bayesian reference knows the family likelihood, an unfair **diagnostic upper bound**, not an equal-knowledge competitor. Historic v9 scalar-baseline REINFORCE uses observed reward and next-observable correlation supervision; v10 logged contextual critic uses random buy/skip and outcome correctness; neither is given hidden z, theta, analytic Bayes probe labels or private switch time as optimizer targets.

## Frozen CPU settings, seeds, and statistical honesty

**Development-only** smoke seed **103000**, 2 v9 steps, 2 longer-v9 steps, 2 actor + 2 critic steps, batch 8, n_eval 64. Evaluator diagnostic seed 103002 for unit testing exact Bayes and seed 103003 for gradient smoke. These were not used for performance selection; no tuning after seeing confirmation.

**Fresh previously unused preregistered CPU engineering confirmation seeds: 103211, 103212, 103213**, each attempted ONCE; no retries or rerun of consumed attempts. Per seed, train all `hybrid`, `transformer`, and smaller `gru` with **three fixed methods**:
- `v9_original`: 240 AdamW updates, batch128
- `v9_500steps`: 500 AdamW updates, batch128, same frozen v9 algorithm
- `v10_contextual`: 240 actor AdamW +260 observed-arm critic updates, batch128
All pre-existing functions imported unchanged, with their historical optimizer learning rates and data sources preserved. CPU torch 2.10, two threads, no external compute or extra training. Same initialization seed and same **first 240 training episodes** across v9 variants and v10 actor. All three methods share exactly **3,072 previously unseen evaluation worlds per family/seed** at seed*1,000,000+88,000+127*family index. Nine models per seed. Record per-seed/group rewards, probe fractions, accuracy, random logged-arm Brier for v10, classical reference scores, actual training wall time, parameter counts and provenance. Parameter-parity only between full hybrid 64,614 and Transformer 64,637; GRU smaller (10,283) and train FLOPs unequal.

### Preregistered decision rules

**Contextual-value method validation (CPU follow-up only):** V10 must exceed BOTH v9-original AND v9-500 **by >= +0.015 mean reward in base on both hybrid AND Transformer**, with >=2/3 paired seed wins per architecture; purchase fractions interior [0.10,0.90] for the full models on **every** seed, while older v9 versions do not attain comparable net reward; no v10 value Brier >0.16. Also disclose v10 training wall time and gains after 500-step control: do not claim compute efficiency from mismatched optimization.

**Original SI hybrid advantage validation (separate gate):** v10 original hybrid must beat *matched v10 Transformer* by >=0.015 mean base reward and in >=2/3 seeds, with no worse mean reward on both OOD families and CPU total training time <=1.5× Transformer. A v9-v10 improvement shared by both architectures is not hybrid-specific.

**Drift robustness gate:** v10 hybrid's mean flip-world reward must be within .05 of its mean base reward AND be >=0.015 above both corrected Bayes always- and never-buy controls, with normal/drift interior purchase rates, plus <=0.035 gap to correctly specified change-aware Bayesian optimal reference. Classical corrected Bayes is expected to be an upper limit; if unattainable explicitly mark fail.

If any gate fails, STOP the unchanged architecture/method as a paid-GPU candidate. No threshold changes based on confirmation, no selective seed/simulator result promotion, no "breakthrough" even if a gate passes. This is low-power engineering data, not statistically decisive replication. Persist complete individual JSONs and CI outcome; no main writes, no paid GPU authorization.
