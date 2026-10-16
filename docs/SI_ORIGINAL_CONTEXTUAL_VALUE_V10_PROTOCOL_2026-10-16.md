# Original SI v10 — contextual, observed-outcome value-of-information gate (2026-10-16)

**Frozen before fresh confirmation attempts.** Isolated branch `experiment/si-original-contextual-value-v10-20261016`, stacked on failed v9 PR #1481 → original full hybrid PR #1459. No main merge, edits to other research branches, automation, paid Modal/Colab/GPU attempts, scaling, or breakthrough authority.

## Motivation and falsifiable hypothesis

v9 trained the original SI hybrid (64,614 params), matched Transformer (64,637), GRU (10,283) on a toy episodic hidden-polarity sensor game **without hidden z or analytic Bayes-purchase labels**. Sensor prediction improved over zero baseline, but reward-trained scalar-baseline REINFORCE bought almost every probe (hybrid 99.25%; GRU 99.86%; Transformer unstable 0%/100%/99.7%) and scored below an always-buy policy and optimal known-model Bayesian controller.

v10 tests whether **learning separate contextual expected action-success values** for “buy” and “skip” from *randomly purchased, actually observed outcomes* fixes degenerate probing and produces reliably better cost-aware decisions. This is a specific value-estimation mechanism, NOT a new full SI architecture or a claim of general/continual intelligence.

## Frozen exact code / source

Full canonical 298-line `tam_research/si_original_hybrid_v0.py` Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` verified against GitHub and local v9 evidence. v8 environment module blob `233dff4d660da142fbb57a91cd06b0c4ec17a0d0`; unchanged v9 is the parent. **New executed file** `scripts/si_original_contextual_value_v10.py`, Git blob **`c8ade2177f495e0448638cc80d98dc6adde4fb51`**, SHA256 **`f06aebb4519abab3b074beff6c42871c60a063110f8720aa188148d1586dd5bb`**. Exact tests `tests/test_si_original_contextual_value_v10.py` Git blob **`dc6f5048950af06ba045f0875076b829414fd56e`**. Five local pytest tests passed before confirmation: original module identity, no inaccessible probe/latent truth inputs, action+correctness target construction, zero critic gradient on unobserved arm, finite reproducible model/feedback.

The v8 generator supplies **6 paired sensor calibration observations** from a hidden episode-specific sensor polarity, then one independent binary query with an optional paid trusted probe. World family `base` used in training only; unseen `inverted_prior`, `noisy_reference`, `weak_calibration` tested separately. Reward = final binary action correctness minus known probe cost. All simulated labels z/theta and counterfactual y_probe are **environment-private** and unavailable to the policy.

### Two-stage policy and logged-bandit signal

Three models use identical protocol, batches, and RNG: **OriginalSIHybrid (64,614 nominal params)**, **TransformerControl (64,637)**, and smaller GRU (10,283; not parameter matched). All receive identical 7D causal-token representation. Full original hybrid uses unmodified attention/recurrence/MoE/refinement; dormant external episodic memory is NOT activated.

1. **Action predictor stage:** acquire probe independently with Bernoulli(.5), execute randomly selected binary final action, observe whether that action was correct. The *executed action + binary correctness feedback* deterministically identifies which binary action is correct; train the final action head with log loss on this observed outcome. **No hidden z/theta passed directly as labels to the optimizer.** This is outcome-supervised training, not completely unsupervised latent inference or real reinforcement learning.
2. **Contextual value stage:** freeze predictor's representation / action head, use fresh independent episodes and randomized Bernoulli(.5) purchased arms, run the frozen action policy with only the actually observed probe result (if purchased), and observe correctness. For each instance regress **ONLY the value head for the purchased arm** to that observed correctness by binary cross-entropy. Do NOT synthesize or train a counterfactual unpurchased arm's target; don't pass simulator future/latent states into the optimizer. The base world environment necessarily generates possible counterfactual sensor draws internally, but pre-acquisition policy input is masked.
3. **Deployment:** from causal context predict `Q_buy=P(correct | buy, history)` and `Q_skip=P(correct | skip, history)`. Buy only if `Q_buy - Q_skip > observed_probe_cost`. Select final action using the frozen learned classifier on actual purchased / nonpurchased observations. Value heads share a *frozen* representation trained from observed action outcomes; no analytical Bayesian teacher appears in loss.

### Exact CPU budget, seeds, conditions

- Development **excluded from confirmation:** smoke seed 102000 (4 actor + 4 value updates, batch8), dev 102010 (150+150 updates, batch128, 2048 eval) and dev 102011 (240+260 updates, batch128, 2048 eval). Observed: v10 hybrid reward 0.7804 / 0.8098 with probe fractions 0.614 / 0.685; Transformer reward 0.7923 / 0.8191, fractions 0.540 / 0.644; GRU still collapsed to always-buy. **No hyperparameter, dataset, or gate tuning after these runs**.
- **Fresh, unused paired CPU engineering seeds: 102211, 102212, 102213**, each one attempt only, no retries/reuse of consumed attempts.
- Each model/seed: **240 actor AdamW steps** and **260 contextual value steps**, batch128, two CPU threads. Actor LR .001, WD .01; critic head-only LR .002, WD 0; grad clip1. Deterministic per-step episode seeds derived from seed*1,000,000+step or +350000+step; purchase/executed-action RNGs separately seeded; **same sample streams across architectures**.
- **4,096 independent held-out worlds/family/seed** at seed*1,000,000+88,000+127*family_index; same episodes and costs for all controls. No use of evaluation outcomes to tune parameters.
- Report paired per-seed and three-seed mean **net reward (primary)**, purchase fraction, action accuracy, `Qbuy-Qskip` conditional value distribution and heldout randomized-logging Brier score (evaluate on a separate single sampled arm at 0.5 probability), parameter count and actual CPU time. Report all three OOD families and base no-calibration evaluation ablation (distribution changing). No active-FLOP parity or high-powered scientific statistics implied.
- Compare **known-model Bayesian value-of-information reference** (upper comparator, knows simulator likelihood) and strong **Bayes all-history always-buy, never-buy, no-calibration** controls. Their posterior access is advantaged vs learned networks; cannot call cross-method parameter/fairness parity.

## Fixed go/no-go thresholds (further CPU research only)

Advance **contextual value mechanism as useful** if either original hybrid OR Transformer normal mean reward beats both the **always-buy** and **never-buy** strong Bayesian controls by >=0.015, buys on 0.1–0.9 of episodes in **every** seed, and has a clear improvement against previous v9 always-buy-collapse behavior. Require heldout Q-arm Brier <=0.16 and mean normal reward no worse than Bayesian reference by 0.035. Even if passes, it is only a toy contextual-bandit signal, not SI.

Advance **original SI architecture as relatively advantageous** only if original hybrid beats same-budget Transformer mean net reward by >=0.015, wins >=2/3 paired seeds, does not lose any OOD family and <=1.5× training CPU time. This is a separate gate. **If either fails, explicitly record failure**; do not manufacture a v10 breakthrough from GRU failure, baseline weakness, or one favorable seed.

If no method passes, stop this frozen v10 mechanism and design stronger separately preregistered off-policy evaluation, calibrated uncertainty, / multi-stage POMDP world experiment on CPU. No paid GPU/Modal/Colab, science scaling, automation or main merge is authorized even on a win. Preserve negative results and infra interruptions; compare original exact-head GitHub CI before considering review.
