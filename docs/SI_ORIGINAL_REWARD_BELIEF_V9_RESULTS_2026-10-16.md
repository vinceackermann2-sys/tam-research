# SI original v9 — label-free latent sensor, observation-prediction + reward training: CPU results (2026-10-16)

**Verdict: FAIL predeclared original SI hybrid advancement gate.** The hybrid slightly exceeds the matched Transformer on mean realized reward because the latter suffers one no-probe seed, but the hybrid purchases the probe for **99.25% of normal-world examples**, scores **below a simple always-probe control**, and is far below the exact selective Bayes reference. No architectural breakthrough or paid GPU scaling.

## Source / exact attempt history

- Isolated branch `experiment/si-original-reward-belief-v9-20261016` stacked on v8 PR #1480 and full original SI PR #1459. No writes to main or other model branches, no Modal/Colab credits, no automations, no paid scientific attempts.
- Protocol **committed before confirmation seeds**, at `docs/SI_ORIGINAL_REWARD_BELIEF_V9_PROTOCOL_2026-10-16.md`; it explicitly recorded two exploratory collapses and forbade tuning after viewing them.
- Exact **full canonical original 298-line module** SHA1 Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`, copied from attached v8 evidence archive and byte-verified with `git hash-object`. V8 world/task file `scripts/si_original_belief_transfer_v8.py` blob `233dff4d660da142fbb57a91cd06b0c4ec17a0d0`, unchanged. New `scripts/si_original_reward_belief_v9.py` **Git blob `9ff12b978ebc4f6554fa8443c13f1c545d4df6ca`** (GitHub and local exact match), SHA256 `3a8e6c4b7ef6fbb282002e19055215c122544765b0a2cab5e438cb4b8f403382`; test source Git blob `226254564caba598db2fb8e4864a76a55deb5ce1` also GitHub/local exact match. Five local tests passed.
- Exploratory excluded seeds **101003,101010,101011**; completed frozen new **101311,101312,101313** each once, with complete JSON stdout and no runtime errors, no reruns, no seed substitutions. CPU PyTorch 2.10, two threads, three models per seed, 240 AdamW steps, batch128, lr=.001, wd=.01, clip1; eval 3,072 independent worlds/family/seed, four families.
- Models: unchanged **OriginalSIHybrid** 64,614 params, matched **TransformerControl** 64,637 params, smaller GRU 10,283 params (NOT parameter/FLOP-matched to hybrid). The full hybrid attention/recurrent/top-1 MoE/optional refinement computation remains; disconnected persistent memory and dormant adapter do not become real continual learning.
- V9 **does not supervise hidden sensor polarity z, supply analytic Bayesian probe choices, or pass true hidden query theta directly into gradient computation**. Cross-sensor correlation target is the next *observable* paired calibration x·y from the causal prefix. Conditional-on-purchase probe correlation target uses y_probe ONLY when purchased. The simulator computes binary action correctness at episode end; from its own executed binary action and correctness feedback the agent reconstructs the correct binary outcome for proper predictive loss. Probe choice is trained via stochastic **REINFORCE** on actual net reward with an exponential-running scalar baseline. This is outcome/reward-supervised, NOT fully unsupervised.
- Neural models receive only actually available cues (not hidden z/theta or unpurchased y_probe). Eval purchases are determined from stage0 logits before stage1 sees the probe. Deterministic seeded input and exploration streams, same across architectures. Bayes is a **known-likelihood upper reference**, not equal-knowledge neural training.

## Frozen three-seed means: reward, accuracy, probe fraction (higher reward better)

| Controller | Base reward | Base accuracy | Base probe rate | Shifted prior reward | Noisy reference reward | Weak-calibration reward |
|---|---:|---:|---:|---:|---:|---:|
| Original SI hybrid | **0.779400** | 0.942274 | **0.992513** | 0.781036 | 0.768240 | 0.763810 |
| Matched Transformer | **0.770612** | 0.879991 | **0.665799** | 0.775861 | 0.752131 | 0.726669 |
| Small GRU | 0.769567 | 0.933485 | **0.998589** | 0.772184 | 0.769066 | 0.764069 |
| Bayesian selective (correct known model) | **0.856832** | not directly comparable in this table | **0.491211** | **0.854573** | **0.834487** | **0.803432** |
| Simple always-probe using Bayes posterior | **0.791820** | not listed | **1.000000** | 0.788247 | 0.783582 | 0.769369 |
| Simple never-probe using Bayes posterior | 0.775825 | not listed | **0.000000** | 0.777886 | 0.736111 | 0.668945 |
| Bayesian without calibration | 0.759485 | not listed | **1.000000** | 0.760035 | 0.759819 | 0.758953 |

**Per-seed base reward / purchase fraction:**

| Seed | Hybrid reward / purchase | Transformer reward / purchase | GRU reward / purchase |
|---|---|---|---|
| 101311 | **0.778200 / 1.0000** | **0.746094 / 0.0000** | **0.762575 / 1.0000** |
| 101312 | 0.787421 / 0.9775 | 0.782363 / 1.0000 | 0.788195 / 0.9958 |
| 101313 | **0.772580 / 1.0000** | **0.783379 / 0.9974** | 0.757932 / 1.0000 |

Hybrid reward wins 2/3 paired seeds vs Transformer, but mean margin is **+0.008788** (less than preregistered +0.015). Crucially, hybrid fails *adaptive purchase* gate because 99% buying, and is **0.012420 BELOW the trivial always-probe Bayesian control** and **0.077432 BELOW selective known-model Bayes**. Transformer is highly seed-sensitive, collapsing to a never-purchase decision in 101311. GRU also buys essentially always. Mean hybrid/Transformer reward alone cannot establish a useful algorithm.

## Evidence of learned observation prediction, but no decision calibration

- Held-out **base next-observable calibration correlation MSE**: hybrid **0.227611**, Transformer **0.228733**, GRU **0.225597**, compared to zero-prediction baseline **0.291084**. Each learned some predictive signal from paired observations; not enough to make cost-aware decisions.
- On OOD `noisy_reference`, zero baseline MSE **0.178549**, whereas hybrid **0.182446**, Transformer 0.178184, GRU 0.175943; on `weak_calibration` zero baseline **0.215430**, versus hybrid **0.230367**, Transformer 0.233928, GRU 0.233248. Prediction transfer is not uniformly good.
- Test-time removal of calibration x/y (distribution-changing ablation) lowers hybrid normal reward **0.779400 -> 0.757737**; Transformer **0.770612 -> 0.672286**, GRU **0.769567 -> 0.758726**. So original hybrid uses some history, but its final action decision is dominated by buying the trusted probe. This is weaker use of calibration than observed in supervised v8.
- Mean CPU train seconds: original hybrid **8.946**, Transformer **6.778**, GRU **1.922**. Hybrid/Transformer ratio **1.32×** on small reference CPU, within prereg 1.5× ceiling but no compute advantage, equal parameter budgets only approximately.
- **Do not mistake classification accuracy ~94.2% for good planning**: it comes largely from always paying for a strong trusted measurement, giving net reward below simple always-buy control and far below selective Bayes.

## Decision and falsifiable next step

**FAIL ALL relevant original hybrid value-of-information advancement requirements** (hybrid vs Transformer margin +0.0088 <+0.015; hybrid below always-buy control; near-100% purchase; reward gap vs Bayes >.03). No cross-family universal advantage or SI breakthrough. The fully source-verified original hybrid is not a justified GPU candidate; no claim of broad architecture impossibility either.

Research diagnosis: **a scalar running-mean REINFORCE baseline gives a high-variance conditional probe-value estimator**. Joint cross-sensor prediction and terminal action feedback learns observation/action structure but does not robustly learn *when measurement cost outweighs benefit*. A next zero-GPU experiment should compare **learned conditional value-of-information / contextual critic** (from observed trajectories only), multi-stage active probing under changing sensor quality, and online belief filters against precisely the **same model backbones and simple policies**; ensure no privileged counterfactual y_probe or hidden sensor-mode label enters action policy. Precommit new unused seeds and strong Bayesian/control thresholds, measure cost-sensitive reward and compute, not accuracy alone.

All outputs/protocol/tests retained. Keep draft. Original exact-head GitHub CI and independent evaluation before any merge. **No Modal credits spent**.
