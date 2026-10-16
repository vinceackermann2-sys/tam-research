# SI Original: learned attention horizon v1 — complete CPU engineering results (2026-10-16)

**Verdict: FAIL to satisfy dual-objective mechanism gate. No breakthrough, no GPU authorized or spent.** Branch `experiment/si-original-adaptive-horizon-v1-20261016` is a stacked addition to *original hybrid* PR #1459, not the separate EHW architecture. This is a simplified attention-submechanism test; it does **not** train the complete original SI model.

## Source and attempt identity
- Frozen before training: source `scripts/si_original_adaptive_horizon_v1.py` at `fd47e2e38cb5d5b5a1c41951589e2a6e9bb3ba72` and original gate protocol `docs/SI_ORIGINAL_ADAPTIVE_HORIZON_V1_PROTOCOL_2026-10-16.md`.
- Python 3.13, torch 2.10+cpu, two threads. 300 updates per candidate, each batch 16, sequence 24, two causal layers, d_model32, heads4, horizon6, AdamW LR 0.001, weight decay 0.01; same model initialization and action/world samples for all modes.
- Initial smoke seed 95000 only (4 steps, batch4). Causal prefix invariance and nonzero finite gate gradient tests passed locally. All candidates instantiate **20,265 parameters**. Fixed modes leave gate network dormant; this does not imply equal active parameters or FLOPs.
- Original preregistered seeds: 95111, 95112, 95113. Seed 95113's local command was **interrupted by runner timeout**, resulting in a 0-byte JSON output. Do not reuse the seed or claim data. `SI_ORIGINAL_ADAPTIVE_HORIZON_V1_INFRA_ADDENDUM_2026-10-16.md` committed *before* executing fresh replacement seed 95114. Completed seeds: **95111, 95112, 95114**. This is engineering screening with a replacement, **not strict unamended scientific replication**.
- Evaluation comprises 4 × 32 independent unseen worlds per mode and seed, under randomized per-world switch times (8..18). True rule, switch times and future states never supplied to model. Training mixtures include stable worlds and switching worlds. Reference baselines are persistence and zero.

## Per-seed and three-completed-seed results: MSE lower is better

| Seed | Candidate | Stable full MSE | Random switch late MSE | CPU training seconds |
|---|---|---:|---:|---:|
| 95111 | Local | 0.291206 | 0.308904 | 2.989 |
| 95111 | Global | 0.290100 | 0.319673 | 2.739 |
| 95111 | Adaptive | **0.289568** | 0.310160 | 4.847 |
| 95112 | Local | 0.304440 | **0.309965** | 3.778 |
| 95112 | Global | **0.301091** | 0.315802 | 2.704 |
| 95112 | Adaptive | 0.302097 | 0.310322 | 3.486 |
| 95114 | Local | 0.295558 | **0.300316** | 2.598 |
| 95114 | Global | **0.293015** | 0.310400 | 2.717 |
| 95114 | Adaptive | 0.293763 | 0.301309 | 3.773 |
| **Mean** | **Local** | 0.297068 | **0.306395** | 3.122 |
| **Mean** | **Global** | **0.294736** | 0.315292 | **2.720** |
| **Mean** | **Adaptive** | 0.295143 | 0.307264 | 4.035 |

Mean nonlinear OOD full MSE: local **0.298148** / global **0.296129** / adaptive **0.296267**. The current learned gate does **not** dominate either fixed window on stable or switching tasks, so fails the dual-endpoint advancement criterion.

### Actual gating behavior

Mean adaptive global-attention mixing fraction = **0.281609 in stable worlds** and **0.281667 in changing worlds**. This difference (<0.0001) is far too small to establish sensitivity to regime change. Gate weights receive gradients and do not saturate 0/1, but output resembles a global fixed mixture. Calling this *adaptive thinking* would be misleading. The adaptive path computes **both** masks using scaled_dot_product_attention and averages their outputs: **4.035 s average CPU training versus 2.720 s global and 3.122 s local** (tool-time CPU figures; not optimized FLOP comparisons). No demonstrated inference cost saving.

## Scientific interpretation / decision

The earlier local-versus-global effect survives a new generator with *random rather than fixed switch times*: a short fixed horizon performs better in randomly changing worlds, a full horizon does better in stable worlds. This is a promising *tradeoff to understand*, not evidence that a learned agent has successfully discovered and used changes in world dynamics.

**Stop the present soft convex-mixing controller as a paid-GPU candidate.** It neither changes its behavior with the environment nor yields a Pareto improvement, and it uses more compute. This is a negative result for the tested mechanism, not evidence that the larger original SI architecture is impossible.

For any subsequent CPU research: first test a strong **no-history (one-token) predictor**, because the simulated world may remain largely unpredictable without identifying episode-specific transitions; and separately test explicit, strictly causal prediction-error-triggered change detection, but avoid using true switch labels or future target information. Evaluate held-out planning outcomes and a second simulator family, with fixed parameter/active-FLOP controls. If attention history does not beat no-history, redesign the task before spending credits.

Modal/GPU/Colab: **zero spend, zero authorization**. Do not rerun consumed 95113. No changes to main, EHW, CHM, RLT, PGW, CPW or other branches.
