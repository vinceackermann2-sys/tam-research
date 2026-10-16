# Original SI v10 — contextual logged-outcome probe value, 3-seed CPU results (2026-10-16)

**Decision: Frozen contextual-value v10 does NOT pass the preregistered promotion gate. No SI breakthrough.** It fixes the all-probe collapse of the previous scalar-REINFORCE architecture for the two larger models, but the original hybrid loses to the matched Transformer on net reward in all three fresh seeds and in every held-out family. The Transformer exceeds strong always/never controls, but remains too far from known-model Bayes to pass the frozen value-mechanism threshold. Do not promote to paid GPU compute.

## Source provenance and attempt integrity

- Isolated stacked research branch `experiment/si-original-contextual-value-v10-20261016`, v9 PR #1481 → original screenshot-inspired full-hybrid PR #1459. Other model branches and `main` untouched; no Modal, Colab, GPUs, automations or scientific seed reuse.
- Protocol preregistered **before** new confirmation attempts: `docs/SI_ORIGINAL_CONTEXTUAL_VALUE_V10_PROTOCOL_2026-10-16.md`. It explicitly records unfavorable development outcomes and freezes selection criteria.
- Canonical original hybrid model Git blob **`575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`** (complete 298-line implementation) verified locally; unchanged v8 task generator Git blob `233dff4d660da142fbb57a91cd06b0c4ec17a0d0`; the v9 parent also unchanged.
- Executed v10 file `scripts/si_original_contextual_value_v10.py` Git blob **`c8ade2177f495e0448638cc80d98dc6adde4fb51`**, SHA256 **`f06aebb4519abab3b074beff6c42871c60a063110f8720aa188148d1586dd5bb`**, byte-identical to GitHub. New test `tests/test_si_original_contextual_value_v10.py` Git blob **`dc6f5048950af06ba045f0875076b829414fd56e`**, also exact local/GitHub match. **Five targeted local tests passed**: original source/parameter count, no future probe/hidden truth in neural inputs, binary correctness-feedback label reconstruction, zero gradient on unobserved value arm, finite bounded training.
- Exploratory excluded seeds: `102000` smoke (4+4 updates), `102010` partial 150+150, `102011` full 240+260 with 2048 evaluation. No hyperparameter/threshold tuning after either development run.
- **Fresh frozen completed confirmation seeds 102211 / 102212 / 102213**; all executed once with complete JSON and no failures/retries/replacement. For every model and seed: **240 action-predictor updates + 260 contextual-value updates, batch128**, CPU PyTorch2.10, two threads, action AdamW LR=.001/WD=.01, value-only output-head LR=.002/WD=0, clip1. All models get identical seeded training worlds, randomized logged probe decisions, and execution feedback. Eval 4,096 independent unseen worlds per family/seed = 12,288 per family, same evaluations for all models and Bayesian controls. The system has no equal active FLOP/energy training comparison, and only three independent model seeds.

## Exact method: offline logged bandit feedback, not oracle prediction

Task has six observed noisy paired sensor calibrations, hidden episode polarity, a new binary-mode query, and an optional trusted probe with observable cost. Train action predictor with **randomly purchased** probes and randomized final actions, observing correctness after executing the action; only action+correctness is used to construct the binary supervised outcome. Freeze predictor/feature representation; independently randomize buy/skip in fresh training episodes, compute actual final success of the frozen policy, and update only the purchased arm's output head. Never create the *unobserved* arm's correctness target. Inference: `buy iff sigmoid(Q_buy)-sigmoid(Q_skip)>known_cost`. **No latent sensor labels, true theta passed directly as optimizer targets, analytic Bayes purchase labels, or unpurchased probe results reach neural inputs.** This is action-feedback-supervised learning with a randomized logging policy, NOT fully unsupervised belief discovery; Bayes is a known-simulator upper reference.

Backbones unchanged: **OriginalSIHybrid 64,614 params**, same-width/nominal-parameter **TransformerControl 64,637**, smaller **GRU 10,283**; GRU is not parameter-matched. Original hybrid retains attention/recurrent/top1 MoE/refinement but its external persistent episodic memory, untrained adapter, real modalities and broader SI features are NOT active. Two sigmoid Q output heads occupy the already-present 3-vector output channels; no additional learnable critic network or future inference cheating.

## Primary 3-seed in-distribution result

| Controller | Net reward (higher better) | Correct final actions | Probe purchased | Heldout randomized arm Q Brier (lower better) |
|---|---:|---:|---:|---:|
| Original SI hybrid | **0.798219** | 0.86922 | 0.60612 | **0.124943** |
| Matched Transformer | **0.808814** | 0.88607 | 0.62085 | **0.120963** |
| GRU | 0.760063 | 0.92464 | **0.99715** | 0.160627 |
| Bayesian selective (knows law) | **0.850508** | — | Adaptive | — |
| Always-probe Bayesian | **0.786841** | — | 1 | — |
| Never-probe Bayesian | **0.779704** | — | 0 | — |
| Bayesian without calibration | 0.759986 | — | Simulated optimal with no calibration | — |

**Per frozen seed on normal worlds**:

| Seed | Hybrid net / probe rate | Transformer net / probe rate | GRU net / probe rate |
|---|---|---|---|
| 102211 | 0.802840 / 0.6040 | **0.804421 / 0.6357** | 0.757332 / 0.9998 |
| 102212 | 0.786900 / 0.5508 | **0.813722 / 0.6125** | 0.760428 / 0.9917 |
| 102213 | 0.804917 / 0.6636 | **0.808299 / 0.6143** | 0.762428 / 1.0000 |

**Comparisons to prerecorded gates**:
- The original hybrid loses **3/3 paired seeds** to Transformer; mean difference **-0.010595**, whereas the gate required >=+0.015 and >=2/3 seed wins.
- Hybrid exceeds always-probe only by **+0.011378** (needs >=+0.015), and never by **+0.018515**. Its Bayes gap **0.052289** exceeds the allowed .035.
- Transformer exceeds both always and never by **+0.021973** and **+0.029110** respectively, and has healthy 0.613–0.636 probe rates and calibrated Q Brier **0.120963**. However its gap to known-model Bayes **0.041694** exceeds the frozen .035 cutoff, so **even the contextual value-mechanism advancement gate formally fails**.
- GRU buys nearly always (99.7% average) despite value head and has Q Brier 0.160627, failing probe and calibration gates.
- Mean training CPU wall time: hybrid **7.594s**, Transformer **6.338s**, GRU **1.801s**; hybrid 1.20× Transformer, within 1.5× ceiling but no speed advantage or actual active FLOP parity.

## OOD transfer, 3-seed mean net reward and probe rate

| World family | Hybrid reward / probe rate | Transformer reward / probe rate | GRU reward / probe rate | Known-model Bayes reward | Always-probe Bayes reward |
|---|---|---|---|---:|---:|
| Training-family held-out base | 0.798219 / 0.6061 | **0.808814 / 0.6209** | 0.760063 / 0.9972 | **0.850508** | 0.786841 |
| Inverted hidden polarity prior | 0.801240 / 0.6166 | **0.809750 / 0.6239** | 0.758945 / 0.9986 | **0.852292** | 0.785409 |
| Noisier trusted reference | 0.774567 / 0.6217 | **0.787713 / 0.6234** | 0.764475 / 0.9972 | **0.829377** | 0.781515 |
| Weaker calibration evidence | 0.750863 / 0.6586 | **0.766221 / 0.7023** | 0.760017 / 0.9997 | **0.801931** | 0.773921 |

No OOD family shows a hybrid net reward advantage. On weak calibration, both learned value heads purchase more evidence, especially Transformer (70%). The GRU still relies on nearly always purchasing the probe.

Removing calibration x/y at evaluation without retraining (distribution-changing ablation): hybrid base reward **0.798219 -> 0.697440**, Transformer **0.808814 -> 0.698962**, GRU **0.760063 -> 0.758755**. The hybrid and Transformer are using calibration histories to improve decisions; the GRU is predominantly using the paid probe. This is not proof of the hybrid's superiority.

## Interpretation, limitations, and stop decision

The major **mechanism-level improvement** over v9 is learning *two separate contextual success-value estimates* from randomized, actually observed logged actions. It prevents the earlier almost-universal purchasing failure for two full-size neural backbones. But **v9 and v10 used different evaluation seeds, training objectives, and architecture control choices**, so the change is an engineering comparison, not a strictly paired causal ablation against an identical v9 training setup. The learned Q estimates remain imperfect, especially under distribution shift. Full original SI recurrence/MoE/refinement still does not provide measured advantage over the equal-size Transformer.

**NO-GO for unchanged original SI hybrid and for the prespecified v10 mechanism promotion gate.** Next CPU research should disentangle representation versus value head with a strict paired v9-v10 same-seed ablation, stronger explicit Bayesian / epistemic uncertainty critics, cross-episode sensor drift with genuine online adaptation, and dynamic multi-stage information acquisition where fixed probe/skip cannot trivially solve the task. Must retain simple classical cost-aware controls, final reward, Q calibration, active hardware-FLOP accounting, no-oracle-input tests, and source identity. No GPU, scaling, automatic tasks, breakthrough claims or main merge.
