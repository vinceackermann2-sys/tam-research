# SI Original — adaptive causal attention horizon v1 (CPU-only engineering protocol)
Date: 2026-10-16. Stacked branch `experiment/si-original-adaptive-horizon-v1-20261016` based on **original SI hybrid** research branch (PR #1459), NOT EHW; no changes to other branches or main.

## Prior observation and hypothesis

Original SI scaffold's full stack lost stable-world quality and CPU training speed versus a Transformer, whereas *a matched dense-attention mask control* found that 6-token local attention reduced late error after abrupt dynamics changes but worsened stable-world loss. Separate module ablations failed to credit recurrence, MoE or extra refinement. We test whether **a learned, causal token-wise gate that interpolates fixed-local and full-prefix attention** outperforms both controls without using switch labels.

## Prototype and explicit omissions

The bounded standalone `scripts/si_original_adaptive_horizon_v1.py` implements an action-conditioned predictive world-model **mechanism**, not a complete SI architecture or a replacement for the original hybrid. It uses two dense residual causal-attention blocks, shared prediction/variance/error-verifier heads, and a per-token learned context gate. The original hybrid with recurrence, sparse MoE, modular memory and planning remains in PR #1459, which is the *parent research track*. This test isolates the attention-horizon subhypothesis.

Comparison modes of the **same** model and parameter count:
- `local`: a 6-position causal mask; only one attention call.
- `global`: full-prefix causal mask; only one attention call.
- `adaptive`: per-token gate from current token features and previous-token differences chooses convex mixing of the two attention outputs. No future token, true regime indicator, or target loss enters the gate. The gate is penalized 0.002 times mean full-attention mixing fraction.

All variants instantiate the same gate parameters; fixed modes do not use/update that gate, so **equal nominal parameter counts do not mean equal trainable-active parameters or compute**. Adaptive performs **two SDPA calls per layer** and does **not** save computation, even if it learns to prefer local evidence. No claims of learned true online continual training, memory, text/image/audio or verified multi-step planning from this mechanism-only task.

## Generation

Each synthetic episode has unknown independent action-conditioned transition matrices and action effects. Model input consists only of `(observed state_t, chosen action_t)` and its causal prefixes; model predicts `state_(t+1)`. Episodes have length 24. Training mixture: 50% random stable laws and 50% regime changes at a separately sampled, *unknown per-episode* time 8–18. Evaluation modes: held-out stable, held-out randomly timed switch, held-out quadratic nonlinearity. Hidden matrices, switch labels and future observations are never provided to the model.

Pre-confirmation implementation smoke: seed **95000**, 4 steps/batch 4, for numerical correctness only. No confirmation seed evaluated yet.

## Frozen confirmation gate (engineering, NOT scientific breakthrough test)

**Fresh paired seeds `95111, 95112, 95113`.** For each mode and seed train 300 AdamW updates, batch 16, sequence length 24, 2 blocks, width 32, 4 heads, window 6; LR=0.001, weight decay 0.01, clip-grad-norm 1.0; same seed model initialization and exact deterministic batches. Torch CPU 2 threads; no GPU, cloud issue, or paid compute. Held-out evaluation 4x32 worlds per mode with deterministic evaluation seeds `seed*1000000+87000+5*i`. No post-hoc seed re-use to call replication.

**Primary:** stable full-sequence mean MSE and switch late-half MSE, **both**, averaged across all three paired seeds. Adaptive must improve *both* over the best fixed mask to call the horizon mechanism promising. Report every seed, OOD nonlinear, baselines (persistence and zero), gate utilization stable vs switched, model parameters, and CPU train seconds.

**Go for future CPU research only:** adaptive improves both predeclared endpoints vs the better fixed candidate on three-seed averages, >0 on >=2/3 paired seeds per endpoint; gate nondegenerate (global fraction 0.1–0.9) with evidence that differences between environments are meaningful; and no severe latency blow-up. Not a breakthrough, and still no paid-GPU promotion without independent task families, active FLOP parity, stronger classical/Transformer/recurrent controls and mechanism ablation.

**Stop/redirect this mechanism:** gate collapses, no dual-endpoint benefit, or overhead dominates. Retain all negative results. This does not logically invalidate original SI program, but it deprioritizes the adaptive horizon as currently designed.

No automatic GPU trigger, Modal/Colab spend, rerun of consumed attempts, or modification of main is authorized. Freeze code/head before measuring confirmation seeds. Original exact-head CI + drift inspection required before considering merge.
