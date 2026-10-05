# CPW v0 — Cortical Predictive Workspace

Status: experimental, branch-isolated, preregistered in issue #1245.

## First-principles objective

A useful general model needs more than one autoregressive output distribution. CPW v0 therefore trains a causal language model whose hidden computation contains several explicit predictive systems feeding a shared latent workspace.

This is functional brain inspiration, not anatomical simulation and not a claim of consciousness.

## Computational mapping

- sensory/cortical processing -> reduced-width causal attention
- world-state prediction -> causal recurrent compressed state
- sequence prediction -> one-step low-rank latent predictor
- memory prediction -> slow prefix-memory latent predictor
- thalamic selection -> learned top-2 router over the four fixed-function branches
- global workspace -> recurrent latent signal carried across depth
- uncertainty -> branch disagreement/router entropy telemetry
- motor/action system -> deliberately outside this text-only v0
- biological reward/emotion/drives -> deliberately absent

## Exact block budget

For d_model=256, a standard full-width attention mixer has 262,144 parameters.

CPW replaces it with:
- half-width attention (inner=128): 131,072
- recurrent world state (state=64): 49,152
- sequence predictor (rank=96): 49,152
- memory predictor (rank=64): 32,768
- predictor subtotal: 262,144
- sparse router 256 -> 4: 1,024

Thus CPW adds only 1,024 router parameters per block. Across 15 blocks this is about a 0.062% model-level increase versus the 24,940,288 parameter Transformer, inside the frozen <=0.1% tolerance.

## Prediction losses

In addition to ordinary next-token cross entropy:

1. one-step predictor reconstructs the normalized current latent state from the previous latent state;
2. recurrent world state predicts the next normalized latent state;
3. slow prefix-memory predictor predicts a latent state four positions ahead.

Targets are detached and these losses cannot reveal future inputs to the causal forward pass.

Total training loss:

L = L_CE + 0.05 * mean(L_sequence, L_world, L_memory).

## Safety by architecture

CPW v0 has no persistent reward head and no intrinsic objective for survival, power, status, pain avoidance, reproduction, resource acquisition or emotional valence. It is a predictive model trained by bounded supervised/self-supervised losses. Tool permissions and agent objectives, if added later, must live outside this core model.

This does not by itself prove a system is safe; learned models can still exhibit unintended behavior. It removes unnecessary biological motivational machinery rather than claiming to solve alignment.

## What counts as evidence

The first screen is only a 25M-scale development test against the repository Transformer. A win is not a breakthrough. A breakthrough claim remains governed by the stronger multi-scale/equal-compute/capability/ablation requirements in docs/breakthrough_protocol.md.
