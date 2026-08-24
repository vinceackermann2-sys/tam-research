# AERA-v18 fast-memory integration protocol

Status: **CPU-first development only. No GPU run is authorized by this document.**

## Why v18 exists

AERA-v17 established a credible routing/depth candidate on real language:

- actual per-example optional-stage rates were close to the intended 50% / 33.3% / 16.7%;
- held-out difficulty-dependent compute was non-degenerate and strongly monotonic;
- quality remained within the preregistered small-model tolerance;
- systems probes showed AERA throughput improves sharply with batch size and crossed the matched Transformer at batch64 on the development checkpoint.

However, the v17 matched language path still calls the model with `update_memory=False`. That means its real-language evidence does **not** validate the integrated fast neural memory required by the pre-100M protocol.

Simply setting `update_memory=True` would also be scientifically weak. The existing deployment update is intentionally `no_grad` and detached so an inference-time write cannot mutate or backpropagate through base weights. Therefore the write-side memory projections and write-strength controller outputs do not receive useful later-chunk task gradients from that deployment rule.

V18 adds a training-only differentiable form of the exact same delta equation.

## Frozen invariants from v17

V18 must not change:

- AERA-v17 stage/router architecture;
- pairwise difficulty-ranking teacher;
- exact optional-stage target rates 0.50 / 1/3 / 1/6;
- deployment hard threshold p >= 0.5;
- chunk size / context geometry;
- expert count or expert execution semantics;
- latent reasoner or maximum depth;
- recurrent stream state;
- fast-memory state shape, learning rate, decay, or delta equation;
- production corpus bytes/tokenizer;
- stored parameter count or checkpoint key layout.

The only intended behavioral difference during base pretraining is that fast-memory writes are enabled and the already-existing delta update remains in the autograd graph so future-chunk loss can train memory q/k/v/out and write strength.

## Training vs deployment semantics

### Base pretraining

For one memory item `x` and write strength `s`:

```text
k      = normalize(K(x))
v      = tanh(V(x))
pred   = k @ M
error  = v - pred
M'     = decay*M + lr*s*outer(k, error)
```

No detach is applied during base pretraining. This is allowed by the master protocol: base pretraining may use ordinary backpropagation.

### Deployment / local adaptation

The established `DeltaFastMemory.local_update` remains unchanged:

- inputs/write strength are detached;
- the returned memory state is detached;
- base parameters are not mutated;
- no full-base backward pass is required for a local write.

Thus pretraining learns *how to use the memory substrate*, while deployment updates the session-local memory state without changing base weights.

## CPU gates before any v18 GPU development run

All must pass:

1. **Equation equivalence** — differentiable and detached updates are numerically equal for identical inputs/state.
2. **Trainability** — a future memory read loss produces finite nonzero gradients for q/k/v/out and write strength.
3. **Deployment immutability** — local deployment writes mutate zero base parameters and return detached memory state.
4. **Checkpoint compatibility** — replacing v17 memories with the v18 wrapper changes neither stored parameter count nor state-dict keys/values at initialization.
5. **Memory-disabled equivalence** — v17 and v18 produce identical outputs when memory updates are disabled.
6. **Causality** — a write performed at a chunk boundary cannot alter logits from that completed chunk; only later chunks may read it.
7. **Equation parity in-model** — differentiable-pretraining and detached-deployment modes produce the same logits for the same write-enabled forward, within numerical tolerance.
8. **Inference-safe default** — a freshly constructed v18 model defaults to detached deployment mode, never differentiable local adaptation.

## What a later v18 development GPU gate must measure

A later explicitly authorized run should not merely ask whether training completes. It must measure:

- language NLL vs the matched Transformer and v17 no-memory control;
- held-out memory-enabled vs memory-disabled NLL on later chunks;
- state-reset vs carried-stream vs carried-stream+fast-memory decomposition;
- memory read/write rates and write-strength distribution;
- whether writes collapse to always-on or always-off;
- base-parameter mutation = 0 during deployment-memory evaluation;
- stale/interference behavior under long-stream repeated-key tests using the trained checkpoint;
- v17 routing/adaptivity gates unchanged;
- batch8 and device-saturating throughput, including memory overhead;
- peak VRAM and memory-state bytes/session.

No v18 result may count as independent replication until the integration configuration is frozen. Development seeds remain excluded from the required >=3 fresh independent seeds.

## 100M boundary

V18 does not authorize 100M. Before scale, the project still requires:

- a viable integrated small-model candidate;
- real-language dense-routing / fixed-depth / memory-state ablations;
- >=3 fresh matched replication seeds after architecture/config freeze;
- reproducible state/update advantage;
- final nine-requirement audit;
- a new explicit user authorization for any 100M-class run.
