# TAM v3 Architecture

## Purpose

TAM v3 is the current primary architecture in this repository. It is a parameter-matched language-model alternative to a vanilla causal Transformer block. The core hypothesis is that replacing part of attention capacity with a learned recurrent world-state can improve language sample efficiency while retaining competitive hardware efficiency.

This file documents the intended architecture independently of any one experiment.

## Design goals

1. **Causal** — no future-token leakage.
2. **Near-exact parameter match** to the Transformer baseline.
3. **Self-similar scaling** from 25M upward.
4. **Useful recurrent state** that can carry/update compact information through a sequence.
5. **Retain attention** for flexible token-token interaction.
6. **Efficient parallel training** — recurrence evaluated via an associative parallel scan rather than a Python sequential loop.
7. **Simple enough to ablate** — TAM v3 intentionally removed the ATAM branch after 3/3 no-memory ablations improved language loss.

## Block

For pre-normalized hidden states `h = LN(x)`:

### Attention path

```text
q, k, v = W_qkv h
A = causal_scaled_dot_product_attention(q, k, v)
a = W_attn_out A
```

The TAM attention path is narrower than full Transformer attention. Its width is chosen so that attention + recurrent-state parameters nearly exactly equal the full Transformer attention mixer.

### World-state path

For every token position `t` and state coordinate:

```text
candidate_t = tanh(W_candidate h_t)
keep_t      = sigmoid(W_keep h_t)
b_t         = (1 - keep_t) * candidate_t
s_t         = keep_t * s_(t-1) + b_t
world_t     = W_state_out s_t
```

Initial state is zero.

The recurrence is a diagonal affine recurrence. Each position represents an affine transform `(a_t, b_t)` with:

```text
s_t = a_t * s_(t-1) + b_t
```

Affine transforms compose associatively:

```text
(a2, b2) o (a1, b1) = (a2*a1, b2 + a2*b1)
```

Therefore the full causal recurrence can be computed by a parallel associative scan. Current canonical implementation: `diagonal_affine_scan()` in `tam_research/models.py`.

### Mixture

TAM v3 uses one scalar gate per block:

```text
g = sigmoid(gate_logit)
m = 2 * ((1-g) * attention + g * world)
```

The factor `2` keeps the initial 50/50 mixture on a comparable residual scale.

The block is:

```text
x = x + TAMV3Mixer(LN1(x))
x = x + MLP(LN2(x))
```

MLP:

```text
Linear(d_model, 4*d_model, no bias)
GELU
Linear(4*d_model, d_model, no bias)
```

## Parameter matching

Current matched family:

| Scale | Transformer | TAM v3 | Difference |
|---|---:|---:|---:|
| 25M | 24,940,288 | 24,941,263 | +975 |
| 50M | 49,799,808 | 49,801,457 | +1,649 |
| 100M | 101,803,520 | 101,806,616 | +3,096 |

Differences are far below 0.01%.

Self-similar TAM ratios:

```text
state_size = d_model / 4
attention_inner = 13*d_model / 16
```

Current scales:

```text
25M:  d_model=256, layers=15, heads=8,  state=64,  TAM-attn=208
50M:  d_model=384, layers=17, heads=12, state=96,  TAM-attn=312
100M: d_model=512, layers=24, heads=16, state=128, TAM-attn=416
```

## What TAM v3 is NOT

- It is not ATAM.
- It has no associative-memory branch in the language mixer.
- Its recurrent state currently persists only through the active sequence, not across independent inference sessions.
- It is not an agent architecture by itself.
- It does not replace external durable agent memory.

## Why ATAM was removed from the main LM path

TAM v2 used attention + ATAM + world-state with a learned router. In the 10M-token mechanism ablation, removing ATAM produced the best loss. The result was replicated across three seeds. This made attention + world-state the simplest mechanism supported by the evidence.

ATAM remains valuable as a separate memory research path because it was excellent on synthetic associative-recall tasks, but current evidence does not justify paying its language-mixer compute cost.

## Current systems choices

- H100 training uses `torch.compile` with `max-autotune-no-cudagraphs`.
- Main stable batch configuration at current scales is micro=64, grad accumulation=2, keeping 128 sequences per optimizer update.
- Persistent Inductor/Triton graph caches are stored on the Modal Volume.
- Exact GPU class is pinned as `H100!` for fair hardware comparisons.

## Core hypothesis to falsify

A useful TAM v3 result requires more than lower equal-token loss. The architecture should eventually show a competitive or superior frontier in:

```text
loss vs parameters
loss vs tokens
loss vs training FLOPs / H100-seconds / dollars
long-context state retention
real downstream capabilities
```

If the recurrent-state advantage disappears at 100M+, under residual controls, on a second corpus, or against strong modern recurrent/hybrid baselines, the hypothesis should be revised rather than defended by moving the goalposts.
