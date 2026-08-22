# TAM v4 Architecture

Status: implementation-ready research candidate. Not yet a breakthrough claim.

## Why v4 exists

TAM v3 produced a real but mixed signal at 100M/2B: slightly better pretraining/final NLL than a parameter-matched Transformer, but lower average downstream MCQ accuracy and materially lower throughput. Its recurrent world-state branch was used (~55% average learned contribution), and a matched synthetic memory test showed a statistically clear aggregate win at 256 tokens, especially state tracking and needle retrieval, but no broad advantage beyond the 512-token training context.

TAM v4 therefore targets the two clearest v3 weaknesses rather than simply scaling v3:

1. **v3 always computes attention and world-state.** v4 makes routing input-dependent and prepares a hard-routing path so expensive branches can be skipped.
2. **v3 state resets every forward pass.** v4 exposes explicit per-layer recurrent state that can carry across chunks.

The hypothesis is narrower than “recurrence beats Transformers”:

> A parameter-matched hybrid with local exact attention, persistent recurrent state, novelty-gated writes, and adaptive routing can preserve or improve TAM's sample-efficiency/memory signal while materially reducing the compute penalty.

## Core block

For pre-normalized hidden states `h = LN(x)`, each block has two branches.

### 1. Local exact attention

Attention is reserved for precise short-term interactions inside the active chunk/window:

```text
A = LocalCausalAttention(h, window=W)
```

Default research window: the current training chunk length. Future compute experiments can reduce `W` below the chunk length.

The attention inner width remains approximately `13/16 * d_model`, preserving the same parameter-sharing logic as TAM v3.

### 2. Persistent recurrent world state

For state coordinate vector `s`:

```text
candidate_t = tanh(W_candidate h_t)
keep_t      = sigmoid(W_keep h_t)
novelty_t   = sigmoid(W_novelty h_t)
effective_keep_t = 1 - novelty_t * (1 - keep_t)
s_t = effective_keep_t * s_(t-1)
    + (1 - effective_keep_t) * candidate_t
world_t = W_state_out s_t
```

`novelty_t` is a scalar write gate. Predictable/redundant inputs can preserve state; surprising inputs can update it more aggressively.

Unlike TAM v3, `s_(-1)` is an explicit input to the block. The final state of chunk `n` becomes the initial state of chunk `n+1`.

Training can use:

- full differentiable state across adjacent chunks for short streams, or
- truncated BPTT by detaching state between chunks for bounded memory.

### 3. Dynamic token router

TAM v3 used one scalar gate per layer for all tokens. TAM v4 uses a learned token-dependent gate:

```text
g_t = sigmoid(W_router h_t + b)
m_t = 2 * ((1-g_t) * A_t + g_t * world_t)
```

This reference form is differentiable and easy to falsify.

**Important:** the correctness/reference path computes both branches. A compute-efficiency claim requires the hard-routing/kernel stage below; soft gating alone does not save branch FLOPs.

### 4. Hard-routing target

After the soft router learns stable behavior, v4 converts routing into grouped conditional execution. The unit of conditional compute is a short token block (for example 16–64 tokens), not an individual scalar operation, because GPU kernels need enough work per dispatch.

Each block/layer chooses one of:

```text
ATTENTION_ONLY
WORLD_ONLY
BOTH
```

A compute regularizer penalizes expected branch cost:

```text
L_total = L_lm + lambda_compute * E[normalized_branch_cost]
```

The hard path is considered successful only if measured H100 wall-clock/throughput improves; sparse-looking routing statistics without hardware savings do not count.

## Streaming model semantics

TAM v4 separates three concepts:

- **local token context:** exact attention inside the current chunk/window,
- **stream state:** compact recurrent state carried across chunks,
- **durable external/user memory:** outside the base LM and not conflated with recurrent state.

The reference v4 model resets local position indices inside each chunk. Continuity is carried by recurrent state, so the architecture is not intrinsically limited by a learned global position table when used in streaming mode.

`state=None` means a new independent sequence/session. State must never be shared across unrelated users or examples.

## Parameter budget

With `state_size=d_model/4` and `attention_inner=13*d_model/16`, the v3 attention+state mixer is already within <0.01% of the matched Transformer at existing scales. V4 adds only O(d_model) router/novelty parameters per layer, so the full-model mismatch target remains **<0.1%**.

Any future expert/MoE extension is not part of TAM v4 and must use a separate active-parameter/total-parameter fairness protocol.

## Required instrumentation

Every run must log:

- parameter count
- total/steady H100 seconds
- tokens/sec
- peak VRAM
- NLL/PPL learning curve
- mean and per-layer router gate
- gate distribution by token difficulty/surprise bucket
- recurrent write rate (`1-effective_keep`)
- recurrent-state norm and saturation
- branch occupancy for hard routing
- measured branch FLOPs/dispatch counts when available
- chunk-state carry vs reset ablations

## Required ablations

At minimum:

1. `v4-full`: dynamic router + novelty writes + carried state.
2. `v4-reset`: identical model but reset state every chunk.
3. `v4-fixed50`: fixed 50/50 blend, carried state.
4. `v4-no-novelty`: write gate removed.
5. `v4-local-attn-only`: matched attention control.
6. `v4-world-only`: recurrent branch only where parameter matching permits.
7. soft routing vs hard/grouped routing.

## Small-scale preregistered gate

Do not spend on 100M+ until a smaller gate passes.

Initial gate:

- scale: ~25M
- tokenizer/data: same exact bytes as matched Transformer control
- context/chunk: 512
- adjacent-stream test: at least 4 consecutive chunks where available
- seeds: >=3
- equal-token and equal-H100-second comparisons
- same optimizer/schedule/precision/batch

Advance only if all are true:

1. v4 does not lose mean held-out NLL to v3/Transformer at equal tokens.
2. v4 improves the equal-compute frontier or reaches **>=90% of Transformer throughput** after reasonable kernel work.
3. carried-state v4 beats reset-state v4 on delayed update/needle/state tasks by a practically meaningful margin with paired confidence intervals.
4. the router is non-degenerate and hard routing reduces measured compute, not just theoretical FLOPs.
5. no instability or cross-example state leakage.

## Scale gate

Only after the 25M gate:

```text
25M -> 50M -> 100M
```

At 100M, the architecture is worth further scaling only if it demonstrates at least one strong advantage at equal hardware budget:

- >=5% relative capability improvement on a preregistered suite, or
- equivalent capability with >=20% lower measured training/inference compute, or
- a clear long-stream/state advantage unavailable to the Transformer, without catastrophic throughput cost.

If v4 remains slower and broadly worse on capabilities, stop TAM scaling and redirect effort to the clean-sheet architecture program.

## Non-claims

TAM v4 is not yet:

- persistent user memory,
- a replacement for backpropagation,
- an MoE system,
- a latent-reasoning architecture,
- a multimodal world model,
- a parallel/non-autoregressive decoder.

Those are intentionally explored separately so TAM remains falsifiable rather than becoming an unfalsifiable bundle of every promising idea.
