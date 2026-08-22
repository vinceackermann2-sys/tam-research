# AERA — Adaptive Event-Routed Architecture

Status: clean-sheet research RFC / mechanism exploration. **Not a novelty or breakthrough claim.**

AERA stands for **Adaptive Event-Routed Architecture**.

The objective is not “replace attention with one new layer.” It is to design a system in which **computation, memory, learning, reasoning depth, generation strategy, and modality processing are conditional on the information actually present**.

## First-principles objective

Current large models often couple several independent functions into one expensive operation:

- representation
- short-term memory
- long-term knowledge
- reasoning
- generation
- adaptation

AERA separates them and gives a learned controller a budget.

The desired scaling law is closer to:

```text
stored capability >> activated capability per event
compute(event) proportional to useful difficulty
```

rather than:

```text
all tokens -> same layers -> same width -> same basic compute path
```

## The computational unit: events, not uniform tokens

AERA receives modality-specific raw input and converts it into **events**: variable-size latent patches representing a useful amount of new information.

```text
text bytes ─┐
image patches│
audio frames ├─> modality adapters -> event boundary / compression -> shared event latents
video        │
actions      │
sensors ─────┘
```

An event can span more raw input when data is predictable and less when information density/surprise is high.

This is motivated by the same broad efficiency principle demonstrated by Byte Latent Transformer (BLT): predictable byte regions can be grouped into larger patches while difficult regions receive more compute. AERA generalizes the idea from byte patching to the whole compute graph.

## The controller

Every event receives a cheap controller pass producing:

```text
difficulty
novelty / surprise
uncertainty
memory_read_need
memory_write_need
reasoning_budget
expert_route
layer/transition route
precision budget
tool/external retrieval request
output-generation mode
```

Conceptually:

```text
control_t = Controller(event_t, stream_state, budget_state)
```

The controller is deliberately much smaller than the expert substrate. Its overhead must remain a small fraction of total active compute.

## Conditional compute substrate

AERA stores many specialized modules but activates only a small subset per event.

Possible module families:

- language/symbolic experts
- code/math experts
- spatial/vision experts
- audio experts
- action/control experts
- factual/semantic experts
- planning experts
- verification experts
- state-transition experts

Routing is hierarchical:

```text
stage router
  -> expert group
      -> top-k experts
          -> optional extra latent reasoning transitions
```

The controller may also skip entire stages.

The accounting contract is explicit:

```text
total_parameters != active_parameters != active_FLOPs
```

All three must be reported.

## Adaptive compute / depth

AERA does not assign the same number of transformations to every event.

A simple event may execute:

```text
encode -> one sparse transition -> decode
```

A hard event may execute:

```text
encode
-> retrieve memory
-> sparse transition
-> latent reasoning loop x N
-> verifier
-> additional expert route
-> decode
```

The latent reasoner has a learned halt probability and a hard maximum budget.

Training includes a compute penalty:

```text
L = L_task
  + lambda_compute * normalized_active_compute
  + lambda_balance * routing_imbalance
  + lambda_halt * unnecessary_reasoning_steps
```

The point is not simply to minimize FLOPs. It is to learn the best quality/compute frontier.

## Memory hierarchy

AERA does not ask attention or base weights to be every kind of memory.

### 1. Working memory

Precise, short-lived, high-bandwidth.

Implementation candidates:

- local attention
- small active KV cache
- event-level scratch slots

Use for exact recent dependencies.

### 2. Stream state

Compact recurrent state carried continuously through a sequence/environment.

Implementation candidates:

- selective SSM / delta-rule recurrence
- gated recurrent state
- compressed state slots

Use for recent history and environment state without rereading the full past.

### 3. Adaptive neural memory / fast weights

A per-session/user memory module that can update during use.

AERA's initial mechanism is a local surprise-gated associative update, separate from base parameters:

```text
error_t = target_memory_t - Memory(key_t)
write_strength = f(surprise_t, utility_t)
M <- decay * M + eta * write_strength * local_update(key_t, error_t)
```

The update must not require a backward pass through the entire base model.

This is inspired by the broader test-time memorization direction exemplified by Titans/MIRAS, but AERA must be tested independently rather than assuming the mechanism will scale.

### 4. Episodic/exact external memory

For exact facts, citations, files, user events, tool outputs, and large sparse histories:

```text
structured DB / vector retrieval / object store
```

This is not forced into neural weights.

### 5. Stable base weights

Contain general skills, abstractions, priors, language/world regularities, and reusable algorithms.

The architecture deliberately distinguishes:

```text
skill != current fact != user memory != temporary state
```

## Learning without global backprop for every experience

AERA does **not** assume full backprop can be removed immediately. Initial pretraining may still use backprop.

The research target is a two-timescale learning system:

### Fast path — online/local

During use:

- write fast memory
- update small adapters/fast weights
- update state
- store verified experience in replay

No full-base backward pass is required.

### Slow path — consolidation

Periodically:

- sample high-value replay
- distill recurring lessons
- update only relevant experts/modules when possible
- occasionally update the shared base

This converts continual experience into reusable skill without retraining the entire parameter set for every new fact.

## Post-training / experience reuse

AERA treats agent trajectories as reusable data rather than disposable RL samples.

```text
attempt
 -> verifier/environment outcome
 -> replay record
 -> lesson extractor
 -> fast-memory/module update
 -> future retrieval/use
 -> periodic consolidation
```

Replay records include:

- context/event state
- selected modules
- reasoning budget
- action/output
- verifier result
- causal lesson / correction
- novelty and future utility

The research question is whether this reduces the amount of newly generated RL data required for the same improvement.

## Latent reasoning

AERA separates **thinking state** from **communication tokens**.

After input encoding, a latent workspace can undergo recurrent transformations:

```text
z_0 = summarize(current event + memory)
z_1 = Reason(z_0)
z_2 = Reason(z_1)
...
z_k = halt when verifier/controller confidence is sufficient
```

No English chain-of-thought is required for these transitions.

The decoder converts the final latent state into human-visible text/actions only when needed.

AERA must demonstrate that latent transitions improve quality per unit compute; simply hiding text tokens in a larger dense network is not a win.

## Parallel / block generation

Autoregressive output remains a safe baseline, not a sacred requirement.

AERA exposes a block generator that can eventually support:

```text
predict/denoise a block of event/output units
-> verify uncertain positions
-> accept confident units
-> refine/retry remainder
```

This is conceptually compatible with diffusion/masked generation and speculative verification. The Fast BLT line is evidence that block diffusion/speculation can reduce full-model forward passes for byte-level generation; AERA tests the principle at the shared event/output layer.

## Multimodal world state

All modalities map into a shared event/world-state space, but keep modality-specific front/back ends.

```text
vision ─┐
audio ──┤
text ───┤
actions ├─> shared world state -> prediction -> planning -> action/output
sensors ┤
memory ─┘
```

The core training objective should eventually include more than next-symbol prediction:

- predict future latent state
- predict effects of actions
- reconstruct/contrast relevant observations
- estimate value/uncertainty
- language/output likelihood where appropriate

This allows the internal world model to represent continuous/spatial/action information without pretending every modality is naturally text.

## How AERA maps to the nine target requirements

| Requirement | AERA mechanism |
|---|---|
| 1. waste from dense activation | hierarchical top-k expert/stage routing |
| 2. uniform compute per token | variable events + adaptive depth/halt |
| 3. attention doing all memory | local attention + recurrent stream state + neural/external memory |
| 4. facts stored in base weights | stable skills vs updateable knowledge/user/episodic memory separation |
| 5. global backprop cost | local fast-memory/module updates + periodic consolidation |
| 6. wasteful RL/post-training | verifier-backed replay and lesson reuse |
| 7. English-token reasoning | latent recurrent reasoning workspace |
| 8. strictly autoregressive generation | block masked/diffusion/speculative generation interface |
| 9. everything as tokens | modality adapters into shared event/world latents |

## What is genuinely uncertain

AERA deliberately contains mechanisms that are individually plausible but whose combination could fail badly.

Major risks:

- router collapse or load imbalance
- routing overhead erases sparse-compute savings
- experts become redundant rather than specialized
- local learning destabilizes memory
- fast memory catastrophically interferes
- latent reasoning fails to outperform explicit token reasoning
- block generation loses quality or needs too many verification passes
- multimodal shared state becomes an information bottleneck
- system complexity makes optimization worse than a simpler Transformer/Mamba hybrid

Therefore AERA must be developed as **isolated mechanism gates**, not as a giant all-at-once training run.

## Research phases

### Phase 0 — mechanism sandbox

No large language training.

Validate independently:

- actual sparse expert execution
- adaptive halting
- external per-session fast-memory updates without base-weight gradients
- memory isolation
- routing/compute accounting

### Phase 1 — 10–25M text-only core

Use the same corpus/token budget as a matched Transformer and strong recurrent baseline.

Enable only:

- event/patch input (or fixed tokens as control)
- sparse experts
- adaptive depth
- local attention + stream state
- fast memory

Keep output autoregressive to isolate the core.

### Phase 2 — latent reasoning

Add latent recurrent reasoning and evaluate math/code/algorithmic tasks under fixed inference FLOPs.

### Phase 3 — experience replay + local adaptation

Use verified interactive tasks. Measure improvement per generated training token and resistance to stale/incorrect memory.

### Phase 4 — parallel generation

Compare AR vs block diffusion/speculative generation at matched quality and hardware.

### Phase 5 — multimodal world model

Only after text/state mechanisms are validated.

## Go/no-go thresholds before scale

AERA should not receive large-scale compute unless the small system demonstrates several of these simultaneously:

1. **Active compute:** <=25–40% of stored expert parameters active for typical events.
2. **Measured efficiency:** >=1.5x useful throughput or >=30% lower measured inference/training cost at matched quality on the targeted workload.
3. **Adaptive compute:** hard examples receive more transitions than easy examples and the allocation predicts useful quality gains.
4. **Memory:** persistent/fast memory materially improves long-stream update/retrieval while preserving isolation.
5. **Continual learning:** verified experience improves future behavior without a full-base backward pass.
6. **Capability:** no large collapse versus matched Transformer/recurrent controls.
7. **Routing:** no severe expert collapse; utilization is measurable and stable.
8. **Ablation:** removing the claimed mechanism removes the gain.

A single lower NLL result is insufficient.

## Initial references / prior art anchors

These are anchors for comparison, not claims that AERA invented their mechanisms:

- DeepSeek-V3 / DeepSeekMoE: arXiv:2412.19437 — sparse MoE, 671B total / 37B activated per token.
- Byte Latent Transformer: arXiv:2412.09871 — entropy-based dynamic byte patches.
- Mamba: arXiv:2312.00752 — selective state spaces and linear sequence scaling.
- Titans: arXiv:2501.00663 — neural long-term memory/test-time memorization.
- LLaDA: arXiv:2502.09992 — diffusion language modeling at 8B scale.
- Fast Byte Latent Transformer: arXiv:2605.08044 — block diffusion and speculative byte generation.

A proper novelty review is required before any publication claim.
