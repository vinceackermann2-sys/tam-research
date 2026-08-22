# AERA pre-100M validation protocol

Status: preregistered pre-scale gate. **A 100M-class AERA run is forbidden until the mandatory gates below pass.**

This protocol exists to prevent a common architecture-research failure mode: combining many plausible mechanisms, running a large model, then attributing any result to whichever story sounds best afterward.

AERA is evaluated as a system of independently falsifiable mechanisms first, then as an integrated small model, and only then at 100M-class scale.

## 1. Is AERA a next-token-prediction model?

**The architecture is not tied to next-token prediction.** AERA defines a compute/memory/reasoning substrate. Different objectives can train that substrate.

For the first text-only matched comparison, next-token prediction remains the primary objective on purpose:

```text
L_nxt = CE(p(x[t+1] | x[<=t]), x[t+1])
```

That keeps the first AERA-vs-Transformer experiment interpretable. If both see the same bytes/tokens and optimize the same primary language objective, differences can be attributed more cleanly to architecture.

The full AERA objective family is broader:

```text
L = L_next_token_or_byte
  + alpha_event   * L_next_event_latent
  + alpha_state   * L_future_state/action
  + alpha_memory  * L_memory_utility
  + alpha_route   * L_routing/load_balance
  + alpha_compute * measured_or_proxy_compute
  + alpha_halt    * L_unnecessary_reasoning
  + alpha_replay  * L_verified_replay
  + alpha_block   * L_masked/block_generation    # later phase
```

Not every loss is enabled in every experiment. Every run must record exactly which objectives are active.

At inference, the initial fair text model may still decode autoregressively. AERA's later block/masked generator is a separate gate. Therefore:

- first fair text experiment: **next-token training + autoregressive decoding**;
- full architecture: **not fundamentally next-token-only and not fundamentally autoregressive-only**.

## 2. Full AERA component inventory

A 100M run is not allowed to hide missing components behind the name "AERA". The following are separate components with separate status.

### A. Event representation

Purpose: predictable raw input should be compressed into larger events; difficult/high-surprise input should receive finer events and more core compute.

Required mechanisms:

- fixed-token control;
- fixed-byte-block control;
- surprise/entropy-driven variable patches;
- event metadata carrying raw span length and modality.

Required measurements:

- raw bytes per expensive core event;
- patch-length distribution by surprise bucket;
- likelihood normalized per raw byte;
- robustness on code, numbers, rare strings, repeated input.

### B. Cheap controller

The controller may predict:

- expert route;
- stage/layer route;
- reasoning budget;
- attention/working-memory need;
- fast-memory read/write need;
- novelty/surprise;
- retrieval/tool request;
- precision class (research signal until hardware path exists);
- output generation mode.

Controller overhead counts toward active compute.

### C. Conditional expert substrate

Required:

- true top-k execution, not dense expert evaluation multiplied by sparse weights;
- stored vs active parameter accounting;
- capacity/load statistics;
- route entropy and collapse detection;
- load-balancing mechanism;
- grouped/fused GPU path before a speed claim.

Reference target before scale: typical event activates <=25-40% of stored expert parameters.

### D. Working memory

AERA must contain a precise short-horizon mechanism independent of long-term memory.

Initial implementation: local causal attention with a bounded window.

Measure:

- recent exact retrieval;
- causality;
- scaling with window size;
- wall-clock and KV/cache footprint.

### E. Recurrent stream state

Compact state persists across chunks/events.

Required tests:

- finite-state tracking;
- repeated overwrite/current-value tracking;
- chunk boundary continuity;
- long distractor intervals;
- reset-state ablation;
- session isolation.

### F. Fast neural memory / local updates

Fast memory is separate from base weights and stream state.

Required properties:

- read/write during inference;
- verified overwrite of stale values;
- no full-base backward pass for a local write;
- interference/stability measurement;
- per-user/session isolation;
- explicit reset/export/import semantics;
- write gating based on novelty + utility/verifier evidence.

### G. External exact memory / tools

AERA should request external retrieval when neural memory is inappropriate.

The base architecture provides the decision signal/interface; databases/search/tool execution live outside the neural core.

Required controller tests:

- known/in-memory vs exact-retrieval-needed classification;
- false retrieval and missed retrieval rates;
- tool cost included in system compute/latency accounting.

### H. Adaptive latent reasoning

Reasoning occurs in latent state rather than requiring English scratchpad tokens.

Training reference path may evaluate all candidate depths and use a differentiable budget distribution. Inference must execute only the chosen/needed transitions.

Required tests:

- learned difficulty-to-budget relationship;
- accuracy vs latent transitions;
- adaptive vs fixed-depth comparison at matched inference compute;
- hard maximum budget;
- no hidden unbounded loop.

### I. Experience replay and consolidation

Verified trajectories are reusable experience.

Required mechanisms:

```text
attempt -> verifier -> replay record -> local lesson/memory update
                              -> prioritized reuse
                              -> optional small-module consolidation
```

Required tests:

- second-attempt/future-task improvement;
- replay reuse count;
- stale/corrected lesson replacement;
- base-model retention;
- comparison with full fine-tuning and retrieval-only controls.

### J. Block / non-AR generation

This is a separate output-path gate, not required to interpret the first next-token core comparison, but its mechanism must exist and pass a small proof before 100M if the 100M model is advertised as "full AERA".

Required small-scale controls:

- AR decoding;
- block draft + verification;
- masked/block denoising where trained;
- accepted units per expensive core call;
- rejection rate;
- quality and latency at matched compute.

### K. Multimodal adapters and shared world state

Before 100M text scaling, a synthetic/controlled multimodal proof must show that modality adapters can share a latent state without leakage or shape-specific hacks.

Minimum modalities for the mechanism gate:

- symbolic/text-like vectors;
- image/spatial vectors;
- action vectors.

Required tests:

- cross-modal pair retrieval/alignment;
- action-conditioned state prediction;
- modality isolation and shared-state fusion.

Internet-scale multimodal pretraining is **not** required before a 100M text experiment.

### L. Dynamic numerical precision

The controller may expose a precision budget, but **no efficiency claim is allowed from this signal alone**. Real mixed-precision conditional kernels are a later systems project. Until then this field is instrumentation/research only.

## 3. Mechanism gates before any language pretraining

### Gate M0 — correctness

Must pass CI:

- causality where applicable;
- deterministic state semantics;
- no cross-session state leakage;
- exact top-k assignment accounting;
- hard reasoning-step ceiling;
- memory writes do not mutate base parameters;
- replay records immutable provenance/verifier outcome;
- block generator never accepts unverified low-confidence positions in verification mode.

### Gate M1 — learned sparse routing

Synthetic task families with known specialization labels.

Pass:

- route prediction >=95% on held-out synthetic families;
- no expert receives >50% of balanced held-out traffic when task distribution is balanced;
- selected experts actually execute only selected assignments;
- dense-all-experts ablation measured separately.

### Gate M2 — learned adaptive compute

Synthetic examples have known difficulty/depth.

Pass:

- harder examples receive more latent transitions with monotonic positive relationship;
- >=90% budget-class accuracy on held-out synthetic examples;
- hard runtime step count matches selected budget;
- mean transitions lower than max-depth baseline.

This establishes learned allocation plumbing, **not reasoning superiority**.

### Gate M3 — memory stress

At least thousands of randomized operations including:

- write;
- read;
- repeated-key overwrite;
- delayed read;
- distractors;
- two independent sessions.

Pass:

- current-value retrieval materially above chance;
- stale resurrection rate <=5% on the controlled benchmark;
- cross-session leakage = 0;
- base parameter mutation = 0.

### Gate M4 — stream-state carry

Train/evaluate a tiny state tracker across chunk boundaries.

Pass:

- carried-state model > reset-state model on post-boundary queries;
- no advantage when state is deliberately randomized;
- state norm remains finite over long streams.

### Gate M5 — event patching

Controlled predictable + high-surprise raw streams.

Pass:

- predictable spans form larger events than high-surprise spans;
- raw order preserved exactly;
- max/min patch constraints always satisfied.

### Gate M6 — controlled multimodal alignment

Train tiny adapters on paired synthetic modalities.

Pass:

- held-out cross-modal retrieval >=90% in the controlled task;
- unpaired labels remain near chance;
- action-conditioned transition head learns the controlled dynamics.

### Gate M7 — replay/local-learning loop

Pass:

- a verifier correction changes subsequent behavior without base-weight update;
- replay can reproduce the correction after state export/import;
- incorrect/unverified replay cannot overwrite verified memory under the reference policy.

### Gate M8 — block-generation mechanism

Pass on deterministic synthetic sequences:

- block head can learn multi-unit prediction;
- verification rejects intentionally corrupted drafts;
- accepted units/call >1 at high accuracy on the controlled task.

## 4. Integrated small-model gate

Only after M0-M8.

### Stage S1 — 1-5M integration smoke

Goal: prove gradients, optimization, checkpoint/state semantics, and objective accounting.

Use a tiny text corpus or deterministic language-like stream.

Required:

- next-token loss decreases;
- auxiliary losses finite;
- router does not collapse;
- memory/state can be disabled by ablation;
- hard inference and soft training modes agree within defined tolerance where expected;
- checkpoint save/load preserves base model but session state remains separately managed.

### Stage S2 — 10-25M matched text experiment

Controls on the same GPU/data:

1. dense Transformer;
2. AERA dense-routing ablation;
3. AERA sparse core;
4. AERA no-fast-memory;
5. AERA fixed-depth;
6. TAM-v4 or strong recurrent baseline when implementation quality is sufficient.

Fairness requires two views:

**Active-compute matched**
- AERA active parameters/event and measured GPU seconds are compared with a dense model of similar active compute.

**Stored-parameter matched**
- AERA total stored parameters are compared with a dense model of similar total parameters.

A sparse model is not allowed to claim a 10M-vs-10M parameter win if it stores 40M and activates 10M without reporting both numbers.

Required metrics:

- held-out NLL/PPL;
- equal-token curve;
- equal-wall-clock curve;
- stored params;
- active params/event;
- active expert fraction;
- routing entropy/utilization;
- mean reasoning steps;
- memory read/write rates;
- train/inference throughput;
- peak VRAM;
- compile/setup time;
- long-stream memory/state suite;
- standard downstream suite if model quality is sufficient.

### Small-model advance condition

Do **not** authorize 100M unless AERA demonstrates all of:

1. stable training across >=3 small matched seeds;
2. no severe expert collapse;
3. learned adaptive compute is non-degenerate;
4. fast/stream memory gives a reproducible state/update advantage;
5. no catastrophic quality collapse versus dense control;
6. either a measured quality/compute improvement or a unique adaptability/state advantage large enough to justify extra complexity;
7. a credible grouped/fused sparse GPU path. Python-loop sparsity is insufficient.

Preferred quantitative target before 100M:

- >=1.25x measured useful throughput at matched quality **or** >=20% lower measured compute at matched quality, plus
- meaningful memory/continual-update gain.

The broader RFC target remains >=1.5x / >=30% before a strong efficiency claim.

## 5. 100M-class definition and fairness

For MoE/sparse AERA, "100M parameters" is ambiguous. Every 100M run must use explicit terminology:

- `stored_parameters`: every learned base parameter in all experts/modules;
- `active_parameters_per_event`: expected parameters participating in a typical event;
- `active_FLOPs_per_event`: measured/estimated executed operations;
- `controller_and_memory_parameters`: always counted.

The primary dense comparison should be **~100M active-equivalent AERA vs ~100M dense Transformer at matched measured H100 compute**, while also reporting a total-stored-parameter control.

We will not call a 400M-stored / 100M-active AERA simply a "100M model" without qualification.

## 6. 100M launch gate

A 100M-class AERA launch requires a new explicit user authorization after the small-model report exists.

The launch issue must preregister:

- exact commit SHA;
- exact data bytes/hash;
- stored + active parameter counts;
- active expert count;
- objective weights;
- seed(s);
- context/event policy;
- optimizer/schedule;
- GPU type;
- hard dollar/seconds ceiling;
- no automatic retry policy;
- checkpoints/evals;
- matched Transformer control.

## 7. Architecture risks we actively test

AERA can fail for many reasons:

- router overhead erases sparse savings;
- expert imbalance/collapse;
- experts become redundant;
- differentiable training path does not transfer to hard sparse inference;
- fast memory interferes or stores hallucinated/unverified facts;
- recurrent state becomes a lossy bottleneck;
- adaptive reasoning learns to always use max compute;
- compute regularization causes underthinking;
- latent reasoning learns shortcuts that do not generalize;
- replay amplifies erroneous verifier labels;
- event patching hides important rare bytes;
- block generation requires so much verification that latency gains vanish;
- multimodal shared state becomes a bottleneck;
- complexity makes optimization/debugging worse than a simpler Transformer/Mamba hybrid;
- hardware kernels fail to realize theoretical sparsity.

A negative result at any gate is a reason to redesign before scale, not to relax the gate after seeing the result.

## 8. Prior-art anchors that the validation must beat or distinguish from

AERA combines directions with established prior art. Any novelty claim must compare against them rather than redescribing them:

- DeepSeekMoE / DeepSeek-V3: sparse top-k experts, active-vs-total parameters, load balancing, multi-token objective;
- Mixture-of-Depths: token-dependent layer/depth compute under a fixed budget;
- BLT / Fast BLT: entropy-based variable byte patches and block/diffusion/speculative generation;
- Mamba-2 / SSD and strong delta-rule recurrent baselines: efficient recurrent state;
- Titans/MIRAS: test-time neural memory and surprise-based memorization;
- Coconut and recurrent-depth latent reasoning: non-text latent reasoning/test-time depth;
- diffusion/masked language models such as LLaDA: non-autoregressive generation objectives.

The research question is not whether each component can work in isolation; prior work already shows many can. The question is whether AERA's **integrated separation of compute, working state, fast memory, durable memory, reasoning and generation** produces a better measured frontier without untenable systems complexity.
