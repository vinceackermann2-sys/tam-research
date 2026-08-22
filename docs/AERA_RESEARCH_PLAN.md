# AERA Research Plan

Status: preregistered exploration plan. No large training run is authorized by this file.

## Research question

Can an event-based architecture with sparse conditional compute, adaptive depth, explicit memory tiers, local online updates, latent reasoning, and eventually block generation achieve a better **quality / measured active-compute / adaptability** frontier than a matched dense Transformer and strong recurrent/hybrid controls?

## Phase 0 — mechanism sandbox (current)

Code: `tam_research/aera.py`.

No language pretraining yet. The goal is to falsify plumbing assumptions cheaply.

### 0A. Sparse execution

Test:

- N experts stored
- top-k active per event
- exact assignment counts
- route balance
- wall-clock vs dense-all-experts baseline

Pass:

- selected experts only are executed in the reference path
- routing is deterministic under fixed seed/input
- no expert receives all traffic by construction

GPU performance is **not** inferred from Python-level sparsity. Fused/grouped kernels are required before any efficiency claim.

### 0B. Adaptive reasoning

Construct easy/hard algorithmic examples with known difficulty.

Measure:

- reasoning steps per example
- accuracy vs steps
- compute vs fixed-depth baseline

Pass:

- hard examples receive more useful computation on average
- adaptive model matches/exceeds fixed-depth quality using lower mean transitions

### 0C. Fast memory

Tasks:

- key/value write-read
- repeated-key overwrite
- stale fact replacement
- delayed query
- interference with distractors
- separate-session isolation

Pass:

- new verified facts can change later outputs without updating base parameters
- stale values are replaced rather than resurrected
- unrelated session state produces no leakage

### 0D. Stream state

Tasks:

- finite-state machine tracking
- hidden-state prediction
- long repeated updates
- chunked needle/state tracking

Compare against:

- no state
- GRU/SSM state
- TAM-v4 state

## Phase 1 — small text core

Only after Phase 0 tests are stable.

### Scale

Start at 10–25M active parameters, not 100M+.

### Controls

At minimum:

- dense Transformer
- TAM-v4
- strong recurrent baseline (Mamba/delta-rule style when implementation quality is sufficient)
- AERA dense-routing ablation
- AERA no-fast-memory ablation
- AERA fixed-depth ablation

### Data/fairness

- same exact token/byte source and split
- same seed pairs
- same optimizer/precision where applicable
- report stored parameters, active parameters, measured FLOPs proxy, tokens/bytes seen, wall-clock, VRAM, compile overhead
- equal-token **and** equal-H100-second comparisons

### Primary endpoints

1. held-out NLL / byte or token likelihood
2. quality at equal H100-seconds
3. active expert fraction
4. measured throughput
5. state/memory tasks
6. downstream capability suite

### Stop rule

Do not scale if AERA is more complex but fails to improve either:

- measured compute at matched quality, or
- capability/state behavior at matched compute.

## Phase 2 — variable events

Introduce entropy/surprise-based event boundaries.

Controls:

- fixed GPT-2/BPE tokens
- fixed byte blocks
- learned/dynamic event patches

Measure:

- raw bytes per expensive core step
- likelihood normalized per byte
- throughput
- robustness to rare strings/code/numbers
- whether difficult regions receive smaller patches/more compute

## Phase 3 — latent reasoning

Use tasks with verifiable answers:

- arithmetic
- program execution
- shortest path/planning
- code unit tests
- symbolic transformations

Compare under fixed inference FLOPs:

- text chain-of-thought generation
- fixed latent steps
- adaptive latent steps

A latent-reasoning win requires better answer quality/latency or lower measured compute. Hidden states alone are not evidence.

## Phase 4 — experience replay / continual learning

Interactive verified environment:

```text
attempt -> verifier -> replay -> local lesson -> future attempt
```

Measure:

- improvement per verifier-labelled trajectory
- fraction of experience reused
- base-backprop steps avoided
- retention of old capabilities
- stale belief replacement
- cross-user isolation

Compare:

- no memory
- retrieval-only memory
- fast local memory
- periodic adapter/expert consolidation
- full fine-tuning baseline

## Phase 5 — block generation

Only after the core is competitive.

Compare:

- autoregressive baseline
- masked/block diffusion
- speculative local decoder + full-model verification
- diffusion + verification

Metrics:

- accepted output units per expensive full-model call
- end-to-end latency
- quality at matched compute
- verification rejection rate

## Phase 6 — multimodal world model

Start with a controlled paired-modality environment rather than internet-scale multimodal data.

Examples:

- text + image state changes
- simple video + actions
- synthetic spatial navigation

Objectives:

- future latent prediction
- action consequence prediction
- cross-modal retrieval
- planning/value prediction

The shared latent state must outperform separate modality-only models on cross-modal/action tasks before scaling.

## Compute accounting contract

Every AERA result must report:

```text
total stored parameters
active parameters/event
experts selected/event
mean latent reasoning transitions
memory reads/writes
attention window / state size
measured training seconds
measured inference latency
peak VRAM
compile/setup time
quality metric
```

The controller/router itself counts toward active compute.

## Breakthrough threshold

AERA is an interesting signal if one mechanism clearly helps in a controlled experiment.

A broad architecture claim requires approximately:

1. >=3 seeds.
2. >=3 scales including 100M+ active-equivalent scale.
3. clear quality/compute frontier improvement.
4. measured sparse hardware savings, not theoretical activation counts only.
5. robust memory/state gains.
6. continual/local adaptation that avoids full-base backprop for useful updates.
7. latent reasoning advantage at fixed inference compute.
8. no catastrophic routing collapse.
9. strong recurrent/MoE/hybrid baselines.
10. second corpus/modality/environment replication.

## Immediate next experiments

With the current code and without a large GPU run:

1. Benchmark `SparseExpertLayer` against dense-all-experts on CPU and a cheap GPU when available.
2. Build synthetic difficulty-labelled tasks to test whether adaptive halting learns/usefully correlates with difficulty.
3. Build stale-fact/update/interference tasks for `FastWeightMemory`.
4. Compare AERA fast memory, TAM-v4 persistent state, and ordinary recurrent state under identical synthetic streams.
5. Only then design the first 10–25M language experiment.
