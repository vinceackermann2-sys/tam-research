# AERA hardware-aware core

Status: pre-100M research architecture. No breakthrough claim. No 100M authorization from this document alone.

## Why this revision exists

The original integrated AERA reference proved mechanism compatibility on CPU, but real L4 measurements showed that token-level sparse dispatch, explicit local-attention masks, token-level recurrence, and per-event memory writes could erase the theoretical compute savings. The hardware-aware core keeps the research goals but changes the execution granularity.

## Causal execution model

AERA processes a long sequence as bounded chunks. Within a chunk, exact short-term detail is handled by Flash-compatible causal attention. Between chunks, compact state and fast memory carry information forward.

For chunk j:

1. Embed tokens/events with positions local to the chunk.
2. Use only the first event of chunk j plus state from chunk j-1 to make chunk-start decisions. This prevents future-token leakage.
3. Read persistent/session memory once for the chunk and broadcast the recalled context.
4. Run exact Flash causal attention inside the bounded chunk.
5. Route the entire contiguous chunk through a stacked expert bank. The controller may choose one or two active experts; stored experts can be much larger than active experts.
6. Use the final causal token representation as the chunk summary.
7. Run adaptive latent reasoning on that summary. This may alter the boundary prediction and future state, but never earlier token logits.
8. Update compact stream state once per chunk.
9. If novelty/write controls permit it, make one compressed fast-memory update for the chunk.
10. Carry stream state and memory into the next chunk.

This produces the memory hierarchy:

- Flash causal attention: exact working memory inside the current chunk.
- Stream state: compressed cross-chunk recent history.
- Fast neural memory: overwriteable session/user/event knowledge without changing base parameters.
- External retrieval: exact files/database/search/tool information outside model weights.
- Stable parameters: reusable skills and abstractions.

## Conditional computation

The controller exposes separate decisions for:

- expert identity,
- expert count (1 or 2 in the reference),
- latent reasoning depth,
- state read strength,
- memory read/write,
- novelty,
- retrieval need,
- precision budget,
- block-generation suitability.

The architecture therefore does not assume equal computation for every chunk. In production, easy chunks can use top-1 experts and shallow latent depth, while harder chunks can use top-2 and more reasoning steps.

## Generation

Next-token prediction remains the primary objective for the first matched Transformer comparison so architecture and objective are not confounded. AERA also retains:

- next-event latent prediction,
- block-draft prediction plus verification,
- latent reasoning at chunk boundaries,
- future support for masked/block or diffusion-style generation.

AERA is therefore not defined by next-token prediction even though the first controlled language experiment uses it.

## Learning speeds

Stable parameters still use ordinary backpropagation in the initial language experiments. Fast memory has a separate local delta-rule update that does not mutate base parameters. Verified replay/consolidation remains a separate slower learning path. This deliberately separates:

- immediate local memory updates,
- replay/consolidation into selected modules,
- occasional global model training.

Replacing global backprop at frontier scale remains an open research goal, not a solved claim.

## Evidence that forced this design

Valid L4 systems results before this revision:

- Original token-level top2/8 experts: 0.59x dense speed (failed).
- Patch32 top2/8 experts: 0.50x dense (failed).
- Full-chunk grouped top2/8 experts at batch16: 2.52x dense (passed throughput).
- Full-chunk batch1 ModuleList/gather expert path: 0.94x dense (failed latency).
- Stacked fused batch1 top2/8 candidate: 1.45x dense with exact numerical agreement (real speedup, but below the old 2.0x target).
- Chunk adaptive depth: 2.60x for depth2 vs depth4 at batch1 and 1.33x for mean2.5 vs depth4 at batch16 (passed).
- Flash chunk attention replacement: 2.71x faster than the old explicit masked local-attention implementation.
- Fused stream implementation: 1.57x faster than the old Python scan, but token-level recurrence remained too expensive in absolute terms; the new core therefore updates stream state once per chunk.
- Patch16 fast-memory writes: 11.37x faster than per-event writes; the new core writes once per chunk/event summary and can further suppress writes by novelty.

These measurements justify the execution redesign. They do not prove language quality or a breakthrough.

## Causality constraints

Any controller decision that changes representations for every token in a chunk must not inspect future tokens. The canonical start controller therefore uses only the first event plus prior state. End-of-chunk controls can inspect the final causal representation, but may only affect the boundary position, persistent state, memory, or future chunks.

A future learned event patcher must preserve this rule.

## Remaining gates before 100M

1. Full repository CI and causal/state/memory gradient tests for the hardware-aware core.
2. CPU learned integration test using the new core.
3. End-to-end L4 systems benchmark of the integrated core rather than isolated components.
4. Tiny real-corpus language training (roughly 1-5M scale/tokens) verifying stable loss reduction, healthy routing/count/depth decisions, checkpointing, state carry, and block drafting.
5. Matched 10-25M AERA vs dense Transformer experiment on the same data and measured GPU budget.
6. Only if the quality/compute frontier is promising: freeze a 100M-active protocol and compare AERA, TAM, and a matched Transformer with stored parameters, active parameters, wall-clock GPU seconds, tokens/sec, loss curves, downstream evals, memory/state evals, and generation latency all reported separately.

No threshold may be retroactively weakened to call an existing failed experiment a pass. Architecture changes create new experiments with explicitly stated new hypotheses and thresholds.
