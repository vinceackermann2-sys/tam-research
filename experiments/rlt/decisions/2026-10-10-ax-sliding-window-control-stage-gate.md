# AX stage gate — independent 8-token sliding-window Transformer control

Date: 2026-10-10. Research parent: exp/rlt-colab at b1bac447f9327ec15e0b17f60df03519988aba6e.
Status: CPU-only model and data correctness preflight. No new training result or scientific claim.

## Architecture and resource access

- AW carry and reset models: 27,712 trainable parameters, two encoder/decoder stages, attention reset every eight-token chunk, one recurrent scan state (carry only).
- AX independent Transformer: 27,680 parameters (32 fewer, -0.115%), width 32, two causal layers, four attention heads, FF inner 145, tied token/output embedding, local phase positional embedding.
- Each AX layer attends to at most eight positions via true causal sliding attention. The per-layer streaming KV cache retains at most eight tokens, including the newest. Unlike AW reset, AX caches span chunk/API boundaries.
- AX model-level receptive distance is 2*(8-1)=14 tokens, not just 7. Stacked attention can pass information older than the last eight tokens; tests explicitly establish distance-14 sensitivity and distance-15-plus invariance.
- AX streaming KV inference state is two (K,V) tensors per layer, each [batch, 4, at-most-8, 8] (2*2*8*32 = 1024 scalar KV elements per sample at full capacity). AW carry alone is 32 scalar state elements per sample; its per-chunk attention tensors are transient and separately accounted for.
- Fixed-window AX forward reference uses a causal W-band attention mask, validated against a separate token-by-token KV implementation. Global position modulo 8 makes AX chunk-partition invariant; AW intentionally is not.
- This is close parameter matching, NOT equal FLOPs, wall time, tokens or architecture. Benchmark compute, training throughput and activation memory before performance claims.

## Required non-leakage test

Use a balanced last-write counterfactual pair at distances 16, 64 and 127. Flip the last SET instruction without touching subsequent tokens. The labels become opposites but the entire final 15 tokens (the complete causal AX receptive field) remain identical; AX final logits must be invariant. At distances 1 and 4 the answer is accessible normally. Distance 8 may be accessed through layered cache propagation and must not be called impossible.

Because an AX two-layer Transformer is guaranteed to lack distant state for distances over 14, any advantage of carry-RLT at 16–127 would demonstrate a longer accessible memory horizon under this controlled constraint, NOT superiority over full-attention, deeper, larger-window, memory-augmented or language-trained Transformers.

## AW failure diagnosis scope

The frozen AW result shows accuracy 53.906% at distance 127, class0 94.531% and class1 13.281%: severe class bias. It does not include a trained model checkpoint. Therefore direct gate decay, state norms, overwritten writes or optimizer-caused forgetting CANNOT be recovered from AW results alone. The AX diagnostic tool can trace gate saturation, direct retention products, state norms and write norms only for separately supplied model weights. Its initialization-only smoke trace is not an AW learned-state measurement.

## Validation and go/no-go

Preflight locally: 10/10 baseline unit tests passed with torch 2.10.0 CPU; require independent GitHub Actions run before promotion. Run all new AX tests plus frozen AW model correctness tests; do NOT invoke AW's consumed training workflow. Do not write to frozen jobs/results, main, or other architecture lanes.

Next separate pre-registered zero-GPU attempt (not authorized merely by this document): CPU training with fresh unique job ID/seeds, fixed data/optimizer, equal time AND reported tokens, per-class accuracy/NLL at all distances, inference KV/state bytes, training activation memory, tokens/s, effective forward/FLOPs. Compare carry-RLT, AW-style reset, and AX Transformer. Train a fresh carry model with checkpoint if gate decay at 127 is to be measured. Maintain failure evidence.

Interpretation ceiling: correctness gate only; zero new comparative model-quality evidence; no GPU, 250M/5B scale, replication, or breakthrough claim.
