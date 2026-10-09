# Streaming affine state carry — zero-GPU stage gate

**Status: scan primitive validated; streaming language model NOT implemented. No breakthrough or GPU authority.**

On `exp/rlt-colab`, added the pure function `chunked_affine_scan` in `experiments/rlt/streaming_affine_scan.py`, and CPU tests `tests/test_rlt_streaming_affine_scan.py`. Workflow https://github.com/vinceackermann2-sys/tam-research/actions/runs/37900161637: **47 tests passed, 1 nonfailing warning**, no GPU.

## Mathematical guarantee within tested scope

For a learned gate/write stream and initial state,

```
s[t] = gate[t] * s[t - 1] + write[t]
```

the implementation splits gates/writes into arbitrary chunks, runs the existing inclusive associative prefix scan on each, and carries its last state into the next chunk **without detaching**. Tests cover:
- Full-sequence and explicit sequential-reference forward agreement over lengths 1, 3, 17, 65, 129 and chunk sizes 1, 2, 8, 16, 64, 256 (including irregular final chunks).
- Shared and per-batch nonzero initial states in float32/float64.
- Full-BPTT gradient agreement against a monolithic scan, nonzero earlier-chunk gradients, and finite grads.
- Independent call boundaries/resumption with state carry across three separated calls.
- Prefix non-leakage when later gates/writes change; input tensors untouched.
- Validation of chunk-size, shapes, dtype and nonempty sequence.

The same mathematical state rule can be carried over an arbitrarily long stream **of already-computed gate/write vectors**. This is not evidence of efficient end-to-end language-model context extension or a scientific performance advantage.

## Why this does not yet make RLT a streaming LM

Current `GatedScanLightStateRLT` and its residual/adaptive descendants compute `memory = encode(tokens)` using causal self-attention over the entire supplied prefix, position embeddings limited by `max_seq_len`, and decoder cross-attention to the encoder's available memory. Their decoder also uses sliding-window attention *within* the call. Carrying just the affine scan state across two disjoint chunks does NOT preserve encoder features, cross-attention history, global absolute-position semantics or the decoder's local KV cache. Do not claim `forward(concat(a,b)) == streaming_forward(a,b)` for the full model.

## Next falsifiable model-level objective

Build a **separately named streaming-capable model**, rather than patching the frozen 15M language architecture in place. Its specification must explicitly define:

1. State and cache boundary API, including affine state, encoder attention state or a deliberate fixed-context encoder, bounded cross-attention memory, sliding decoder KV, and position handling beyond prior `max_seq_len`.
2. Exactly which history is accessible at each token, and matched information access for an equally parameter/compute-budgeted Transformer control.
3. Full-sequence vs chunkwise causal/prefix equivalence tests at inference **and training**, gradients to prior chunks where promised, memory growth bounds with stream length, and robust reset/batch handling.
4. A hard task requiring information beyond both models' finite attention windows: balanced last-write/copy/state-binding heldout measures at longer horizons. No training advantage can be inferred from the current 64-token toy because encoder-only ablations solved most of it.
5. CPU-only correctness and leakage tests before any paid experiments; any new GPU job needs fresh seed, immutable preregistration, preclaim data/hash checks, one-shot `retries=0`, and separate authorization scope. No protected 58232/58233 seeds, 250M/5B scaling, or breakthrough claims.

## Prior evidence cross-reference

- AQ: adaptive beats static residual by only 0.010055 NLL on engineering seed but loses to Transformer by 0.095355; no scientific advantage.
- AT: adaptive collapsed to 50% on one toy-task seed, while residual/Transformer reached 100%.
- AU: on a fresh CPU seed, both residual and adaptive reached 100%; residual encoder-only still reached >=98.4% by gap, so recurrence has not been isolated as necessary. The adaptive collapse was not reproducible and should not be stated as an inherent limit.

All prior AR–AU results remain immutable. Work only on `exp/rlt-colab`; do not modify `main`.
