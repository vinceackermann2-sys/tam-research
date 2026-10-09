# AU recurrence-path attribution — post-AT evidence and next gate

**Decision:** The AT adaptive-gating collapse is not reproducible on AU's independent CPU seed, and the 64-token last-write toy does **not** isolate a recurrent-memory advantage over causal attention. **Stop further same-task accuracy tuning**. Next do zero-GPU *streaming-state semantics* engineering before any long-horizon model training. No GPU, scientific replication, breakthrough, or scale authority.

Branch: `exp/rlt-colab`. Frozen AU result: `experiments/rlt/cpu/results/rlt-recurrence-path-attribution-20261009-au.json`. Workflow https://github.com/vinceackermann2-sys/tam-research/actions/runs/37899278588 succeeded. Model parameters 29,504 each. CPU-only, batch 8, 20s easy 8-token + 90s mixed-gap 64-token training, new seeds 20261064, 20271064, 20291064, 20301064. Distances 1, 4, 16, 40, 63. Independent per-distance evaluation has 128 examples of each class. AdamW easy LR 0.003, long LR 0.001. Tests verified semantics and restoration of inference-only interventions.

## Frozen AU results

Both static residual scan and coupled adaptive scan reached **100% held-out accuracy for every gap**, with near-zero binary NLL. Both retained 100% easy-task accuracy directly after phase1.

- Residual scan: 6,622 long-phase steps, 3,390,464 tokens, 90.011s, 37,667 tokens/sec.
- Adaptive scan: 6,597 long-phase steps, 3,377,664 tokens, 90.000s, 37,529 tokens/sec.

| Inference path | Residual scan: gap 1/4/16/40/63 | Adaptive scan: gap 1/4/16/40/63 |
|---|---|---|
| Full trained model | 100% / 100% / 100% / 100% / 100% | 100% / 100% / 100% / 100% / 100% |
| Encoder-only decoder initialization | 100% / 100% / 100% / 99.61% / 98.44% | 98.83% / 100% / 52.73% / 51.95% / 99.61% |
| Recurrent-only initialization | 99.61% / 100% / 100% / 96.48% / 100% | 95.70% / 96.09% / 96.48% / 100% / 100% |
| Static recurrent gain | 100% / 100% / 100% / 100% / 100% | 100% / 100% / 95.70% / 86.33% / 100% |

**Caveat:** These are interventions at inference on already-trained weights. Ablating the scan or encoder can move activations off their training distribution. A performance drop suggests dependence on a learned pathway, **not** that the task could not have been learned without that pathway.

For residual scan, encoder-only still achieves >=98.4% across all gap bins, so the scan state has not been shown necessary for this 64-token toy. Adaptive's much larger encoder-only deterioration for gaps 16 and 40 indicates different pathway use in this seed; it does not establish superior generalization or training efficiency.

On AT's previous seed, adaptive collapsed to 50% while residual and Transformer solved all gaps. On AU's new seed, adaptive solved all gaps. **Do not call the adaptive failure an intrinsic architecture limitation.** This is uncontrolled seed sensitivity in a tiny synthetic training setting. AU did not include a new Transformer training run; comparisons to AT's Transformer are cross-seed and are not causal estimates.

## Next engineering question: genuine streaming state

The present scan can update a state from `(gate[t], write[t], prior_state)` but the **full RLT language model** still requires a causal encoder operating on the in-window token prefix and decoder cross-attention to that prefix; it has no proven persistent-state/streaming interface beyond configured sequence length.

Zero-GPU work should first:
1. Implement **pure associative affine state carry across chunks** (both forward and gradients), and verify exact semantic agreement up to floating-point tolerance with one full-sequence scan and a sequential reference, including irregular/short chunks and nonzero batch-specific initial states.
2. Verify continuity of state after each chunk, no future-token leakage, finite gradients to earlier chunks, and shape/dtype validation. Keep existing public models and all frozen result paths untouched.
3. Document explicitly that this is a scan primitive, **not yet** a streaming language model, and not evidence of generalization or compute advantage.
4. Only after these invariants pass, design a genuinely state-limited long-context architecture and matched control, where the attention-only baseline gets the same available information/computation. Any GPU training would require a separate preregistration, fresh seeds, immutable claim, and explicit scope gate.

No modifications to `main`; never rerun consumed AQ, AT, AU jobs or reuse scientific replication seeds.
