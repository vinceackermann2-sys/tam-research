# Bounded-chunk streaming RLT AW — model-level CPU stage gate

**Decision:** First end-to-end bounded-attention streaming state-carry demonstration is positive for intermediate horizons, but NOT a Transformer-beating or general-purpose language-model breakthrough. Keep the result and no-carry baseline frozen; next investigate gate-state decay and a stronger bounded-window Transformer control, zero GPU first.

Branch: `exp/rlt-colab`. One frozen 20261081-seeded CPU engineering attempt:
- [AW workflow, success](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38066660152)
- Result `experiments/rlt/cpu/results/rlt-bounded-streaming-vs-reset-20261010-aw.json`
- Preregistered job `experiments/rlt/cpu/jobs/rlt-bounded-streaming-vs-reset-20261010-aw.json`
- Model `experiments/rlt/model_streaming_bounded.py`, correctness [21 passing CPU tests](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38066301595).

## Exactly frozen protocol

Two variants share the **same 27,712 trainable parameters**, initial weights, causal per-chunk encoder and decoder attention, cross-attention confined to the current chunk, local position embeddings and gated scan. Full input 128 tokens is processed in sixteen **8-token chunks**. The *only change* is the carried previous scan state versus resetting to the learned initial state at every chunk boundary. The control is therefore a **no-carry RLT control**, not a standalone Transformer.

Both models trained 15s CPU on the eight-token immediate-write task (100% easy holdout for both), then 90s CPU on balanced binary last-write at distances 1, 4, 16, 64, 127. Same frozen seeds and held-out sequences, 128 examples per target class per distance, no cap on optimizer steps. The generation pipeline verified a counterfactual intervention: flipping a write in a prior chunk flips the answer but does not change the final 8-token chunk.

## Independent held-out binary-query accuracy (256 balanced examples/bin)

| Last-write distance | Streaming carry RLT | Same-compute reset control |
| --- | ---: | ---: |
| 1 | 100.000% | 100.000% |
| 4 | 99.219% | 100.000% |
| 16 | **92.188%** | **50.000%** |
| 64 | **94.141%** | **50.000%** |
| 127 | **53.906%** | **50.000%** |

Full held-out NLLs, respectively, by distance:
- Carry: **0.00467, 0.01794, 0.25866, 0.19497, 0.68971**.
- Reset: **0.00235, 0.00282, 0.69610, 0.69539, 0.69592**.

The reset control had 100% class-zero and 0% class-one accuracy at all out-of-window distances (16, 64, 127), confirming a constant-class response under inaccessible history. The carried model's distance-127 accuracy was only 53.9%, class-zero 94.53% and class-one 13.28%, indicating a near-collapse to one class at the longest distance. Do not call this successful 127-token retention.

Training throughput during long phase: carry **10,526.3** tokens/s, **926** steps/948,224 input tokens; reset **10,625.1** tokens/s, **934** steps/956,416 input tokens. CPU wall-clock budgets were 90.081 and 90.015 seconds. Both have exactly matched trainable parameters and attention access, but the small difference in tokens seen means this is **equal wall time, not equal token count**.

## Interpretation limits

AW tests fixed-size recurrent state as *the only cross-chunk information channel*. The no-carry control is deliberately memoryless between chunks. The observed benefit for 16–64-token updates is a capability difference under this intentionally imposed information bottleneck, not proof of superior general language intelligence or superiority over a Transformer with a memory mechanism, KV retention, extended effective context, or equal deployment resources. The baseline is not the conventional full-prefix Transformer used in previous language comparisons.

All 21 model-level CPU tests passed, including exact fixed-chunk API/forward equivalence, no future leakage, carried-state dimension, and gradients from a later chunk to an earlier recurrent state. Full BPTT training activation memory still grows with the number of chunks; the [batch,width] carried state is bounded at inference, but whole-model training is **not constant memory**. Position embeddings are reset per chunk and arbitrary chunk-boundary segmentation is NOT invariant.

## Next zero-GPU research gate

1. Design an independently preregistered, strong **causal sliding-window Transformer** control with a real 8-token KV window *across* chunk boundaries, parameter/compute budgeting, no access to older positions, and no learned cross-chunk state. This distinguishes beating a reset-chunk toy from beating standard bounded-local attention. Do not conflate the baselines.
2. Verify on CPU that exactly the same final accessible-window input can have opposite labels, so the bounded-window control cannot infer distant writes from prompt information, even with perfect optimization.
3. Examine learned gate retention and write magnitudes at distance 127, and whether recency-related information decay, class imbalance, insufficient training time or optimizer setting caused the failure. Use a new CPU job/seed; never rerun AW. An event-driven gate variant would embed task-specific prior knowledge and must be compared separately, not presented as generic memory.
4. Test multiple independent tasks and fresh seeds only if gates pass. A positive toy-task panel could motivate a new, separately authorized scientific language study. Prior AK–AQ evidence still finds RLT weaker than matched Transformer at 15M on FineWeb-Edu NLL.

Governance: freeze results and claims AK–AW, never rerun consumed GPU jobs, protect seeds 58232/58233, no 250M/5B scale or breakthrough claim, read live `main` and `exp/rlt-colab` before every repo mutation, and write only to the isolated branch.
