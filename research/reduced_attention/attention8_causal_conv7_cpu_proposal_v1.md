# Attention-8 + causal depthwise conv7: CPU-only hypothesis v1

**Status: untrained architecture prototype only.** This is a new isolated research branch. There is **no GPU training, no scientific seed, no result claim, and no authority to scale up**. The frozen unsuccessful scientific Pair-1 attempt (#1300, seed 60232) must not be rerun or altered.

## Hypothesis (not a result)

The prior 8/24-attention sparse model was +0.045086746 NLL worse than the parameter-matched Transformer after ~2B token exposures, although 1.074995× faster at training. Its 16 FFN-only layers have no cross-token exchange. Add a small **causal depthwise convolution** to each of those 16 blocks while retaining the same eight full-attention layers [3,6,9,12,15,18,21,24]. A local token mixer could recover some intermediate sequence information at far less cost than full attention, but it may also remove the speed benefit or fail to improve quality.

## Construction

- 24 residual blocks, width 512, 16 heads on each of eight original full-attention blocks.
- In 16 FFN-only blocks, add a **left-padded (strictly causal) depthwise 1D convolution**, kernel 7 (per-channel weights, no bias), followed by a trainable scalar residual gain initialized to zero.
- Keep the FFN-only block residual and existing normalization/FFN. No access to future tokens is allowed. CPU tests check this and the zero-gain baseline equivalence.
- Decrease every block's FFN hidden width from 2731 to **2729** for parameter parity.

### Exact parameter arithmetic

| Item | Parameters |
|---|---:|
| Existing attention-8 model, FFN hidden 2731 | 101,795,328 |
| FFN hidden-width decrement of 2 across 24 blocks | −49,152 |
| New 16 × (512 × 7 depthwise weights + 1 scalar gain) | +57,360 |
| **New candidate total** | **101,803,536** |
| Matched Transformer baseline | 101,803,520 |
| **Difference** | **+16** |

Matches *trainable parameter count*, not training FLOPs, memory or wall time. The causal convolution introduces another memory/operation pattern, and latency improvements must not be presumed. The proposal does not claim matched initialize/optimizer trajectory.

## What can be asserted without GPUs?

- Parameter formula and instantiated model parameter count; attention/conv placement; no-replication/no-GPU flags.
- Strict causality at the FFN-only conv block under perturbations of future tokens.
- At gain=0, the upgraded block computes exactly what the original tokenwise FFN-only block computes when sharing LN and FFN weights.
- **Not** training feasibility at 100M/2B, throughput, final perplexity, or superiority.

## Required controls before any paid investigation

1. Run original first-attempt CI and standalone CPU tests on this branch. Failure freezes the failed head; no rerun/amend.
2. If viable, separately preregister baseline and candidate, evaluation dataset, final NLL/tokens-per-second thresholds, fixed token exposures, H100 allocation budget, seeds, durable first-attempt reservations, and full train-horizon optimizer schedule. No seed is assigned by this prototype.
3. Prefer a genuinely predictive low-budget engineering control whose learning-rate trajectory **matches the prefix of a later longer run**, unlike the previous engineering-versus-scientific mismatch.
4. Require explicit user authorization for any paid H100 or scientific seed. Do not infer authorization from "continue". No replication/250M/5B/breakthrough authority.

Source of prior results: [scientific issue #1300](https://github.com/vinceackermann2-sys/tam-research/issues/1300) and [read-only verification #1331](https://github.com/vinceackermann2-sys/tam-research/issues/1331). Frozen postmortem is research/reduced_attention/attention8_pair1_successor_postmortem_v1.md.
