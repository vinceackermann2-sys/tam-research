# Gated-Scan RLT, 15M: post-AM stage-gate decision

Status: **STOP current compute-matched gated-scan configuration; no breakthrough or scaling authority.**

Scope: clean-room `exp/rlt-colab` research branch. Decision is evidence-bound to
15,129,344 parameters, seq=64, batch=64, pinned FineWeb-Edu D and the training
budgets stated below. No modification of `main` is authorized here.

## Frozen evidence

| Run | Protocol | Light-state RLT | Gated-Scan RLT | Transformer |
| --- | --- | ---: | ---: | ---: |
| AG | engineering, 60s, batch 16/32/48/64 | batch 64 wins sub-64 quality/sec sweep | — | — |
| AI | engineering, synthetic data, same GPU, 12s | 227,756 tokens/sec (full) | — | 340,786 tokens/sec |
| AJ | engineering, synthetic data, same GPU, 12s | 185,036 tokens/sec | 312,366 tokens/sec | 323,126 tokens/sec |
| AK | engineering, 60s post-compile each, pinned D | NLL **5.730703**, 246,908 tokens/sec | NLL **5.672780**, 304,254 tokens/sec | NLL **5.531180**, 330,164 tokens/sec |
| AL | engineering, six optimizer schedules, pinned D, 60s each | — | best constant LR `1e-3`, NLL **5.690430** within AL | — |
| AM | fresh-seed *scientific* equal-time, pinned D, 60s each | — | NLL **5.736319**, 275,255 tokens/sec | NLL **5.586229**, 313,392 tokens/sec |

Ground truth result blobs (do not edit/retry consumed jobs):

- `experiments/rlt/modal/results/rlt-systems-lightstate-lowbatch-qualitysec-modal-20261007-ag.json`
- `experiments/rlt/modal/results/rlt-systems-lightstate-component-attribution-modal-20261008-ai.json`
- `experiments/rlt/modal/results/rlt-systems-gated-scan-throughput-modal-20261008-aj.json`
- `experiments/rlt/modal/results/rlt-systems-gated-scan-realdata-modal-20261008-ak.json`
- `experiments/rlt/modal/results/rlt-systems-gated-scan-optimizer-qualitysec-modal-20261008-al.json`
- `experiments/rlt/modal/results/rlt-gated-scan-compute-matched-scientific-scale15m-60s-modal-20261008-am.json`

## Conclusions within measured scope

1. The original light-state model's major systems penalty came from the serial
   recurrent state update (AI removing it sped its full path up ~1.42x).
2. The new associative affine gated-state scan passes exact serial math/gradient
   agreement, causal tests, parameter-equality tests, and CPU compilation checks;
   its parallel implementation is *not* semantically identical to the original
   light-state recurrence. The independent AJ synthetic benchmark increased
   training throughput by **1.688x** versus light-state, to **96.67%** of Transformer.
3. On the actual pinned language data, AK improves over original light-state
   by **0.057924 NLL** yet still trails Transformer by **0.141600 NLL** at 60s.
   AK is engineering evidence only, not a scientific replicate.
4. AL optimizer calibration confirmed the frozen constant `1e-3` as best among
   six specified candidates at batch 64. The six AL NLL values must not be
   directly compared with different-seed AK or AM losses.
5. AM, with a fresh scientific seed and frozen independently chosen optimizers,
   showed Gated-Scan RLT *worse* than Transformer by **+0.150090 NLL**. Both
   models had exactly 15,129,344 parameters, verified pinned data hashes,
   and 60 seconds of postcompile training each. AM throughput ratio was
   **0.87831** Gated-Scan/Transformer.
6. **No breakthrough:** The two time-matched comparisons (AK engineering and
   AM scientific) are negative on language NLL. Optimizer, sub-64 batch size
   and serial recurrence efficiency have been adequately screened to
   disfavor further same-family microtuning at this scale.

## Decision and next architecture objective

Freeze `GatedScanLightStateRLT` as a negative language-model baseline with a
useful systems discovery. Do **not** run 250M/5B scale, replicate old consumed
scientific attempts or spend more on the six tested optimizer settings.

The next candidate must address **representation/learning quality**, not simply
faster recurrence. Candidate hypothesis for *zero-GPU design and tests*:

- Preserve a direct causal encoder-to-decoder residual information path in
  parallel with the scan-based memory pathway, rather than forcing all decoder
  initialization through the gated recurrent state alone.
- Remain exactly parameter-matched to Transformer for any fair paid comparison,
  and prove causal forward/backward correctness and active parameter gradients.
- Use a preregistered quality-per-wall-time screen on pinned D with fresh,
  independently reserved seeds and frozen optimizer selection; a single
  positive seed may justify a separately authorized replication, never a
  breakthrough claim.
- Test whether recurrence actually helps on explicit longer-horizon or
  state-tracking tasks at matched context and compute, rather than assuming
  standard short-context NLL measures all useful capabilities.

These hypotheses are *not results*. No new architecture training, replication,
scaling, or unbounded search is authorized by this decision memo.

## Governance

Treat consumed scientific/engineering job claims as immutable. Use new job IDs
and unused seeds; durable reservation before GPU, retries=0, no resume.
Re-audit live `main` and `exp/rlt-colab` before every mutation. All writes
are restricted to `exp/rlt-colab`. Data hashes and negative findings must
remain visible; never relabel infrastructure failures as scientific evidence.
