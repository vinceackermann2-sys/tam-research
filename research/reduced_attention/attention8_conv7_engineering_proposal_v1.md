# Conv7 attention-8 successor: draft engineering screen

**Stage: ZERO-GPU DESIGN PROPOSAL; not preregistered, not authorized.** No seed, paid H100 job, retries, replication, resumed work, or scientific claim is created by this PR. Seed \`60232\` from the original attention-8 experiment is permanently consumed.

## Why this proposal exists

The frozen [attention-8 scientific screen](https://github.com/vinceackermann2-sys/tam-research/issues/1300) found that eight full-attention layers were 7.50% faster to train than 24 full-attention layers, but validation NLL was +0.045087 worse at 2B exposures. The new CPU-validated causal Conv7 variant adds a lightweight local mixer to 16 formerly tokenwise FFN-only blocks without materially increasing parameter count. It has never been trained.

The historical 200M engineering comparison must not be used as a substitute for a full-horizon matched test: it ended its cosine learning-rate schedule at 200M, while the 2B scientific run used a 2B-horizon schedule. The engineering seed was also different.

## Proposed three-way pilot (subject to separate authorization)

| Model | Parameters | Role |
|---|---:|---|
| Fresh 24-attention Transformer | 101,803,520 | Quality and throughput baseline |
| Frozen eight-attention v2 | 101,795,328 | Parent architecture / causal-mixer ablation |
| Eight-attention + depthwise Conv7 | 101,803,536 | New candidate |

- Shared dataset fingerprints from the original experiment, context 512, gradient accumulation 2, batch 64, 65,536 token exposures per optimizer step.
- Stop each model **after 3,052 optimizer steps** (~200,015,872 token exposures).
- **Crucial difference from the old engineering panel:** use the *first 3,052 steps of a 30,518-step full-horizon cosine schedule*, not a new 3,052-step cosine schedule. The full-horizon warmup is 610 steps (floor of 30,518 × 0.02) rather than the old short-horizon warmup. Identical scheduling across all three models.
- Fresh initialization, independent training state per model but identical prescribed seed and deterministic data ordering; evaluate on the same held-out batch stream (50 batches × 32 sequences × 512 tokens); frozen optimizer/precision/compiler controls and H100 class to be pinned in the later execution protocol.
- The **single engineering seed is currently null** and the **hard spend cap is currently null**: absence of either must be a blocking preflight error for any future runner. The preferred workspace is primary only, without fallback.
- Model-specific duration/throughput comparisons will not be inferred from CPU tests, and any H100 benchmark requires a separate exact launch authorization.

## Draft go/no-go rules for engineering usefulness only

A future candidate would need *all* of:

1. Candidate − Transformer final NLL ≤ +0.015.
2. Candidate improves final NLL versus the old eight-attention parent by ≥ 0.010.
3. Candidate ÷ Transformer training tokens/sec ≥ 1.03.
4. All three models complete with finite metrics, identical corpus hashes, optimizer steps, token exposures, and held-out evaluation batches.

These are **draft** selection criteria to be approved and frozen before running anything paid. An engineering pass would *not* establish a breakthrough or permit scientific Pair-1, replication, or 250M/5B progression. A separate scientific design and explicit authorization would remain necessary.

## Research interpretation

- If Conv7 improves the parent but remains above Transformer quality, local mixing may recover only part of the global-information deficit.
- If Conv7 restores Transformer-like quality but loses the throughput advantage, it might not improve the efficiency-quality tradeoff.
- If Conv7 fails to improve its parent, one possible cause is that seven-token local mixing is too weak; but a single engineering seed cannot identify the causal mechanism conclusively.
- At the 200M prefix, no success is guaranteed to persist to 2B. Prefer predeclared longer-horizon gates and replication only under separate authorization.

## Safety/governance

This PR adds only a draft JSON protocol, this explanation, and static CPU consistency tests. It contains no Modal runner and has **zero GPU authority**. Do not treat \`continue\` as permission for paid scientific attempts. No previously consumed scientific seeds may be reused. Keep the completed experiment and its volume records immutable.
