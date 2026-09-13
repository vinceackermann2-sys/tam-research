# Physics tokenizer Phase 9 — matched-bit causal innovation screen

Date: 2026-09-13

Status: **COMPLETE — SCREEN_FAIL**

Branch: `experiment/physics-tokenizer-phase9-innovation-tokenizer-20260913`

Parent Phase-8 HEAD: `fa24420b1fb0323109903d47bbded6546a7b1680`

## Why this phase was run

Phase 8 isolated a recurring wave combined-OOD failure around system identification from quantized context. The unchanged 16-feature candidate law could predict extremely well from raw context, and it could still beat persistence when started from a quantized state and recursively requantized. In contrast, fitting the same law from decoded-token context was unstable. That suggested a narrow representation hypothesis: independently quantizing absolute state frames may distort the small temporal changes used to infer dynamics.

Phase 9 therefore tested a causal predictive/innovation representation, not a larger black-box dynamics model.

## Frozen design

The protocol was committed before any registered Phase-9 execution at `5e0d1494ece81caebe4152dd02a9cd937eb6ad48`.

Both arms use the same nominal transmitted rate: a 2048-entry coarse code plus a 128-entry residual code, i.e. 18 nominal bits per cell per saved frame. The baseline independently RVQ-quantizes each absolute physical state. The innovation arm sends an absolute state-RVQ anchor at frame 0, then encodes `x_t - xhat_{t-1}` relative to the previous decoder reconstruction using a separate 2048+128 residual codebook. Decoder reconstruction is causal: `xhat_t = xhat_{t-1} + decoded_innovation_t`.

The downstream identifier was not changed: three context frames, the Phase-7 16-dimensional D4/local polynomial feature basis, per-sequence RMS feature scaling, and ridge `0.001`, all in decoded physical coordinates. No law ID, family router, hidden parameter label, governing equation, Laplacian, or known temporal integrator was supplied.

The primary test was the combined wave OOD split: 48 sequences, `c in [1.15, 1.35]`, `max_mode=6`, horizon 3, scored against raw-state persistence. Secondary tests were ID horizon 8 and spectral-only OOD horizon 3.

Three paired fresh engineering replicates were frozen in advance. The GO rule required all three conditions:

1. Innovation must have a strictly lower primary ratio than state RVQ in all three paired replicates.
2. Median `innovation_ratio / state_ratio <= 0.80`.
3. Innovation must beat raw persistence on the primary split in at least two of three replicates.

No registered replicate was retried or replaced. The implementation was sealed before registered execution, and only non-registered smoke seeds were used for implementation checks.

## Primary results

| Replicate | State RVQ / persistence | Innovation RVQ / persistence | Innovation / state | Innovation beats persistence? |
|---|---:|---:|---:|---:|
| 1 | 1.281125 | 1.071555 | 0.836417 | No |
| 2 | 0.540693 | 0.553298 | 1.023313 | Yes |
| 3 | 0.678074 | 0.729863 | 1.076377 | Yes |
| Median | 0.678074 | 0.729863 | 1.023313 | 2/3 |

The innovation arm was better than state RVQ only in replicate 1. The frozen all-three superiority condition therefore failed. Its median paired ratio was `1.023313`, also failing the required `<=0.80` threshold. Although innovation beat persistence in two of three replicates, that third condition alone cannot rescue the screen.

**Frozen decision: `SCREEN_FAIL`.**

## Secondary results

On spectral-only OOD, innovation improved the rollout ratio in all three paired replicates:

- rep 1: state `0.328870`, innovation `0.207288`, innovation/state `0.630303`
- rep 2: state `0.109959`, innovation `0.090734`, innovation/state `0.825163`
- rep 3: state `0.171495`, innovation `0.112022`, innovation/state `0.653207`
- median innovation/state: `0.653207`

ID horizon-8 was mixed: innovation/state was `0.961666`, `1.510788`, and `0.866752` across the three replicates. It therefore did not provide a consistent ID advantage either.

## Failure decomposition inside Phase 9

The combined-OOD diagnostic metrics point away from a universal advantage of causal innovation coding at this fixed global codebook:

- Median context reconstruction MSE was `0.00444648` for state RVQ versus `0.0279915` for innovation RVQ, about **6.30x worse** for innovation.
- Median context-delta MSE was `0.00522228` for state RVQ versus `0.0165501` for innovation RVQ, about **3.17x worse** for innovation.
- Median target-oracle ratio was `0.409811` for state RVQ versus `0.599224` for innovation RVQ, about **1.46x worse** for innovation.
- With the downstream map fitted from raw context, median recursive projection ratio was `0.430907` for state RVQ versus `0.619288` for innovation RVQ.

At the same time, innovation coding helped on the spectral-only shift. A plausible follow-up hypothesis is therefore not that innovation coding is intrinsically bad, but that the globally normalized in-range innovation codebook suffers a stronger magnitude/tail distribution shift when both wave speed and spatial frequency move OOD. This is an interpretation of the observed diagnostics, not a Phase-9 success claim and not proof of the mechanism.

A valid next phase would have to preregister a new representation hypothesis before execution. Examples include a fixed robust companding transform for innovations or another causal scale-adaptive residual coordinate system that does not receive future states, hidden parameters, or extra OOD labels. Phase 9 itself must remain sealed as a negative result; its thresholds, seeds, and predictor are not to be retuned post hoc.

## Execution provenance and claim boundary

Registered Phase-9 work ran on local CPU only. No Modal job, GPU job, or paid scientific workflow was used. The repository's unrelated research was not modified, and `main` was not touched.

This was an engineering representation screen on known synthetic PDEs. The result does not establish novel physics, general scientific-law discovery, AGI, or SSI. It also does not authorize a confirmatory success stage. Any follow-up requires a separate preregistration and fresh seeds.
