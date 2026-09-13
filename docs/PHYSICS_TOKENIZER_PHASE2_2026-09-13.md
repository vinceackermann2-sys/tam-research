# Physics Tokens Phase 2 — Structured System Identification (2026-09-13)

## Status

**PASS under the preregistered confirmatory gate.**

Phase 1 showed that the unconstrained learned token transition failed unseen spatial frequencies even though the frozen 256-token codec retained sufficient information. Phase 2 kept that codec and the evaluation task fixed, but replaced the unconstrained transition with a structured local wave-operator family. The per-sequence propagation speed was inferred only from the three quantized context frames by minimizing context-transition error; the true speed label was not supplied. Every predicted state was requantized before the next rollout step.

## Confirmatory results

| Replicate | ID h8 / persistence | Spectral OOD h3 | Combined OOD h3 | Pass |
|---|---:|---:|---:|---|
| 1 | 0.4357 | 0.3136 | 0.7802 | yes |
| 2 | 0.4178 | 0.2950 | 0.8860 | yes |
| 3 | 0.4220 | 0.3434 | 0.7956 | yes |
| **Mean** | **0.4252** | **0.3174** | **0.8206** | **3/3** |

The frozen gate required every replicate to beat raw-state persistence on all three primary tests. All three passed.

## What this changes

The Phase-1 failure was not evidence that discrete state tokens are inherently unable to generalize. The same frozen token representation supports spectral and wave-speed OOD prediction when the transition architecture encodes a local operator family and identifies a latent sequence parameter from context.

That is a meaningful architectural result: **state discretization can survive OOD; unconstrained transition learning was the bottleneck.**

## What this does not show

The wave-equation operator family was supplied. Therefore this is not discovery of a new physical law. It is a system-identification control showing that the discrete representation contains enough information and that operator-like structure can recover extrapolation. The next meaningful step is to remove knowledge of the exact operator and ask whether a constrained model can *learn the operator itself* from low-frequency training data, then extrapolate to unseen frequencies and speeds.
