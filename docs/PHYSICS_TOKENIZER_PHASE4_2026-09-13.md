# Physics Tokens Phase 4 — Residual-VQ Local State Space (2026-09-13)

## Status

**PASS under the preregistered three-replicate confirmation gate.**

Phase 4 removes the supplied wave-equation spatial operator and the supplied velocity-Verlet temporal integrator. The state is represented discretely with a two-stage residual vector quantizer: 256 coarse codes plus 32 residual codes per spatial cell (two tokens, nominally 13 bits/cell). The codec is trained only on the original low-frequency wave training distribution.

For each fresh replicate, the transition law is learned from new quantized low-frequency trajectories as a generic local one-step state-space family. Each sequence first gets a 2×18 local linear map from the full 3×3 neighborhood of both state channels. PCA over those sequence maps yields a mean map plus 12 learned latent directions. At inference the 12 latent coefficients are inferred only from the three quantized context frames. Rollout is fully recursive and is requantized through both VQ stages after every step.

No true wave-speed labels, true Laplacian coefficients, or known numerical integrator are used.

## Representation

- coarse codebook: 256 entries
- residual codebook: 32 entries
- tokens per cell: 2
- nominal discrete capacity: 13 bits/cell
- training reconstruction NMSE: **0.000837**

## Confirmatory results

| Replicate | ID h8 / persistence | Spectral OOD h3 | Combined OOD h3 | Pass |
|---|---:|---:|---:|---|
| 1 | 0.0693 | 0.1714 | 0.7761 | yes |
| 2 | 0.0653 | 0.1393 | 0.8967 | yes |
| 3 | 0.0982 | 0.2821 | 0.9391 | yes |
| **Mean** | **0.0776** | **0.1976** | **0.8706** | **3/3** |

The frozen pass rule required every independently trained transition family to beat raw-state persistence on all three primary tests. All three passed.

## Interpretation

This resolves the Phase-4 exploratory failure with the single 256-code state: the generic learned spatiotemporal law could beat persistence before requantization but lost enough information when repeatedly projected back into one coarse token. A very small residual codebook (32 entries) removes that bottleneck while staying fully discrete.

The strongest supported statement so far is therefore: **a hierarchical discrete scientific-state representation can support recursive, out-of-distribution local dynamics learned directly from trajectories, without being given the known wave operator, the latent wave speed, or the numerical time integrator.**

This remains a controlled synthetic result, not evidence of discovering new fundamental physics. The architecture still assumes local 3×3 interactions, locally linear maps, and a low-rank latent family. The next critical test is whether the same representation/learning recipe transfers to a qualitatively different dynamical law rather than only the wave equation.
