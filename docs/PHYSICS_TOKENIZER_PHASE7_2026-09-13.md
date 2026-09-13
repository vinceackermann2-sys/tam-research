# Physics Tokens Phase 7 — Physical-Coordinate Shared System Identification (2026-09-13)

## Status

**FAIL under the preregistered 3/3 gate.**

Phase 7 kept one shared discrete representation for both wave and Gray-Scott systems, but identified the transition law in decoded physical coordinates using a generic D4-symmetric radius-2 spatial library plus center polynomials through degree 3. It received no law ID, family router, governing equations, or hidden parameter labels.

## Primary results

| Replicate | Wave ID h8 | Wave spectral h3 | Wave combined h3 | Gray ID h8 | Gray sharp h3 | Gray parameter h3 | Gray combined h3 | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.2766 | 0.1817 | 1.5015 | 0.0974 | 0.1597 | 0.0489 | 0.3823 | no |
| 2 | 0.2202 | 0.1453 | 0.9384 | 0.3947 | 0.1554 | 0.0618 | 0.1140 | yes |
| 3 | 0.2234 | 0.2021 | 1.1895 | 0.7598 | 0.1042 | 0.1302 | 0.1216 | no |

The frozen gate required every replicate to beat raw persistence on all seven tests. Replicates 1 and 3 fail only the combined faster-wave + unseen-spectrum condition at 1.5015× and 1.1895× persistence; replicate 2 narrowly passes it at 0.9384×. Every Gray-Scott comparison passes.

## Interpretation

The repeated combined-wave failure across Phases 6 and 7 shows that the remaining limitation is not solved by a larger shared codebook, physical-coordinate fitting, or symmetry-aware features alone. The next diagnostic must separate measurement/quantization noise in the observed context from inadequacy of the local law family.

This remains synthetic systems research and is not evidence of new fundamental physics.
