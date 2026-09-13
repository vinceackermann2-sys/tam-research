# Physics Tokens Phase 5 — Gray-Scott Transfer, Corrected Confirmation (2026-09-13)

## Status

**PASS under the corrected preregistered three-replicate gate.**

This result supersedes the scientific interpretation of the original Phase-5 confirmation. The first confirmation run accidentally inherited Phase-4 KMeans initialization seeds `74193/74210` rather than the Phase-5-preregistered codec seeds `75193/75210`. That first run is preserved as protocol-invalid and its replicate seeds were not reused. Repair 1 used the correct codec seeds but became infrastructure-incomplete when the local execution ceiling terminated replicate 3; all of those seeds were also retired. Repair 2 was preregistered with fresh seeds and identical scientific settings, with each replicate executed once in its own process and durably persisted.

## Frozen model

- system: Gray-Scott reaction-diffusion, used only to generate trajectories
- model receives no Gray-Scott equations, feed/kill labels, reaction terms, or diffusion coefficients
- state representation: two-stage residual VQ, 256 coarse + 32 residual codes per spatial cell
- codec KMeans seeds: coarse `75193`, residual `75210`
- local features: 3x3 neighborhood of both channels plus generic center monomials through cubic order (25 features total)
- per-sequence transition map: ridge linear local map
- family model: 8-dimensional PCA basis over per-sequence maps
- context: 3 quantized frames; latent coefficients inferred from the two observed transitions
- rollout: recursive and requantized through both VQ stages after every step

## Corrected confirmatory result

| Replicate | ID h8 | Sharp-init OOD h3 | Parameter OOD h3 | Combined OOD h3 | Pass |
|---|---:|---:|---:|---:|---|
| 1 | 0.2503 | 0.0893 | 0.0585 | 0.1215 | yes |
| 2 | 0.4325 | 0.0694 | 0.0569 | 0.0764 | yes |
| 3 | 0.5873 | 0.0696 | 0.0634 | 0.1079 | yes |
| **Mean** | **0.4234** | **0.0761** | **0.0596** | **0.1019** | **3/3** |

All values are rollout MSE divided by raw-state persistence MSE. The frozen gate required every replicate to remain below 1.0 on all four tests. All 12 primary comparisons pass.

Mean residual-VQ reconstruction NMSE is **0.0001621**. The top eight transition-family components explain **99.284%** of per-sequence map variance.

## Interpretation

The corrected result supports the same qualitative conclusion as the original exploratory result, now under the exact preregistered codec configuration: the hierarchical discrete-state + context-inferred local dynamics recipe transfers from linear wave propagation to a qualitatively different nonlinear reaction-diffusion law and extrapolates to sharper initial conditions, unseen Gray-Scott parameter ranges, and both shifts simultaneously.

The strongest supported statement is: **a hierarchical discrete scientific-state representation can support recursively learned dynamics across at least two qualitatively different synthetic PDE families without receiving the governing equations or hidden system parameters.**

This is still not unrestricted scientific-law discovery. The architecture is given important priors: local 3x3 interactions, polynomial features through degree 3, linear local readout, and a low-rank family. The next critical test is therefore to train a single shared model across multiple law families and require it to infer the active dynamics from context without an explicit law identifier or family-specific feature selection.
