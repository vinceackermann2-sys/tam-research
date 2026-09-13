# Physics Tokens Phase 5 — Gray-Scott Transfer (2026-09-13)

## Status

**PASS under the preregistered three-replicate transfer gate.**

Phase 5 asks whether the hierarchical discrete-state + learned local low-rank dynamics recipe transfers beyond the wave equation to a qualitatively different nonlinear PDE: Gray-Scott reaction-diffusion.

The learned model is not given the Gray-Scott equations, F/k labels, reaction terms, or diffusion coefficients. It receives quantized trajectories only. The supplied architectural priors are 3×3 locality, generic center monomials through degree 3, a local linear readout, and an 8-dimensional low-rank family inferred from three context frames.

Each confirmatory replicate independently regenerates its training data, fits normalization and a 256+32 residual-VQ state representation, learns the transition family, and evaluates on fresh ID/OOD sets.

## Primary results

| Replicate | ID h8 | Sharp-init OOD h3 | Parameter OOD h3 | Combined OOD h3 | Pass |
|---|---:|---:|---:|---:|---|
| 1 | 0.3898 | 0.1118 | 0.1241 | 0.0875 | yes |
| 2 | 0.3345 | 0.0933 | 0.0915 | 0.1752 | yes |
| 3 | 0.5132 | 0.0872 | 0.0568 | 0.1041 | yes |
| **Mean** | **0.4125** | **0.0974** | **0.0908** | **0.1223** | **3/3** |

Values are MSE divided by raw-state persistence MSE; lower than 1 is a pass. Every fresh replicate passes every frozen gate.

## Representation and family diagnostics

Mean residual-VQ reconstruction NMSE is **0.000153**. The top eight learned transition-family components explain **99.33%** of the variation among per-sequence local maps.

A purely linear 18-feature local model was tested during exploration before preregistration. It failed the long ID rollout despite doing well on short OOD horizons. Adding only a generic polynomial basis through cubic order fixed that nonlinear-rollout limitation. The preregistered confirmatory model uses the smallest exploratory configuration that passed: eight latent components.

## Scientific interpretation

This is the first cross-law result in this research line. The successful recipe is not specific to wave propagation: it also models a nonlinear reaction-diffusion process and extrapolates to sharper initial conditions, unseen Gray-Scott parameter ranges, and both shifts simultaneously.

The strongest supported statement is now: **hierarchical discrete scientific-state representations can support learned local dynamics across at least two qualitatively different synthetic PDE families, including nonlinear dynamics and parameter extrapolation.**

This still does not demonstrate unrestricted scientific-law discovery. Locality and a generic degree-3 polynomial feature family are supplied. The next critical experiment should combine both law families under one shared representation/model and require the system to infer which dynamics apply from context, without an explicit law identifier.
