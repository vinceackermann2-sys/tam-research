# Phase 26 — Burgers interaction-class confirmation

## Result

**PASS.** The generic 62-feature center-times-spatial interaction library passed the preregistered Burgers transfer gate on all three fresh registered replicates: **12 / 12 cells below raw-state persistence**.

Median token-62 rollout ratios versus persistence:

- Burgers ID-h8: **0.1866x**
- spectral OOD-h3: **0.1223x**
- viscosity/parameter OOD-h3: **0.1432x**
- combined spectral + parameter + amplitude OOD-h3: **0.5519x**

The worst registered token-62 cell was still below persistence at **0.6208x**.

The frozen 26-feature control from Phase 25 passed only **6 / 12** cells. Its median Burgers ID-h8 ratio was **6.8871x** persistence and its median combined-OOD ratio was **1.1845x**.

## What changed from Phase 25

Nothing about the tokenizer, compander, context length, representation-training families, spatial support, ridge value, or evaluation distributions changed. The only promoted hypothesis-class change was the feature library:

- base: constant + 18 full real 3x3 Fourier spatial coordinates + seven center degree-2/3 monomials
- added: every product between center `u` or `v` and each of the 18 spatial coordinates, adding 36 generic interaction terms
- total feature dimension: **62**

No Burgers data entered representation fitting. The tokenizer remained fitted only from wave + Gray-Scott trajectories.

## Interpretation

Phase 25 showed that the older linear-spatial-plus-center-polynomial class fails on a nonlinear-flux PDE even though the token representation still retains enough information for a richer interaction model. Phase 26 independently confirms that diagnosis on fresh seeds.

The important result is therefore not simply that a larger model performs better. It is that **multiplicative state × local-spatial interactions are an essential hypothesis-class component for this benchmark's Burgers family**, and those interactions can be estimated from the same quantized context representation without law labels or hidden viscosity parameters.

This does **not** yet establish the 62-feature library as a unified replacement for the earlier four-law identifier. A larger feature class can overfit, destabilize, or regress simpler families. The next required test is a separately preregistered five-family regression phase using the unchanged 62-feature primary library across wave, Gray-Scott, NLS, advection, and Burgers.

## Claim boundary

All systems are known synthetic dynamics. This is evidence about a discrete scientific representation and a context-only local hypothesis class under the fixed benchmark. It is not discovery of new physics, unrestricted scientific-law discovery, AGI, or a universal operator library.
