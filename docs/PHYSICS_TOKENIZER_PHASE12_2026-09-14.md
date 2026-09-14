# Physics Tokenizer Phase 12 — unseen cubic NLS transfer

Status: **FAIL** under the preregistered all-cells gate.

Phase 12 froze the Phase-10/11 representation and identifier, trained the representation only on the existing wave + Gray–Scott mixture, and evaluated a third, completely unseen two-channel cubic nonlinear Schrödinger / Gross–Pitaevskii-like synthetic PDE family. No NLS frames were used to fit normalization statistics, RVQ codebooks, compander thresholds, feature choices, or ridge strength.

The exact discrete NLS update lies inside the already-frozen 16-feature family; the representability diagnostic is ~1.16e-16 MSE. Three registered replicates were run exactly once with no retries or replacement seeds.

## Registered result

Companded rollout MSE / raw persistence:

- NLS ID h8: 2.3678, 1.2533, 1.4727 — **0/3 pass**, median 1.4727.
- NLS spectral OOD h3: 1.5678, 0.6397, 0.7441 — **2/3 pass**, median 0.7441.
- NLS parameter OOD h3: 0.4379, 0.4876, 0.4387 — **3/3 pass**, median 0.4387.
- NLS combined parameter+spectral OOD h3: 0.4356, 0.2799, 0.3615 — **3/3 pass**, median 0.3615.

The preregistered gate required all 12 cells to be below 1.0. The result is therefore FAIL (8/12 cells pass).

## Failure decomposition

The result does **not** indicate that the frozen token representation cannot encode NLS states or that the feature library cannot express the NLS law.

For NLS ID h8, the median target-token oracle ratio is ~1.45e-5 versus persistence, and the median rollout ratio when the transition map is fit from raw context but projected through the frozen companded codec is ~0.0165. Thus the representation and recursive projection have very large margin when the dynamics coefficients are estimated correctly.

The failure appears when the generic absolute-next-state map is estimated from only three decoded/quantized context frames. Small coefficient errors accumulate across the h8 rollout. The same short-context estimator is especially fragile for the slow/dispersive ID regime, even though stronger-parameter and combined OOD trajectories provide larger changes and pass reliably.

This sharpens the research target: preserve or infer **small dynamical increments / generator coefficients** robustly from discrete observations, rather than merely minimizing state reconstruction error.

## Claim limit

This is a negative result for one known synthetic unseen PDE family under this exact short-context representation + identifier. It is not evidence for or against universal law discovery, novel physics, AGI, or SSI.
