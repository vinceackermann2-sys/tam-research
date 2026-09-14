# Phase 16 — generic parity-complete operators

Status: **PASS (45/45 registered cells)**

Phase 16 tested whether the Phase-15 advection failure was caused by a missing operator rather than a failed discrete representation. The Phase-10 tail-aware 18-bit companded representation, context length 8, ridge 1e-3, and context-only fitting were held fixed. No law-family router was added.

The old 16-feature symmetric library was replaced by one generic 18-feature library: constant; for each channel identity, centered Dx, centered Dy, Laplacian, and bi-Laplacian; plus the same generic degree-2/3 center polynomial terms. This was fixed before registered execution.

The representation codebooks were still fit only on wave + smooth Gray-Scott trajectories. NLS and advection were unseen during representation fitting.

Three fresh registered replicates each evaluated 15 tests: wave (ID, spectral OOD, combined OOD), Gray-Scott (ID, sharp OOD, parameter OOD, combined OOD), NLS (ID, spectral OOD, parameter OOD, combined OOD), and advection (ID, spectral OOD, speed OOD, combined OOD).

All three replicates passed all fifteen tests, for **45/45 below raw-state persistence**. Median error/persistence ratios were:

- wave ID 0.00383; spectral 0.00312; combined 0.0653
- Gray ID 0.130; sharp 0.0211; parameter 0.0112; combined 0.0279
- NLS ID 0.521; spectral 0.413; parameter 0.259; combined 0.124
- advection ID 0.211; spectral 0.163; speed 0.0643; combined 0.0223

Worst registered cell: **0.5511× persistence**.

## Interpretation

Phase 15 showed that a symmetric hypothesis class could be out-of-class for directional transport. Phase 16 repairs that boundary with a general parity-complete differential vocabulary rather than an advection-specific branch. The same context-only identifier and representation now handle four qualitatively different synthetic PDE families.

A useful secondary observation is that some unprojected continuous 18-feature rollouts diverged while the recursively requantized token rollout remained stable. In this setup, projection into the discrete state representation can act as a stabilizing state constraint as well as a compression step.

## Limits

This remains a synthetic known-law validation. The operator vocabulary itself is still hand-designed. Phase 16 therefore does not demonstrate unrestricted equation discovery or novel physics. The next falsifiable step is to replace named operators with a generic local stencil basis and test whether the needed operators can be recovered from context without being specified as Dx, Dy, Laplacian, etc.
