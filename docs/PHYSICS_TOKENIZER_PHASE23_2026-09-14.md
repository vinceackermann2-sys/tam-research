# Phase 23 — Compact geometry/symmetry basis

## Result

**CONTROL_FAIL.** The preregistered rank-5 control did not satisfy the required 45/45 gate, so no rank-4 sufficiency claim is made.

- rank 5: **44 / 45** registered cells below raw-state persistence
- rank 4: **43 / 45** registered cells below raw-state persistence

The sole rank-5 failure was replicate 2 NLS ID-h8 at **1.1748x persistence**. Rank 4 additionally failed replicate 2 NLS ID-h8 at **1.4251x** and replicate 3 Gray ID-h8 at **1.1890x**.

## Basis

The basis was fixed and data-independent. Per channel:

- rank 4: center, axial-even neighbor sum, x-odd difference, y-odd difference
- rank 5: the same four plus diagonal-even neighbor sum

The identifier, context length, ridge, tokenizer, compander, training families, and recursive token projection were unchanged from the earlier unified experiments. NLS and advection remained excluded from representation training.

## Interpretation

The compact geometry basis is close to sufficient but not robust enough under the frozen criterion. The failures are concentrated in long-horizon ID behavior rather than broad OOD collapse, which differs from the severe random/PCA low-rank failures in Phases 18–21.

Phase 22 remains the stronger compression result: several structured rank-8 Fourier subspaces passed the full 45/45 transfer gate. Phase 23 shows that aggressively collapsing those spatial directions into a much smaller hand-aggregated geometry basis loses robustness.

The next useful experiment is therefore to stay inside the fixed Fourier representation and test combinations of the four individually dispensable sine modes. That can determine whether a structured rank-7, rank-6, or rank-5 Fourier subspace remains sufficient without replacing the basis by manually aggregated physical features.

## Claim boundary

These are known synthetic systems. The result is about robustness of a fixed local representation and context-only identifier under this benchmark; it is not discovery of new physics or a universal minimal operator basis.
