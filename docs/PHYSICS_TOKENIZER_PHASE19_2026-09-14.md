# Physics Tokenizer Phase 19 — Transition-predictive stencil basis

**Status: FAIL**

Phase 19 tested whether Phase 18 failed merely because PCA optimized the wrong objective. Instead of preserving high patch variance, the new 5-component/channel basis was learned to preserve spatial directions predictive of the next physical change on the seen wave + Gray–Scott training trajectories.

## Frozen design

- Same 18-bit companded state/change representation.
- Same 8 context frames / 7 observed transitions.
- Same 18-feature budget: constant + five learned 3×3 patch projections per channel + seven center polynomial terms.
- Basis fitted from wave + Gray–Scott only.
- NLS and advection excluded from both representation and basis fitting.
- Predictive basis solved a generalized eigenspace problem using patch-to-next-change cross covariance, normalized by input patch covariance.
- No named derivatives, law-family ID, raw stencil injection, or hidden simulator parameters.
- PASS required all 45 registered split-replicate cells to beat raw-state persistence.

## Registered result

Phase 19 failed in all three registered replicates. Only **19/45** cells beat persistence.

Median error / persistence ratios:

- wave ID h8: **4.317×**
- wave spectral OOD h3: **0.782×**
- wave combined OOD h3: **0.136×**
- Gray ID h8: **0.176×**
- Gray sharp OOD h3: **0.0711×**
- Gray parameter OOD h3: **0.0416×**
- Gray combined OOD h3: **0.125×**
- NLS ID h8: **534.02×**
- NLS spectral OOD h3: **34.90×**
- NLS parameter OOD h3: **31.09×**
- NLS combined OOD h3: **13.62×**
- advection ID h8: **54.16×**
- advection spectral OOD h3: **5.371×**
- advection speed OOD h3: **3.773×**
- advection combined OOD h3: **1.180×**

Worst registered cell: **578.57× persistence**.

## Interpretation

Phase 18 showed that high state-variance retention is insufficient. Phase 19 shows that optimizing the compressed subspace for transition predictiveness on the *seen* laws is also insufficient.

The consistent contrast is the Phase-17 full raw 3×3 stencil control, which remains highly accurate on the unseen NLS and advection systems. At five of nine spatial dimensions per channel, the learned projection has a four-dimensional nullspace. Operator directions that are absent or weakly useful in the wave+Gray training laws can fall into that nullspace and become unrecoverable when a genuinely different law is encountered.

This supports a more structural design rule for scientific representations:

> If the intended downstream task includes discovering previously unseen local operators, do not compress away locally possible operator directions solely according to statistics of previously observed laws.

A useful next falsification is a data-independent random orthogonal 5D/channel projection. If that also fails under the same four-family gate, the evidence shifts from “the learned objective chose the wrong subspace” toward “the rank itself is insufficient for robust operator-complete transfer.”

## Claim boundary

These are synthetic known-law systems. This does not establish a theorem for all scientific data, unrestricted law discovery, or novel physics.