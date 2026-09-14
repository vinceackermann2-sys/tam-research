# Physics Tokenizer Phase 18 — Learned stencil basis

**Status: FAIL**

Phase 18 tested whether the successful raw 3×3 local stencil from Phase 17 could be compressed into a basis learned only from the seen wave + Gray–Scott training trajectories.

## Frozen design

- Frozen 18-bit companded state/change representation.
- 8 observed context frames / 7 observed transitions.
- No law-family ID or hidden simulator parameter.
- Representation and spatial basis fitted from wave + Gray–Scott only.
- NLS and advection remained unseen during representation/basis fitting.
- Per input channel, a 9-dimensional 3×3 patch was centered and reduced to the top 5 PCA components.
- Identifier features: constant + 5 learned spatial components per channel + the same seven center polynomial monomials = 18 total features.
- PASS required all 15 splits in all 3 fresh registered replicates to beat raw-state persistence (45/45).

## Registered result

Phase 18 failed in all three registered replicates. Only 25/45 split-replicate cells beat persistence.

Median error / persistence ratios:

- wave ID h8: **1.684×**
- wave spectral OOD h3: **0.993×**
- wave combined OOD h3: **0.158×**
- Gray ID h8: **0.120×**
- Gray sharp OOD h3: **0.0439×**
- Gray parameter OOD h3: **0.0135×**
- Gray combined OOD h3: **0.0505×**
- NLS ID h8: **26.913×**
- NLS spectral OOD h3: **14.845×**
- NLS parameter OOD h3: **1.300×**
- NLS combined OOD h3: **5.627×**
- advection ID h8: **0.717×**
- advection spectral OOD h3: **4.379×**
- advection speed OOD h3: **0.119×**
- advection combined OOD h3: **0.533×**

Worst registered cell: **164.94× persistence**.

## Important finding

The learned bases retained about 99% of patch variance in both channels in every replicate, but still lost spatial directions that were crucial for dynamics on unseen families. In contrast, the Phase-17 raw 3×3 stencil control remained near-exact on NLS/advection in the same evaluations.

This separates two objectives that are often conflated:

> high state-reconstruction / variance retention does not imply preservation of the sufficient statistics needed to infer dynamics.

The next useful learned-compression objective should therefore be transition- or dynamics-aware rather than reconstruction/PCA-driven.

## Claim boundary

These are synthetic known-law systems. This result does not demonstrate unrestricted operator discovery or novel physics. It identifies a failure mode of unsupervised variance compression in this controlled system-identification benchmark.