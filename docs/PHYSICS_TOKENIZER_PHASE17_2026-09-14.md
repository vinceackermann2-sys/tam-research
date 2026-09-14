# Phase 17 — unnamed local stencil

Status: **PASS (45/45 registered cells)**

Phase 17 removed the named differential operators introduced in Phase 16. The representation, context length, ridge, training families, and evaluation families were kept unchanged.

The identifier was given only a generic periodic 3×3 neighborhood for each state channel plus the same degree-2/3 center polynomial terms. It was not told which combinations correspond to Dx, Dy, Laplacian, or any other physical operator. The resulting library had 26 features.

The 3×3 basis can span the first-order centered derivatives and Laplacian exactly, but the regression must recover those combinations from context.

Three fresh registered replicates each evaluated the same 15 tests used in Phase 16: three wave tests, four Gray-Scott tests, four NLS tests, and four advection tests. NLS and advection remained absent from representation fitting.

All three replicates passed all fifteen tests: **45/45 below raw-state persistence**.

Median error/persistence ratios:

- wave ID 0.00515; spectral 0.000684; combined 0.0556
- Gray ID 0.186; sharp 0.0263; parameter 0.0111; combined 0.0276
- NLS ID 0.562; spectral 0.331; parameter 0.164; combined 0.0794
- advection ID 0.153; spectral 0.0985; speed 0.0391; combined 0.0107

Worst registered cell: **0.5766× persistence**.

## Interpretation

For these four synthetic PDE families, manually naming derivative operators is not necessary. A generic local spatial hypothesis space is sufficient for context-only identification to recover useful dynamics from the frozen discrete scientific representation.

This is a stronger result than Phase 16 because the model is no longer handed an explicit differential-operator vocabulary. However, locality itself, the 3×3 radius, periodic topology, center polynomial order, ridge form, and simulator families are still hand-specified.

The next useful test is to compress or learn the spatial basis from training trajectories rather than expose all 18 raw neighborhood values directly, while keeping NLS and advection unseen during representation/basis training.
