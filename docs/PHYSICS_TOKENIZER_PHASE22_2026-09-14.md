# Phase 22 — Fourier leave-one-direction-out spatial basis

## Result

Phase 22 **PASS_WITH_STRUCTURED_RANK8_SUFFICIENCY**.

The full fixed real 3x3 Fourier basis passed the frozen four-family transfer gate in all three registered replicates (45/45 cells). More importantly, four different rank-8 structured subspaces also passed 45/45:

- omit `sin_x`
- omit `sin_y`
- omit `sin_diag_plus`
- omit `sin_diag_minus`

The remaining five omissions failed the frozen gate:

- omit `dc`: 12/45 cells passed; worst ratio 5300.59x persistence
- omit `cos_x`: 19/45; worst 257.55x
- omit `cos_y`: 18/45; worst 291.11x
- omit `cos_diag_plus`: 31/45; worst 153.96x
- omit `cos_diag_minus`: 43/45; worst 1.7469x

The full control's worst registered ratio was 0.8103x persistence. The four sufficient rank-8 omissions had worst registered ratios between 0.6819x and 0.8101x.

## Experimental design

The spatial basis was a fixed, data-independent, orthonormal real discrete Fourier basis of each channel's periodic 3x3 neighborhood:

`dc, cos_x, sin_x, cos_y, sin_y, cos_diag_plus, sin_diag_plus, cos_diag_minus, sin_diag_minus`.

The rank-9 control retained all nine modes. Each rank-8 leave-one-direction-out variant removed one named mode from both state channels. Within a registered replicate, every variant shared exactly the same representation codebooks, trajectories, eight-frame contexts, targets, ridge strength, token projection, and simulator parameters.

The tokenizer remained the frozen 18-bit Phase-10 `asinh` innovation representation trained only on wave and Gray-Scott trajectories. NLS and advection remained excluded from representation fitting. No law-family ID, named derivative, or hidden simulator parameter was supplied to the context-only identifier.

## Registered conclusion

Phase 21 showed that nested *random* rank-8 projections failed the strict 45/45 gate. Phase 22 demonstrates that this is not a universal nine-dimensional lower bound: carefully aligned rank-8 subspaces can retain all tested transfer capability.

The operative variable is therefore not raw dimension alone. **Which spatial direction is discarded matters strongly.** Blind or training-distribution-driven compression can remove a low-dimensional direction that is essential for an unseen operator, while a structured basis can discard a different direction with no registered loss under this suite.

The Fourier labels should not be over-interpreted as a direct map to physical derivative terms. For example, an individually dispensable sine mode does not imply odd/directional information is globally unnecessary; the retained modes can span related structures jointly. Phase 22 only establishes sufficiency/criticality for these specific leave-one-mode-out subspaces under this frozen benchmark.

## Next falsification

The next test should reduce the basis much more aggressively using only grid geometry and symmetry, not physics-law training data. A compact candidate per channel is:

1. center value
2. axial-even sum: N + S + E + W
3. x-odd contrast: E - W
4. y-odd contrast: S - N
5. diagonal-even sum: NE + NW + SE + SW

With the constant and seven center polynomial terms this returns to an 18-feature dynamics library, matching the compact dimensionality of Phase 16 but without supplying named derivatives. A paired rank-4 control that removes the diagonal-even term can test whether that fifth geometry direction is required.

## Claim boundary

These are known synthetic systems under a fixed periodic 3x3 neighborhood, tokenizer, context length, polynomial center library, regression method, and pass criterion. This is evidence about representation geometry and system identification, not a universal minimal-basis theorem, unrestricted law discovery, or discovery of new physics.
