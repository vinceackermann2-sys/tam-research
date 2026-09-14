# Phase 24 — Exhaustive Fourier sine-subset lattice

## Result

**PASS.** Within the preregistered structured Fourier subset family, the smallest tested spatial rank that remained sufficient across all registered runs was **rank 6 per channel**.

- rank 9: 1/1 variants sufficient at 45/45
- rank 8: 4/4 variants sufficient at 45/45
- rank 7: 6/6 variants sufficient at 45/45
- rank 6: **4/4 variants sufficient at 45/45**
- rank 5: **0/1 sufficient**; the cosine-only core failed replicate 2 advection ID-h8 at 1.0529x persistence

The four rank-6 variants each retain the five Fourier directions Phase 22 found operationally critical plus exactly one of the four sine directions. Their worst registered ratios were between 0.698x and 0.804x persistence.

## Design

The spatial basis was the same fixed real orthonormal 3x3 DFT basis used in Phase 22. The omission pool was frozen to the four sine directions that Phase 22 had independently shown could each be removed while retaining 45/45 performance:

- sin_x
- sin_y
- sin_diag_plus
- sin_diag_minus

Phase 24 exhaustively evaluated all subsets of that four-element pool: 1 rank-9 control, 4 rank-8 single omissions, 6 rank-7 pair omissions, 4 rank-6 triple omissions, and the single rank-5 four-way omission. Within each replicate all variants shared the same codebooks, physical trajectories, contexts, targets, ridge, and recursive token projection.

The representation stayed frozen: 18-bit asinh-companded causal innovation tokens, eight context frames, wave+Gray representation fitting only, with NLS and advection unseen during representation training.

## Interpretation

This resolves the apparent contradiction between Phases 20–22. Random low-rank projection fails because it can remove dynamically important spatial directions. Full rank 9 is not intrinsically required: once the subspace is structured so the important even/DC directions are preserved, substantial compression is possible.

Within this benchmark, deleting any three of the four tested sine directions is robust, but deleting all four is not. In other words, the tested four-law suite needs at least one member of this odd/sine directional subspace in addition to the five retained DC/cosine directions.

This is a benchmark-conditional result, not a universal rank-6 theorem. Other dynamics may require additional odd directions or larger spatial support.

## Next falsification

The next useful step is to stop optimizing compression on the same four laws and introduce a qualitatively new nonlinear-flux PDE. A two-component viscous Burgers system is a clean test because its dynamics require products between local state and spatial-gradient information. The current linear-spatial-plus-center-polynomial identifier does not contain those multiplicative spatial interactions. A failure there would identify the next hypothesis-class boundary while keeping the tokenizer and spatial support fixed.

## Claim boundary

These experiments use known synthetic dynamics. The results are evidence about representation and context-only system identification under the frozen benchmark, not discovery of new physical laws, AGI, or a universal minimal operator basis.
