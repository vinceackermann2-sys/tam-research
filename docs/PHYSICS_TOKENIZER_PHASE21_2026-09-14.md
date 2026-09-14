# Phase 21 — Paired random spatial-rank sweep

## Result

Phase 21 found a replicated threshold **within the preregistered random nested-projection protocol**: rank 9 per channel was the smallest tested rank that passed the frozen transfer gate.

| spatial rank / channel | feature dim | passing registered cells | result |
|---:|---:|---:|---|
| 5 | 18 | 20 / 45 | FAIL |
| 6 | 20 | 21 / 45 | FAIL |
| 7 | 22 | 29 / 45 | FAIL |
| 8 | 24 | 33 / 45 | FAIL |
| 9 | 26 | 45 / 45 | PASS |

The gate required every one of three fresh registered replicates to beat raw-state persistence on all 15 wave, Gray-Scott, NLS, and advection evaluation splits.

## Experimental design

Each channel's 3x3 neighborhood was represented in one seeded, data-independent 9-dimensional orthogonal basis. Within a replicate, ranks 5 through 9 used nested prefixes of exactly the same basis and exactly the same physical trajectories, codebooks, contexts, and targets. The only changed quantity was the number of retained spatial directions.

The frozen dynamics identifier remained an eight-frame context-only ridge regression. Predicted states were recursively projected through the same Phase-10 18-bit companded token representation after every step. NLS and advection remained excluded from representation training.

## Registered outcomes

Rank 9 passed 15/15 splits in all three fresh replicates. Its worst registered ratio was 0.4912x persistence, and its median ratios remained below persistence on every split. Representative rank-9 median ratios were 0.0061 for wave ID, 0.2013 for NLS spectral OOD, 0.3748 for NLS ID, 0.0945 for advection ID, and 0.0426 for advection spectral OOD.

Every lower tested rank failed. Rank 8 was especially informative: replicate 1 passed 14/15 cells and missed only NLS spectral OOD at 1.0767x persistence, while replicates 2 and 3 had worst cells of 31.17x and 34.60x. Thus deleting a single spatial direction can be nearly harmless or catastrophic depending on which direction the random rank-8 subspace omits.

## Interpretation

The result strengthens the Phase-20 conclusion. The issue is not merely that rank 5 was too aggressive or that PCA selected the wrong directions. Under paired random projections, adding spatial directions progressively improves transfer, but robust transfer across all four synthetic law families appears only when the complete nine-dimensional 3x3 spatial patch is retained.

This does **not** establish a universal nine-dimensional lower bound. The result is conditional on the fixed 3x3 neighborhood, random orthogonal projection family, token representation, context length, polynomial center features, ridge identifier, synthetic simulator suite, and pass criterion. In particular, a carefully structured rank-8 basis may still preserve all dynamically relevant operators.

## Next falsification

The natural next experiment is therefore not another blind rank sweep. Phase 22 should use a fixed interpretable orthonormal basis of the 3x3 neighborhood and perform leave-one-direction-out rank-8 tests. A clean data-independent choice is the real 3x3 discrete Fourier basis: DC plus cosine/sine pairs for the four nonredundant spatial-frequency directions. Omitting one Fourier mode at a time can test which local spatial direction is responsible for the rank-8 failures and whether any structured rank-8 subspace can still satisfy the full transfer gate.

## Claim boundary

These experiments recover and transfer known synthetic dynamics. They are evidence about representation and system-identification structure, not evidence of novel physical laws, unrestricted scientific discovery, AGI, or a universal theorem about physical operator dimension.
