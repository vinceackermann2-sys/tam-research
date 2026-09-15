# Phase 25 — Unseen viscous Burgers transfer

## Result

**FAIL.** The frozen 26-feature identifier passed only **6/12** registered Burgers cells. All three fresh replicates failed ID-h8, and all three failed combined OOD.

Median primary token-26 ratios versus raw-state persistence:
- Burgers ID-h8: **7.392x**
- spectral OOD h3: **0.779x**
- viscosity-parameter OOD h3: **0.763x**
- combined OOD h3: **1.406x**

## Diagnostic result

The preregistered generic 62-feature interaction diagnostic adds every product between center `u` or `v` and each of the 18 linear spatial Fourier coordinates. It was diagnostic only and could not rescue the primary Phase-25 gate.

Nevertheless, token-62 rollout beat persistence on **12/12 registered cells**. Median ratios were:
- ID-h8: **0.208x**
- spectral OOD: **0.124x**
- parameter OOD: **0.167x**
- combined OOD: **0.711x**

Raw-context 62-feature one-step fits were much stronger still, with median error/persistence ratios between approximately `1.9e-6` and `9.0e-5` across the four splits.

## Interpretation

This is strong evidence that the Phase-25 failure is primarily a **hypothesis-class boundary**, not an obvious tokenizer-capacity boundary. The frozen 18-bit token representation preserves enough information for the same short context to support Burgers prediction once generic multiplicative state-times-spatial interactions are available.

The previously successful four-law identifier contained linear spatial terms plus polynomial center terms, but not products between state amplitude and local spatial structure. Burgers flux requires exactly that kind of multiplicative interaction.

This does not make the 62-feature diagnostic a confirmed model. It was selected before execution as a secondary diagnostic and therefore requires its own separately preregistered fresh-seed confirmation before promotion.

## Next falsification

Phase 26 should promote the unchanged 62-feature generic interaction library to the primary identifier and test it on fresh Burgers seeds. A stronger follow-up should then verify that the richer library does not regress the previously solved wave, Gray-Scott, NLS, and advection families.

## Claim boundary

These experiments use known synthetic PDEs. They test representation and system-identification structure; they do not constitute discovery of new physics, unrestricted law discovery, or AGI.
