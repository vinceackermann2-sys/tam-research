# Physics Tokens Phase 1 — Generalization Result (2026-09-13)

## Status

**FAIL under the preregistered Phase-1 gate.** This is a useful negative result. Phase 0 established that a 256-entry direct-state vector-quantized representation can support recursive physics prediction. Phase 1 asks whether the same frozen representation plus the learned centroid-residual dynamics extrapolates beyond the training spectrum. It does not.

## Frozen setup

- Training data: 16×16 periodic wave fields, initial spatial modes up to 3, wave speed `c ∈ [0.6, 1.0]`, `dt=0.12`.
- Codec: frozen direct-state VQ, 256 centroids over normalized `(u, v)` state.
- Context: 3 frames.
- Discrete dynamics: residual convolution operating on decoded centroid states; every predicted frame is requantized before the next recursive step.
- Fresh model seeds: `9132611`, `9132612`, `9132613`.
- Pass rule: **every seed** must beat raw persistence (`MSE ratio < 1`) on ID h8, spectral-OOD h3, and combined-OOD h3.

## Primary result

| Metric | Seed 9132611 | Seed 9132612 | Seed 9132613 | Mean |
|---|---:|---:|---:|---:|
| ID h8 / persistence | 0.1650 | 0.1792 | 0.1619 | **0.1687** |
| Spectral OOD h3 / persistence | 1.5498 | 1.5537 | 1.5831 | **1.5622** |
| Combined OOD h3 / persistence | 1.9674 | 1.9316 | 2.0190 | **1.9727** |

The result is seed-consistent: 3/3 seeds pass the long ID test, 0/3 pass unseen higher-frequency initial conditions, and 0/3 pass higher-frequency + faster-wave combined OOD.

## Matched raw-continuous control

The same residual ConvNet architecture trained directly on normalized continuous states passes the OOD tests: mean spectral-OOD h3 ratio **0.2885** and mean combined-OOD h3 ratio **0.2894**. Therefore the evaluation set is not intrinsically too difficult for this model class.

## Where the failure occurs

The discrete model is already bad **before** projecting its continuous output back to the token vocabulary. For seed 9132611, spectral-OOD h3 is ~1.506× persistence before requantization and combined-OOD h3 is ~1.932×. Therefore repeated nearest-centroid projection is not the main cause of failure.

The target-token oracle remains feasible: decoding the correct target tokens gives ~0.163× persistence on spectral OOD and ~0.690× on combined OOD. So the frozen 256-token vocabulary does not make the gate mathematically impossible.

## Known-solver diagnostic

Starting from the quantized state and applying the known wave-equation solver provides a stronger separation test. At h3:

- spectral OOD: **0.067× persistence** from only the initially quantized state; **0.294×** with requantization after every step;
- combined OOD: **0.224× persistence** from only the initially quantized state; **0.877×** with requantization after every step.

This shows that the tokenized state still contains enough physical information to beat persistence even in the combined OOD condition. The dominant failure is therefore the **learned transition law's extrapolation**, not catastrophic information loss in the codec.

## Scientific interpretation

The Phase-0 result should be narrowed to: *a compact discrete state vocabulary can support accurate recursive prediction within and near its training regime.* Phase 1 rejects the stronger claim that the current unconstrained learned transition automatically discovers a distribution-independent physical update rule.

The next justified experiment is to keep the codec and Phase-1 benchmark frozen and add a structured local transition / latent-parameter identification mechanism. If that fixes spectral OOD, it would support the hypothesis that discrete state is viable but the transition architecture needs an operator-like inductive bias. It would still not constitute discovery of new physics, because this benchmark is synthetic and the wave equation is known.
