# Physics Tokens Phase 3 — Learned Local Operator (2026-09-13)

## Status

**PASS under the preregistered 3-replicate gate.**

Phase 3 removes the exact spatial Laplacian supplied in Phase 2. The frozen 256-token codec remains unchanged. For each replicate, a fresh low-frequency training set is tokenized with the frozen codec, and a full unconstrained 3×3 periodic linear spatial kernel is fit by ridge least squares from quantized trajectories. No true wave-speed labels and no true Laplacian coefficients are used to fit the kernel.

At inference, a single sequence-specific scalar is estimated from the three quantized context frames, and rollout uses the learned spatial kernel inside a supplied velocity-Verlet temporal update. Every predicted state is requantized after every step.

## Confirmatory results

| Replicate | ID h8 / persistence | Spectral OOD h3 | Combined OOD h3 | Pass |
|---|---:|---:|---:|---|
| 1 | 0.4378 | 0.3048 | 0.9054 | yes |
| 2 | 0.3723 | 0.4138 | 0.8136 | yes |
| 3 | 0.3797 | 0.3381 | 0.7616 | yes |
| **Mean** | **0.3966** | **0.3522** | **0.8269** | **3/3** |

The frozen gate required every fresh operator-training/evaluation replicate to beat persistence on all three tests. All three passed.

## Learned operator diagnostics

The learned 3×3 kernels are reproducible across the three independently sampled operator-training sets. Their raw cosine similarity to the canonical five-point Laplacian is moderate (mean **0.576**), not near 1.0. Their coefficient sums are nevertheless close to zero, which is consistent with a derivative-like operator that annihilates constant fields. Because scale is absorbed into the inferred sequence scalar, raw coefficient equality is not expected.

The inferred sequence scalar strongly tracks the true squared wave speed on ID and spectral-OOD sets (secondary diagnostic only), despite never receiving the speed label.

## Scientific interpretation

This is stronger than Phase 2: **the exact spatial law no longer has to be supplied.** A local operator learned from quantized low-frequency trajectories extrapolates to unseen higher-frequency states and to faster dynamics outside the training speed range.

The result still has important limits. We supply 3×3 locality and the velocity-Verlet temporal update structure, so this is not unconstrained discovery of an arbitrary scientific law. It is evidence that a discrete scientific-state vocabulary plus structured operator learning can recover an extrapolative local law in a controlled synthetic system.

The next justified test is to remove the supplied temporal integrator form and learn the time-update structure itself while keeping the codec and OOD gates frozen.
