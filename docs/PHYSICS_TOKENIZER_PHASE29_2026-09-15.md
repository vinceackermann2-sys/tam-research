# Physics Tokenizer Phase 29 — Hamilton–Jacobi Boundary

Status: **FAIL** under the preregistered 12/12 gate.

Phase 28 had produced a five-family unified context-only selector between the 26-feature simple library and the 62-feature center×spatial interaction library. Phase 29 introduced a sixth unseen synthetic family: two independent viscous Hamilton–Jacobi fields with dynamics containing squared spatial gradients, `nu*Laplacian(z) - 0.5*lambda*((Dx z)^2 + (Dy z)^2)`. Hamilton–Jacobi data remained completely absent from representation fitting.

The primary model was unchanged from Phase 28. Across three fresh registered replicates it passed only **7/12** Hamilton–Jacobi cells. ID-h8 failed in all three replicates with ratios `1.527`, `1.130`, and `4.512`. Combined OOD passed only one of three (`0.826`, `1.314`, `1.462`). Spectral and parameter OOD passed 3/3.

A preregistered diagnostic expanded the 62-feature library with all within-channel quadratic products among the four parity-odd real-Fourier spatial coordinates, producing an 82-feature library. The raw 82-feature one-step fit was near-exact in every replicate: median one-step ratios were approximately `1.9e-3` on ID, `2.2e-4` on spectral OOD, `1.9e-4` on parameter OOD, and `1.5e-5` on combined OOD. This confirms that the declared richer hypothesis class contains an excellent local description of the synthetic update.

However, fitting that 82-feature law from quantized context and recursively rolling it out was not robust: token-82 passed only **8/12** cells. ID-h8 passed only 1/3 and combined OOD only 1/3. Therefore the Phase-29 boundary is not merely missing candidate terms. Once the richer law is estimated from short quantized context, coefficient error/conditioning can dominate long-horizon rollout even when the raw-state law is representable.

This narrows the next question to evidence quality and system identification under quantization: can more observed transitions, or a better noise-aware estimator, recover the richer law reliably without changing the tokenizer?

No registered seed was retried or replaced. All runs were local CPU only. This remains a benchmark on known synthetic equations; it is not new physics or unrestricted scientific-law discovery.
