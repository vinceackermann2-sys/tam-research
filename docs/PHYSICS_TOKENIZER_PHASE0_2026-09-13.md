# Physics-token Phase-0 — 2026-09-13

## Scope

This is a feasibility experiment for **non-text discrete tokens of physical state** using a periodic 2-D wave equation. It is not evidence of a new law of physics, AGI, or general scientific discovery.

The numeric state at each cell is `[u, du/dt]`. The fixed primary gate is: on the held-out in-distribution split at autoregressive horizon 3, decoded **discrete-token rollout MSE must be below raw-state persistence MSE**.

Locked benchmark: 16×16 grid, sequence length 8, context 3, `dt=0.12`, 128 training sequences, 48 ID validation sequences, 48 OOD-speed sequences, benchmark seed 74193.

## Learned neural codec: fixed gate was infeasible

The learned autoencoder + 256-code VQ run reconstructed validation data with normalized MSE **0.06051**, but the oracle ground-truth-token decode at ID h3 was **0.0197467 MSE**, worse than raw persistence at **0.0134323** (1.4701×). Therefore perfect token prediction still could not satisfy the fixed h3 gate with that codec.

## Direct-state VQ: codec fixed, categorical dynamics failed

Replacing only the codec with 256 centroids learned directly on normalized per-pixel `[u,v]` reduced validation reconstruction normalized MSE to **0.006954**, used **256/256** codes with perplexity **243.29**, and moved the oracle h3 floor to **0.0024099** (0.1794× persistence).

The completed 20-epoch categorical token dynamics run still failed ID h3: **0.0213066 vs 0.0134323**, ratio **1.5862×**. It did beat persistence on OOD-speed h3 (0.7459×), indicating useful motion signal. A later 80-epoch extension did not produce a durable result before the execution ceiling and is classified as **infrastructure-incomplete, not scientific evidence**; it was not retried.

## Centroid-aware residual token dynamics: PASS

The successful ablation keeps the same direct VQ codec and benchmark but respects codebook geometry. It decodes the last three token grids to centroid values, predicts the next normalized `[u,v]` state with a small circular-padding residual ConvNet, then **immediately requantizes the prediction into the same 256-token vocabulary before the next autoregressive step**. Recursive state therefore remains discrete.

Initial model seed 9132601 at 30 epochs:

| Split | h3 model MSE | h3 persistence | ratio |
|---|---:|---:|---:|
| ID | **0.0034487** | 0.0134323 | **0.2567×** |
| OOD speed | **0.0134018** | 0.0527624 | **0.2540×** |

The fixed ID gate passes with about **74% lower MSE than persistence**.

## Fresh-seed replication

After the first pass, the replication rule was frozen before running seeds 9132602 and 9132603: identical data, codebook, architecture and hyperparameters; both must independently pass the same ID h3 gate; no retries or replacement seeds.

| Model seed | ID h3 MSE | ID ratio | OOD h3 ratio | Pass |
|---:|---:|---:|---:|---|
| 9132601 | 0.0034487 | **0.2567×** | 0.2540× | yes |
| 9132602 | 0.0033904 | **0.2524×** | 0.2351× | yes |
| 9132603 | 0.0038226 | **0.2846×** | 0.2515× | yes |

Replication: **3/3 total initializations pass, including both preregistered fresh replication seeds**. Mean ID h3 ratio is **0.2646×**, sample SD **0.0175**.

## Interpretation and next phase

The supported claim is narrow: a compact discrete vocabulary learned from numeric physical state can preserve enough information for geometry-aware autoregressive dynamics, including an unseen wave-speed range, while requantizing every recursive step.

This does **not** demonstrate unknown-physics discovery. The simulator is known, local, low-dimensional, and supplies both field and velocity. A stronger Phase 1 should freeze this design before testing longer horizons, new initial spectra and boundary conditions, different PDE families, a raw-continuous neural baseline, numerical-solver baselines, and independently generated datasets.

## Compute provenance

These Phase-0 runs were executed on the local CPU runtime available in this ChatGPT session. **No Modal GPU run was launched.**
