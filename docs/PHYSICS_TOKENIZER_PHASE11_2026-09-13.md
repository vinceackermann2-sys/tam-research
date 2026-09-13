# Physics tokenizer Phase 11 — frozen compander cross-law validation

Date: 2026-09-13

Status: **CROSSLAW_PASS (synthetic engineering validation only)**

Branch: `experiment/physics-tokenizer-phase11-crosslaw-compander-20260913`

Parent Phase-10 HEAD: `283218939a0676c6efff2b6457da346cfc9d499f`

## Question

Phase 10 showed that a fixed tail-aware causal innovation coordinate system solved the preregistered combined wave speed+spectral OOD representation screen, but it had mixed behavior on easier/long-horizon ID conditions. Phase 11 therefore did not tune the representation further. It froze the Phase-10 compander and asked a harder question: can the same 18-bit discrete coordinate system support short-context system identification across both the wave equation and nonlinear Gray-Scott reaction-diffusion under seven preregistered ID/OOD tests?

## Frozen representation and identifier

The Phase-11 protocol was committed before any registered execution at `5c9d0ba5281ff2dc53086d7121c0634fc612202c`.

The compander was unchanged from Phase 10:

- causal innovation `d_t = x_t - xhat_{t-1}`;
- training-mixture standardization `z=(d-mu)/sigma`;
- fixed `asinh(z)` companding;
- fixed standardized support `z in [-16,+16]` via clipping in companded coordinates;
- exactly 512 uniform levels per channel;
- two 9-bit indices packed into one 18-bit token per cell per saved frame;
- same 18-bit state-RVQ absolute anchor for frame 0.

The downstream identifier was also unchanged: 3 decoded context frames, the same 16-dimensional local/D4/polynomial physical-coordinate feature basis, per-sequence RMS feature scaling, ridge `0.001`, no law ID, no family router, no hidden parameter labels, and no supplied governing equation or known integrator.

State RVQ and plain learned innovation RVQ remained matched-rate paired controls but could not rescue a compander failure.

## Registered gate

A `CROSSLAW_PASS` required **every one of three fresh registered replicates** to have companded recursive rollout MSE / raw-state-persistence MSE `< 1.0` on all seven tests:

1. wave ID, horizon 8;
2. wave spectral OOD, horizon 3;
3. wave combined speed+spectral OOD, horizon 3;
4. Gray-Scott ID, horizon 8;
5. Gray-Scott sharp-initial-condition OOD, horizon 3;
6. Gray-Scott parameter OOD, horizon 3;
7. Gray-Scott combined parameter+sharp OOD, horizon 3.

No registered replicate could be retried or replaced, and no range, bit budget, feature, ridge, horizon, or threshold could change after consumption.

## Implementation provenance

The Phase-11 runner was sealed at commit `28b3bcc2b61d3e0a5676f15e71d88cea535e8b79`, blob SHA `b8106005a1e9d29b8bc26534b9c38389658787d0`. Tests were sealed at `a1599c6f758cbbf4abfa01b08f7ccf905918acbd`, blob SHA `a6fdded7c757194cf878f8b7cde01fa467071795`.

Exact committed bytes reproduced locally and the invariant suite passed 4/4 before any registered Phase-11 seed was consumed. A separate full-cost non-registered smoke completed in about 7.3 seconds and was used only to establish runtime/numerical viability.

## Registered results

Companded rollout ratios versus raw persistence:

| Test | Rep 1 | Rep 2 | Rep 3 | Median |
|---|---:|---:|---:|---:|
| wave ID h8 | 0.0805 | 0.4784 | 0.4343 | 0.4343 |
| wave spectral OOD h3 | 0.0508 | 0.0405 | 0.0556 | 0.0508 |
| wave combined OOD h3 | 0.2988 | 0.2501 | 0.4011 | 0.2988 |
| Gray ID h8 | 0.3451 | 0.2844 | 0.0745 | 0.2844 |
| Gray sharp OOD h3 | 0.0962 | 0.0906 | 0.1081 | 0.0962 |
| Gray parameter OOD h3 | 0.0924 | 0.0799 | 0.0798 | 0.0799 |
| Gray combined OOD h3 | 0.1350 | 0.1611 | 0.1223 | 0.1350 |

All **21/21 preregistered split-replicate cells** beat raw persistence.

Registered result commits:

- replicate 1: `759931b77f0f04b9d11d4581266646760d38f3a3`
- replicate 2: `427905c79293ff19ac1f2d5eb4bcdd553df5526b`
- replicate 3: `6d96195f6248e97e2cf1a196f0dabe018df55176`

Aggregate result: `results/physics_tokens/phase11_crosslaw_compander_summary.json`, commit `835b5cd0f686d3935e5c6e58b6391a21a2a01897`.

**Frozen Phase-11 decision: `CROSSLAW_PASS`.**

## Paired-control interpretation

The result does **not** show that the companded codec is uniformly better than every matched-rate alternative.

On wave combined OOD, the compander beat both state RVQ and plain innovation in all three registered replicates. Median ratios were approximately state `1.054`, plain innovation `0.919`, companded `0.299`.

On Gray combined OOD, the compander beat state RVQ in all three replicates, but it did not beat plain innovation in all three. Median ratios were approximately state `0.187`, plain innovation `0.139`, companded `0.135`.

On several easier Gray/ID conditions, plain innovation was also lower-error than companding. For example, the median Gray-parameter-OOD ratio was about `0.064` for plain innovation versus `0.080` for companding. The Phase-11 evidence therefore supports **robust viability under the frozen all-seven gate**, not universal codec superiority.

## What changed scientifically from Phase 9

The useful result is that the representation failure seen in Phase 9 was not inevitable for causal discrete innovations. A fixed tail-aware coordinate transform, chosen before the fresh Phase-10/11 registered runs and then held unchanged, substantially reduced the hard wave combined-OOD representation failure and remained usable when transferred to a qualitatively different nonlinear PDE family.

This is consistent with the idea that representation geometry/support can determine whether a short-context law estimator sees a stable coordinate system under distribution shift. It does not establish that `asinh` is uniquely optimal, nor that the supplied 16-feature estimator discovered unrestricted governing laws.

## Execution and claim boundary

All registered Phase-11 work ran on local CPU. No Modal/GPU/paid scientific run was used. No registered Phase-11 seed was retried or replaced. `main` was not modified and unrelated repository research was not touched.

This is evidence only for the specified known synthetic PDE systems, discretization, grid, bit budget, feature prior, context length, and OOD shifts. It is **not** evidence of novel physics, unrestricted scientific-law discovery, a universal scientific tokenizer, AGI, or SSI.
