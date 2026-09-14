# Physics Tokenizer Phase 15 — out-of-class directional advection

Date: 2026-09-14

## Purpose

Phase 15 deliberately tests a fourth synthetic PDE family whose exact update is outside the frozen Phase-10 16-feature hypothesis class. The representation and context-8 identification recipe are inherited from the sealed Phase-14 PASS. The unseen family is periodic 2-D directional advection-diffusion. It is never used while fitting the wave+Gray-Scott representation.

The frozen 16-feature library contains symmetric spatial neighbor sums and polynomial channel terms, but no antisymmetric first derivatives. Directional advection requires centered Dx/Dy terms, so the exact law is not in the frozen feature span.

## Frozen preregistered gate

Three fresh registered replicates were fixed before execution. Each replicate evaluates four splits: ID h8, spectral OOD h3, speed OOD h3, and combined speed+direction+spectral OOD h3. Phase 15 passes only if all 12 registered primary companded error/persistence ratios are below 1.0. No retries or replacement seeds are allowed.

## Result: FAIL

Nine of twelve registered primary cells beat persistence, but three fail. Only replicate 2 passes all four splits. Therefore the preregistered all-replicate gate fails.

Primary ratios by replicate:

| split | rep 1 | rep 2 | rep 3 | median |
|---|---:|---:|---:|---:|
| adv ID h8 | 7.3233 | 0.9302 | 1.1526 | 1.1526 |
| adv spectral OOD h3 | 1.0020 | 0.9296 | 0.8949 | 0.9296 |
| adv speed OOD h3 | 0.8404 | 0.8281 | 0.8974 | 0.8404 |
| adv combined OOD h3 | 0.8666 | 0.9537 | 0.8949 | 0.8949 |

Replicate-level all-four status: FAIL, PASS, FAIL.

## Feature-class diagnostic

A raw diagnostic augments the frozen basis with centered Dx and Dy for both channels, increasing feature dimension from 16 to 20. This control cannot redefine the Phase-15 pass/fail decision.

Median context one-step error/persistence ratios:

| split | frozen 16 | directional 20 | improvement |
|---|---:|---:|---:|
| adv ID | 0.7642 | 1.274e-5 | ~5.997e4x |
| spectral OOD | 0.8608 | 2.354e-6 | ~3.657e5x |
| speed OOD | 0.8060 | 3.193e-6 | ~2.524e5x |
| combined OOD | 0.8890 | 3.532e-7 | ~2.517e6x |

This is strong evidence that missing directional operators are a genuine bottleneck in the frozen hypothesis class. The current token model is not uniformly catastrophic on the unseen law: persistence is beaten in 9/12 cells, and one replicate passes all four splits. The failure is therefore seed-sensitive, but it is sufficient to falsify the preregistered universal gate.

The 20-feature raw control is not yet a stable replacement model. Although it identifies the one-step law almost exactly, some multi-step rollouts become unstable because the added derivative terms coexist with redundant frozen polynomial/spatial features. The scientifically justified next test would keep the tokenizer fixed while replacing the transition feature set with a compact operator basis that contains directional derivatives, then preregister fresh seeds to test whether the 12/12 gate is restored.

## Interpretation limits

This is a synthetic known-law falsification. It does not establish novel physics, AGI, or unrestricted scientific-law discovery. What it establishes is narrower: the representation survived three prior PDE families, but generalization is bounded by the transition hypothesis class when the required local operator is absent.

Execution used local CPU in this environment; no Modal GPU run was used for Phase 15.
