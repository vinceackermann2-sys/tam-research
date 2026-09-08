# CORTEX-S v0 100M systems optimization — zero-credit stage

Classification: `ENGINEERING_SYSTEMS_OPTIMIZATION_ONLY`

Parent evidence: issue #769 / source `6cb1882aa5b02e4f8917a7cae44e6d1925b9e290`.

## Why this stage exists

The first H100 production-graph calibration was functionally healthy but failed the preregistered wall-clock gate:

- measured throughput: 133,384.4585 tok/s;
- measured 40 steps / 2,621,440 tokens in 19.6533 s;
- peak VRAM: 48.3247 GiB;
- finite loss: PASS;
- exact 101,778,112 parameter count: PASS;
- projected full envelope: 16,676.5342 s;
- allowed envelope: <= 8,500 s;
- result: `ABORT_FULL_RUN`.

Calibration seed `910001` is consumed and must never be retried. No paired seed-8100 full run occurred. Reserved scientific seeds `48131`, `48132`, and `48133` remain untouched.

## Candidate 1: grouped sparse expert execution

The production `TrulySparseMoE` is mathematically sparse but executes a Python loop over eight experts. Each expert performs two independently shaped GEMMs after dynamic gather, then scatters results back. At eight experts this creates up to sixteen logical expert GEMMs per layer.

`systems_optimization.GroupedSparseMoECandidate` keeps the exact router/top-k semantics and packs the same selected token/expert assignments into expert-contiguous rows. Its intended CUDA path uses two grouped GEMMs per layer — one input projection and one output projection — while still executing only top-2 of eight possible assignments per token.

Important constraints:

- the production model is not switched to this candidate in this stage;
- CPU uses an explicit grouped-slice reference path so forward/backward equivalence can be tested without GPU spend;
- expert parameter count is unchanged;
- routing decisions and softmax weights are unchanged;
- no dense all-expert evaluation is introduced;
- a later CUDA benchmark must measure dtype/autocast behaviour rather than assuming speed from the lower logical launch count.

PyTorch 2.10 exposes `torch.nn.functional.grouped_mm`; on CUDA its grouped-MoE forward form accepts expert-contiguous rows plus cumulative expert offsets. This is only a feasibility basis, not performance evidence for CORTEX-S.

## Candidate 2: allocation-reduced affine scan

The production recurrent world state uses a differentiable Hillis-Steele affine scan. At every logarithmic scan stage it constructs padded shifted tensors with `torch.cat`. The systems candidate preserves the same recurrence but clones the current prefix tensors and updates only the participating suffix.

This is also not wired into production. CPU tests compare both values and gradients against the existing scan. A future systems benchmark must determine whether fewer padding/cat operations actually reduce H100 wall time after compilation.

## Evidence required before any new GPU spend

This zero-credit stage may only establish semantic equivalence and a credible implementation path. A later GPU systems microbenchmark requires all of the following:

1. fresh immutable source SHA;
2. fresh engineering-only seed/namespace — never `910001`;
3. fresh full CORTEX-S CPU CI;
4. explicit benchmark protocol and stop threshold;
5. explicit authorization for that new paid engineering attempt.

A systems speedup does not constitute scientific evidence. The 100M/2B scientific comparison remains unrun until a separately preregistered production preflight passes its budget gate.
