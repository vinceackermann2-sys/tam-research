# CORTEX-S v0 100M systems optimization v2 — fused affine scan contract

Classification: `ENGINEERING_SYSTEMS_OPTIMIZATION_ONLY`.

## Consumed evidence

Grouped v4 preflight issue #807 / Actions run `34362180842` / job `102501691585` / source `230ac116534272f41dc99803df0f59b7012dacf7` is consumed. Engineering seed `2026090906` is consumed. The run measured 242,168.7858 tok/s, 318.283 s compile time, and a 9,402.856 s projected full envelope against the unchanged <=8,500 s gate. `full_run_authorized=false`; no paired seed-8100 training was launched.

The v4 temporary-memory changes were effectively neutral versus grouped v3 steady-state throughput, so this stage does not continue allocation/copy tweaks in the MoE path.

## Recurrent-scan target

Every one of the 24 CORTEX-S blocks contains a persistent-state recurrence

`s_t = a_t * s_(t-1) + b_t`

with state width 128. At training sequence length 512 the production implementation performs nine Hillis-Steele-style whole-tensor stages and materializes shifted tensors repeatedly. The previously tested slice/clone alternative was slower and is rejected as a production candidate.

The next credible systems direction is a genuinely fused training kernel: parallelize over independent `(batch,state)` lanes, execute the length-512 recurrence inside one device program, and use an explicitly derived fused reverse recurrence for backward. This would replace multiple global scan stages with one forward kernel and one backward kernel per recurrent layer if it proves performant.

## Frozen semantics

The new zero-credit module freezes the exact forward and reverse equations before any CUDA implementation:

- forward: `s_t = a_t*s_(t-1)+b_t`;
- reverse adjoint: `lambda_t = grad_t + a_(t+1)*lambda_(t+1)`;
- `d b_t = lambda_t`;
- `d a_t = lambda_t*s_(t-1)`;
- `d initial = a_0*lambda_0`;
- returned full state sequence and chunk-to-chunk final-state carry are unchanged;
- no parameters are added or removed;
- architecture, routing, attention schedule, optimizer, corpus and scientific protocol remain unchanged.

CPU tests compare the semantic oracle and the explicit analytic backward against the current production parallel scan in float64 across multiple lengths, and verify chunk-carry equivalence.

## What this stage does not claim

There is intentionally no CUDA/Triton dispatch in this PR. An unmeasured kernel is not a speedup. This stage proves only that the forward/backward contract for a future fused implementation is explicit and mechanically testable.

No GPU spend, new engineering seed, paid namespace, paired seed-8100 training, scientific seed use, architecture freeze or breakthrough claim is authorized here.

## Next boundary

After fresh exact-head CPU/full-repository CI, a later zero-credit successor may implement a guarded CUDA/Triton kernel behind this contract. That implementation must pass the same forward/backward/chunk-carry tests and static fail-closed checks before any separately preregistered H100 microbenchmark is considered.

A future H100 benchmark requires a new engineering seed, unique result namespace, exact immutable source SHA and explicit spend authorization. The consumed #807/v4 namespace and seed `2026090906` must never be reused.
