# CORTEX-S v0 100M systems optimization v2 — fused affine scan contract

Classification: `ENGINEERING_SYSTEMS_OPTIMIZATION_ONLY`.

## Consumed evidence

Grouped v4 preflight issue #807 / Actions run `34362180842` / job `102501691585` / source `230ac116534272f41dc99803df0f59b7012dacf7` is consumed. Engineering seed `2026090906` is consumed. The run measured 242,168.7858 tok/s, 318.283 s compile time, and a 9,402.856 s projected full envelope against the unchanged <=8,500 s gate. `full_run_authorized=false`; no paired seed-8100 training was launched.

The v4 temporary-memory changes were effectively neutral versus grouped v3 steady-state throughput, so this stage does not continue allocation/copy tweaks in the MoE path.

## Recurrent-scan target

Every one of the 24 CORTEX-S blocks contains a persistent-state recurrence

`s_t = a_t * s_(t-1) + b_t`

with state width 128. At training sequence length 512 the production implementation performs nine Hillis-Steele-style whole-tensor stages and materializes shifted tensors repeatedly. The previously tested slice/clone alternative was slower and is rejected as a production candidate.

The next credible systems direction is a genuinely fused training kernel: parallelize over independent `(batch,state)` lanes, execute the length-512 recurrence inside one device program, and use an explicitly derived fused reverse recurrence for backward. If performant, this can replace the repeated global scan-stage traffic with one forward and one backward device program per recurrent layer.

## Frozen semantics

The zero-credit semantic module freezes the exact forward and reverse equations:

- forward: `s_t = a_t*s_(t-1)+b_t`;
- reverse adjoint: `lambda_t = grad_t + a_(t+1)*lambda_(t+1)`;
- `d b_t = lambda_t`;
- `d a_t = lambda_t*s_(t-1)`;
- `d initial = a_0*lambda_0`;
- returned full state sequence and chunk-to-chunk final-state carry are unchanged;
- no parameters are added or removed;
- architecture, routing, attention schedule, optimizer, corpus and scientific protocol remain unchanged.

The first exact-head contract head `d481e88d06daed86f7d0ac91bb5b67df9394dddf` passed fresh CI `34365940565`, CORTEX-S CPU gate `34365940614`, and 100M/2B zero-credit gate `34365940538` before the guarded kernel implementation was added.

## Guarded Triton candidate

The successor implementation now contains a custom-autograd candidate behind the frozen contract:

- CUDA/Triton path: one forward kernel loops left-to-right along sequence while parallelizing across `batch * state` lanes;
- CUDA/Triton backward: one reverse kernel implements the frozen analytic adjoint directly;
- intended production scan shape is `[64, 512, 128]`;
- CPU/unsupported-device execution uses the independent sequential semantic oracle plus analytic backward, allowing zero-credit custom-autograd validation;
- no call site in `language_model.py` or `PersistentWorldState` has been changed;
- parameter count delta is exactly zero;
- `production_wired=false` and `gpu_benchmark_authorized=false` remain hard-coded.

CPU tests compare the candidate forward and gradients against the current production `affine_scan` in float64 across multiple lengths, verify chunk-carry equivalence, validate failure on shape/dtype drift, and statically verify that the production model does not reference the Triton candidate.

This implementation is **not GPU correctness or performance evidence**. A Triton source file existing in the repository does not establish that it compiles efficiently on H100, that its numerical error is acceptable in BF16, or that it improves end-to-end model throughput.

## Authority ceiling

No GPU spend, new engineering seed, paid namespace, production scan replacement, paired seed-8100 training, scientific seed use, architecture freeze or breakthrough claim is authorized here.

A future H100 systems microbenchmark requires all of the following separately:

1. fresh full-repository CI, CORTEX-S CPU gate and 100M/2B zero-credit gate on one exact final candidate head;
2. explicit merge authorization for that exact head if merge is desired;
3. a fresh post-merge live-main audit;
4. a new engineering seed and unique single-use benchmark namespace;
5. an isolated GPU correctness gate comparing Triton forward/backward against production/reference before timing;
6. explicit paid-spend authorization.

The consumed #807/v4 namespace and seed `2026090906` must never be reused. Paired seed `8100` and reserved scientific seeds `48131`, `48132`, `48133` remain untouched.
