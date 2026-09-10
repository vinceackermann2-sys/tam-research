# CORTEX-S production scan integration v1

Classification: `ENGINEERING_PRODUCTION_INTEGRATION_ZERO_CREDIT_ONLY`.

Tracked by issue #820. This stage is additive and does not modify the historical `language_model.py`, the #817 GPU harness, any Modal workflow, or any scientific/full-training protocol.

## Frozen predecessor

The single H100 engineering attempt is permanently consumed:

- issue: #817
- Actions run: `34450068844`
- job: `102783560405`
- attempt: `1`
- executed source: `39f7e55488fe4c3b03ae2664ef89365fed6d4ce7`
- engineering seed: `2026090907`
- trigger: `[modal-cortex-s-affine-scan-triton-gpu-microbench-v1]`
- result root: `/vol/cortex-s-v0/affine-scan-triton-gpu-microbench-v1`

Never rerun, retry, redispatch, or reuse any of those identifiers.

The remote H100 result completed and durably wrote `RESULT.json` before the local GitHub runner encountered a post-result Modal deserialization failure caused by the local environment not having `torch`. The outer red job is therefore infrastructure evidence only and does not erase the completed remote engineering result.

Frozen remote result:

- device: NVIDIA H100 80GB HBM3, compute capability 9.0
- BF16 production shape `[64,512,128]`: correctness PASS
- FP32 probe: PASS
- chunk-carry: PASS
- production scan median forward+backward: `0.0016666785999998267 s/iter`
- Triton scan median forward+backward: `0.0005455185999998946 s/iter`
- isolated speedup: `3.055218648823612x`
- isolated classification: `PROMISING_SCAN_SYSTEMS_CANDIDATE`

This is systems evidence only, not a whole-model speed result and not scientific evidence.

## Integration shape

`production_scan_integration_v1.py` wraps the already validated candidate behind an explicit conversion seam:

1. `production_affine_scan` dispatches to `affine_scan_triton_candidate` only for a real CUDA tensor when Triton is available.
2. CPU and unsupported environments call the existing production `language_model.affine_scan` directly. They do not route through the candidate's sequential CPU oracle.
3. `ProductionScanPersistentWorldState` changes only the scan execution backend. Candidate/keep/out projections, recurrence inputs, returned full state, final carry, and output projection retain the existing contract.
4. `convert_world_state_to_triton_scan` replaces only each block's world-state module class while reusing the exact existing candidate/keep/out submodules and Parameter objects. Parameter names, identities, and count must remain unchanged.
5. `build_memory_lean_grouped_triton_scan_cortex_100m` composes this seam after the already-production-tested grouped-v4 builder and rechecks the frozen `101,778,112` trainable-parameter count.
6. The context manager only exposes the integrated builder to an existing training harness. It does not execute training or allocate a GPU.

The default `CortexSLM` remains unchanged at this stage. This makes the integration explicitly opt-in for the next exact-source engineering preflight rather than silently changing all CORTEX-S users before integrated CUDA evidence exists.

## Zero-credit falsifiers

CPU/static tests must prove:

- production CPU scan outputs and gradients are preserved;
- CPU fallback never invokes the Triton/sequential candidate path;
- the dispatch seam can be forced under test to prove candidate selection without pretending CPU validates CUDA;
- world-state output, final carry, full states, input gradients, and initial-state gradients remain equal on CPU;
- conversion preserves every Parameter object and parameter name;
- conversion fails closed if applied twice;
- consumed #817 identifiers are bound in status and all new spend/training authorities are false;
- the integration source contains no Modal import, H100 allocation, `.remote()` call, result namespace, trigger, or full-training entrypoint;
- historical `language_model.py` remains untouched in this additive stage.

## Next boundary

CPU/static CI green is necessary but not sufficient for merge. The integrated conversion path must next receive separately preregistered CUDA equivalence evidence on an immutable source. That future attempt must use a fresh engineering seed, fresh trigger title, fresh result namespace, and explicit paid authorization.

Only after integrated CUDA evidence and fresh exact-head CI may this branch reach an explicit merge decision. After merge, a separately preregistered full 100M production preflight must determine whether the whole-model projected full envelope is `<= 8,500 s`.

Nothing in this stage authorizes seed `8100`, scientific seeds `48131/48132/48133`, 2B/full training, architecture freeze, scaling, or any breakthrough/AGI/SSI/alignment/continual-learning claim.
