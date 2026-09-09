# CORTEX-S v0 100M systems microbenchmark v1 repair1

Classification: `ENGINEERING_SYSTEMS_MICROBENCH_ONLY`.

## Why this repair exists

Trigger issue #781 / workflow run `34324639156` failed **before Modal** because the GitHub Actions protocol guard imported the CORTEX-S Python package before PyTorch had been installed. The authoritative job log ended with `ModuleNotFoundError: No module named 'torch'`. `actions/setup-python`, Modal installation/authentication, the zero-GPU corpus verifier, and the H100 launch were all skipped.

Therefore:
- issue #781 and workflow run/trigger namespace `[modal-cortex-s-100m-systems-microbench-v1]` are consumed and must never be rerun or reused;
- no Modal remote function ran and no H100 was allocated;
- engineering seed `2026090901` remains unconsumed;
- paired seed `8100` and fresh scientific seeds `48131`, `48132`, `48133` remain untouched;
- the failure is infrastructure/pre-Modal evidence only and says nothing about CORTEX-S systems performance.

## Repair scope

Repair1 changes **only the launch namespace and the local dependency-free protocol guard**. The actual H100 benchmark implementation remains the already frozen `systems_microbench_v1.py` logic from PR #778:
- exact 101,778,112-parameter CORTEX-S production shape;
- microbatch 64, sequence 512, grad accumulation 2;
- legacy sparse MoE vs packed explicit-BF16 grouped-GEMM candidate;
- same engineering seed `2026090901` because it was not consumed;
- one compile-trigger step, three warmups, 20 measured optimizer steps per variant;
- promising gate >=1.20x, stop below 1.10x;
- semantic loss delta <=0.02;
- grouped peak VRAM <=70 GiB;
- same-allocation affine-scan forward+backward diagnostic;
- no 2B training function and no scientific/full-training authority.

## Fresh immutable namespace

- trigger title: `[modal-cortex-s-100m-systems-microbench-v1-repair1]`
- issue phase: `microbench-v1-repair1`
- Modal app: `cortex-s-v0-100m-systems-microbench-v1-repair1`
- result root: `/vol/cortex-s-v0/100m-systems-microbench-v1-repair1`
- H100 hard timeout: 15 minutes
- one dispatch maximum; the repair1 result root receives a durable `H100_DISPATCH_CONSUMED.json` before benchmark work.

The original v1 result root is not reused even though no Modal function reached it.

## Dependency-free GitHub guard

The repair workflow must not import `architectures.cortex_s` before dependency setup. Its pre-Modal guard uses only shell tests and the Python standard library to inspect source text and freeze:
- engineering seed;
- speedup thresholds;
- exact repair1 app/result namespace;
- absence of any `full_2b` / `train_full_2b` path;
- exact source checkout and equality with live `main`.

Only after that guard passes may the workflow set up Python, install Modal, authenticate, and hand off to Modal. The Modal image itself installs PyTorch 2.10.x before remote benchmark code imports the CORTEX-S module.

## Progression boundary

Repair1 requires fresh full-repo/CPU CI on one exact PR head and an exact-head merge. After merge, live `main` and the fresh repair1 trigger namespace must be audited again before creating exactly one trigger issue. Any failure after the H100 dispatch marker consumes seed `2026090901` and the repair1 namespace; no retry is allowed.

Even `PROMISING_SYSTEMS_CANDIDATE` remains engineering evidence only. It can authorize only a separately reviewed zero-credit production-integration stage, followed by a separately preregistered fresh H100 preflight. It cannot authorize the 2B run or a breakthrough claim.