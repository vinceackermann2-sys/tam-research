# CORTEX-S v0 engineering audit — 2026-09-08

This is an exploratory CPU engineering record, not scientific evidence.

## Repository provenance

- source repository: `vinceackermann2-sys/tam-research`
- source `main` at branch creation: `9fbd9c523bbcf938cc3751209320f9bb89aa3b3d`
- source tree: `e8b039eab134004982476979f08a981ee461caf9`
- project branch: `research/cortex-s-v0-first-principles`
- project namespace: `architectures/cortex_s/`

No files under the existing TAM/AERA architecture tracks were modified.

## Local no-credit environment

- PyTorch: `2.10.0+cpu`
- accelerator: none
- Modal/GPU credits consumed: none

## Primitive tests

Command:

```bash
cd architectures/cortex_s
PYTHONPATH=. pytest -q tests/test_cpu_primitives.py
```

Result: **6 passed**.

Validated:

- finite forward/backward path;
- exact top-k routing semantics;
- state persists across independent stream calls;
- episodic memory can learn one-shot associations without changing base weights;
- deterministic external safety kernel denies unauthorized authority;
- tiny CORTEX-S/Transformer parameter mismatch is < 0.2%.

## Exploratory recurrent-state benchmark

Command:

```bash
PYTHONPATH=. python state_machine.py --seed 123 --steps 600 --output exploratory_cpu_result.json
```

Matched parameter counts:

- CORTEX-S: 58,437
- Transformer: 58,368
- mismatch: 0.1182%

Both received the same generated operation/value sequence, the same per-step state target, the same optimizer family, the same 600 update count, and the same exploratory seed. Training wall times in this CPU environment were ~9.90 s vs ~9.84 s.

Length-32 final-state accuracy after training only on lengths 4–8:

- CORTEX-S: **99.4141%**
- Transformer: **33.2422%**

This result is intentionally classified **EXPLORATORY_ENGINEERING_ONLY** because the task is structurally favourable to a recurrent state machine, thresholds were informed by exploratory development, and seed `123` is consumed. Seed `124` was also used during exploratory tuning and is consumed.

## What this does and does not establish

It establishes that the proposed persistent-state/sparse-recurrence core is differentiable, trainable, can carry state across calls, can learn a simple latent transition rule, and can coexist with a deterministic external authority boundary.

It does **not** establish improved language modelling, general reasoning, continual neural learning, real sparse compute efficiency, alignment, AGI, SSI, or a breakthrough. In particular, the current expert implementation computes all experts before selecting top-k outputs, so no sparse-throughput claim is allowed.

The next permitted stage is CPU/full-repository validation and preparation of the separately frozen 25M-class paid seed-1 falsification harness. The fresh scientific seeds in `PREREGISTRATION.md` must remain unused until that protocol is frozen.
