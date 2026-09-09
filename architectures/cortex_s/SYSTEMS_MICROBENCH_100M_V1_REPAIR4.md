# CORTEX-S 100M systems microbenchmark v1 — repair4

Status: **preregistered engineering systems experiment only**. This document does not authorize 2B training, scientific replication, architecture freeze, S2, or any breakthrough claim.

## Why repair4 exists

Consumed repair3 issue #791 / workflow run `34335827718` used source `1c0ecd321a530daee32a4603a9cd6fb201be2d8a` and engineering seed `2026090903`.

Repair3 established:

- frozen corpus hashes, byte sizes, metadata, and CPU layout/scoping gate: PASS;
- H100 allocated and the bounded engineering benchmark actually ran;
- semantic routing was exact and loss delta was `0.0019292831420898438` (within the preregistered `0.02` gate);
- legacy compiled full-model path passed at `146554.80865678022` training tokens/s with `48.26404094696045` GiB peak allocated VRAM;
- grouped repair3 failed before measurement with `RuntimeError: Expected mat_a stride along 1 dim to be multiple of 16 bytes, got 338.`;
- the recurrence slice diagnostic was slower than legacy (`0.7018636224128108x`) and is not pursued in repair4.

The repair3 workflow itself ended red only after the authoritative H100 result had been printed and persisted, because the GitHub runner had Modal but not local PyTorch and could not unpickle a torch-owned return type. That post-result transport failure is infrastructure, not scientific evidence. Repair4 normalizes remote returns to stdlib JSON types.

Repair3 and seed `2026090903` are consumed. They must not be rerun.

## Root cause hypothesis

Repair2/3 tried to keep logical expert hidden width `338` while using a padded backing allocation and slicing back to 338. The CPU stride probe saw the intended padded stride, but `torch.compile`/Inductor canonicalized the logical activation and grouped-mm backward still saw a physical row stride of `338` BF16 elements.

Repair4 removes the slice/view trick entirely. It keeps trainable parameters at the exact logical shape but pads runtime matrices to a real BF16-aligned hidden width:

- logical hidden: `338`;
- physical runtime hidden: `344`;
- W1 runtime shape: `[8, 512, 344]`;
- W2 runtime shape: `[8, 344, 512]`;
- six added W1 columns are zero;
- six added W2 rows are zero;
- `GELU(0) = 0`, so the padded channels contribute exactly zero;
- no trainable parameters are added or removed.

This is an engineering systems transformation, not an architecture-quality change.

## Frozen repair4 protocol

- trigger title: `[modal-cortex-s-100m-systems-microbench-v1-repair4]`
- phase: `microbench-v1-repair4`
- result root: `/vol/cortex-s-v0/100m-systems-microbench-v1-repair4`
- engineering seed: `2026090904`
- consumed engineering seeds: `910001`, `2026090901`, `2026090902`, `2026090903`
- forbidden scientific/paired seeds: `8100`, `48131`, `48132`, `48133`
- production shape: microbatch `64`, sequence length `512`, grad accumulation `2`
- tokens per optimizer step: `65536`
- compile-trigger steps: `1`
- additional warmup steps: `3`
- measured steps: `20`
- promising full-model grouped speedup: `>=1.20x`
- stop grouped path below: `1.10x`
- max grouped peak allocated VRAM: `70 GiB`
- max semantic loss delta: `0.02`
- H100 timeout: `15 minutes`

## Execution order

1. CPU-only frozen-data and physical-padding contract gate. No GPU is authorized if this fails.
2. On H100, cache the already-frozen corpus.
3. Run a small **compiled forward+backward grouped-mm operator probe** using physical hidden `344`. If this fails, stop immediately before full-model compilation.
4. Only if the operator probe passes, run semantic equivalence checks.
5. Compile and measure the legacy 100M model at the exact production batch shape.
6. Compile and measure the physical-padding grouped candidate at the same shape.
7. Compute within-run grouped/legacy throughput ratio and apply the frozen gates.

The prior recurrence slice diagnostic is not repeated because repair3 already measured it below 1.0x and repair4 changes only the grouped-MoE path.

## Outcome classification

Possible statuses are:

- `STOP_GROUPED_PATH`: operator/semantic/full-model gate fails, or grouped speedup `<1.10x`;
- `INSUFFICIENT_SYSTEMS_GAIN`: all correctness gates pass but speedup is `>=1.10x` and `<1.20x`;
- `PROMISING_SYSTEMS_CANDIDATE`: all gates pass and speedup is `>=1.20x`.

Even `PROMISING_SYSTEMS_CANDIDATE` is only engineering evidence. It does **not** directly authorize the paired 2B run. A separate source-bound decision is required after inspecting the exact result and remaining budget.

## Non-authorizations

Repair4 does not consume paired seed `8100` or fresh seeds `48131`, `48132`, `48133`. It does not authorize full 2B training, reruns, retries, fresh-seed replication, scaling, SSI/AGI/safety claims, or a breakthrough claim.
