# CORTEX-S v0 100M systems microbenchmark — repair3

Classification: `ENGINEERING_SYSTEMS_MICROBENCH_ONLY`

## Why repair3 exists

Consumed repair2 issue #788 launched exact source `cb4f01233090e7e9fa3c4e3db3915fd5b94480e0` through GitHub Actions run `34329354099`, job `102394032871`.

Repair2's zero-GPU corpus + grouped-mm layout gate passed, then one H100 was allocated. Before either legacy or grouped full-model timing started, the runtime failed with:

`UnboundLocalError: cannot access local variable 'torch' where it is not associated with a value`

The exact cause was Python scoping in `_benchmark_full_model_variant`: a later `import torch._dynamo` statement made `torch` a function-local binding, so the earlier `torch.Generator(...)` access failed. This is an engineering harness failure, not architecture/scientific evidence and not a grouped-mm speed result.

Issue #788, run `34329354099`, repair2 result namespace, and engineering seed `2026090902` are permanently consumed and must never be rerun/reused.

## Repair3 frozen change

Repair3 changes only the benchmark harness scoping around Dynamo. The repair2 grouped-mm candidate, logical expert hidden width 338, parameter count, routing semantics, production batch shape, corpus, timing procedure, and speed thresholds are unchanged.

The benchmark now uses `from torch import _dynamo as dynamo` followed by `dynamo.reset()`. A zero-GPU scoping regression fails closed if `_benchmark_full_model_variant` ever binds `torch` locally again. The repair2 stride/layout regression is also rerun before any H100 allocation.

## Fresh single-use namespace

- trigger title: `[modal-cortex-s-100m-systems-microbench-v1-repair3]`
- phase: `microbench-v1-repair3`
- result root: `/vol/cortex-s-v0/100m-systems-microbench-v1-repair3`
- engineering seed: `2026090903`
- hard H100 timeout: 15 minutes
- measured optimizer steps per variant: 20
- tokens per optimizer step: 65,536
- production microbatch/sequence/grad accumulation: 64 / 512 / 2

Consumed engineering seeds: `910001`, `2026090901`, `2026090902`.

Forbidden scientific/control seeds for this engineering stage: `8100`, `48131`, `48132`, `48133`.

## Frozen engineering classification

The full-model speedup is `grouped_training_tokens_per_second / legacy_training_tokens_per_second`.

- speedup < 1.10x: `STOP_GROUPED_PATH`
- 1.10x <= speedup < 1.20x: `INSUFFICIENT_SYSTEMS_GAIN`
- speedup >= 1.20x with all semantic/parameter/VRAM gates passing: `PROMISING_SYSTEMS_CANDIDATE`

Even `PROMISING_SYSTEMS_CANDIDATE` is engineering evidence only. It cannot directly authorize the 2B run. It may only justify integrating the systems path and then running a separately frozen production preflight.

## Authorization boundary

This document and its PR authorize **zero GPU spend**. Before any repair3 H100 dispatch:

1. exact-head full repository CI must pass;
2. exact-head CORTEX-S CPU gate must pass;
3. exact-head 100M/2B zero-credit gate must pass;
4. the exact successful head must be merged through an explicitly authorized merge;
5. live `main` must be re-audited and frozen to the resulting merge SHA;
6. the repair3 trigger namespace must be confirmed unused;
7. exactly one trigger issue may then be created with the exact source SHA.

No rerun/retry/redispatch of repair2 is allowed. No 2B training function exists in the repair3 launcher. `full_training_authorized` and `next_stage_authorized` remain false throughout this stage.
