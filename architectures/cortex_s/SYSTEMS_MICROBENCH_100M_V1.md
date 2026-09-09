# CORTEX-S v0 100M systems microbenchmark v1

Classification: `ENGINEERING_SYSTEMS_MICROBENCH_ONLY`.

This stage exists because H100 preflight-v2 issue #769 measured only 133,384.4585 train tokens/s and projected a 16,676.53 s full envelope, failing the frozen <=8,500 s budget gate. The 2B scientific run was not launched. Calibration seed `910001` and the preflight-v2 namespace are consumed and must never be reused.

## Immutable ancestry

The candidate work merged through PR #775. The branch for this preregistration starts from merge SHA `5c53d405af9d76e20a1a3e10979fe3c88380157d`. Any later GPU dispatch must execute one exact merged source SHA containing this protocol and must record that SHA before allocation.

## Engineering seed and namespace

- engineering seed: `2026090901`
- trigger title: `[modal-cortex-s-100m-systems-microbench-v1]`
- Modal app: `cortex-s-v0-100m-systems-microbench-v1`
- result root: `/vol/cortex-s-v0/100m-systems-microbench-v1`
- one H100 dispatch maximum; a durable dispatch marker is written before benchmark work
- reserved scientific seeds `48131`, `48132`, `48133` are forbidden
- paired scientific seed `8100` is forbidden
- consumed engineering seed `910001` is forbidden

The engineering seed was chosen in a new namespace after the consumed v2 calibration. GitHub code-search was temporarily unavailable while preparing this branch, so static tests additionally forbid all previously known CORTEX-S scientific/engineering seeds. No GPU execution is authorized merely by merging this protocol.

## Frozen corpus and production shape

Before any H100 allocation, the launcher re-verifies the same persistent corpus contract used by the 100M comparison:

- `/vol/data/tam100m-2b-curated-v1`
- train SHA-256 `93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b`
- validation SHA-256 `ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760`
- metadata SHA-256 `14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3`
- microbatch 64, sequence length 512, grad accumulation 2
- exact 101,778,112-parameter CORTEX-S 100M configuration

The microbenchmark is not language-quality evidence. Corpus use is only to reproduce realistic production batch/gather shapes.

## Primary comparison

Two full 100M training graphs are initialized from exactly the same engineering seed and consume the same engineering batch stream:

1. `legacy`: current production `TrulySparseMoE` path under the existing BF16 autocast and compile settings.
2. `grouped_bf16`: systems-only candidate with identical router and expert values but expert matrices stored in grouped-GEMM-friendly orientation. Selected expert rows and expert weights are explicitly cast to BF16 for `torch.nn.functional.grouped_mm`. The first grouped result remains BF16 through GELU and the second grouped GEMM, matching the legacy autocast expert path more closely and avoiding an unnecessary FP32 intermediate. Route weighting/scatter may promote back to the residual dtype exactly where ordinary PyTorch promotion requires it. The casts stay in autograd so FP32 master parameters receive gradients.

The candidate must continue to execute exactly top-2 of 8 expert assignments per token. Dense execution is prohibited.

Each full-model variant gets one compile-trigger step, three additional warmup optimizer steps, then 20 measured optimizer steps. Compile time, measured train tokens/s, finite loss/gradients, and peak allocated VRAM are recorded separately. There is no eager fallback for the primary grouped result: a grouped compile/runtime failure is a systems failure for this candidate, not permission to change the protocol in-place. Compile, warmup, or measured-step failure explicitly releases model/optimizer/compiler references and clears the CUDA allocator cache before any later diagnostic in the same bounded allocation.

## Secondary diagnostics

Within the same already-authorized H100 allocation, benchmark the recurrent affine-scan component at `[batch=64, sequence=512, state=128]` for production `affine_scan` versus the allocation-reduced `affine_scan_slice_candidate`, including forward+backward timing and numerical/gradient checks. This diagnostic cannot by itself authorize production integration.

## Gates

A result is `PROMISING_SYSTEMS_CANDIDATE` only if all are true:

- exact CORTEX parameter count remains 101,778,112;
- grouped BF16 runtime and backward succeed with finite loss/gradients;
- top-k routing semantics remain top-2/8 with no dense expert evaluation;
- pre-training semantic probe has absolute loss delta <= 0.02 between identical legacy/candidate initializations;
- grouped full-model throughput / legacy full-model throughput >= **1.20x**;
- grouped peak allocated VRAM <= 70 GiB.

If speedup is <1.10x, classify `STOP_GROUPED_PATH`. If it is >=1.10x but <1.20x, classify `INSUFFICIENT_SYSTEMS_GAIN`. A >=1.20x result is still only engineering evidence; it does not authorize the 2B run and does not establish that the <=8,500 s gate is reachable.

## Spend and progression boundary

The H100 function timeout is capped at 15 minutes. The benchmark app contains no full-training function and cannot dispatch the paired 2B run. A GPU run requires, after this code is complete:

1. fresh CPU/full-repo CI on one exact PR head;
2. merge of that exact head through an authorized PR;
3. a fresh audit of `main` and the trigger namespace;
4. explicit authorization for the one engineering H100 dispatch;
5. one newly created trigger issue with the exact merged source SHA.

If the grouped candidate is promising, integration into the production graph is a separate zero-credit PR followed by a fresh preflight with another unused engineering seed. Seed `2026090901` is consumed by the first H100 dispatch regardless of PASS/FAIL.