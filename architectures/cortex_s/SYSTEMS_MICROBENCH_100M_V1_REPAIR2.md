# CORTEX-S 100M systems microbenchmark v1 — repair2 preregistration

## Classification

`ENGINEERING_SYSTEMS_MICROBENCH_ONLY`

This document does **not** authorize 2B training, scientific-seed use, architecture freeze, SSI/AGI claims, or any breakthrough claim. It preregisters at most one later, explicitly triggered, bounded H100 systems measurement after fresh zero-credit CI and an exact-head merge.

## Consumed evidence that must not be reused

The original microbenchmark trigger issue #781 / workflow run `34324639156` failed before Modal because the GitHub guard imported CORTEX-S before PyTorch existed. That trigger namespace is consumed and may never be rerun.

Repair1 trigger issue #784 / workflow run `34326234104` used immutable source `f2869318ac9668b8255ad548afa82ab513c0a024`. Its zero-GPU corpus fingerprint gate passed, then its one bounded H100 dispatch actually started. The semantic probe failed before throughput timing at:

`F.grouped_mm(...) -> RuntimeError: strides should be multiple of 16 bytes`

Therefore repair1 engineering seed `2026090901`, issue #784, run `34326234104`, Modal/result namespace `/vol/cortex-s-v0/100m-systems-microbench-v1-repair1`, and the repair1 trigger title are consumed and must never be retried or reused. This failure is engineering implementation evidence only; it says nothing positive or negative about CORTEX-S language-model quality.

Calibration seed `910001` is also consumed by the earlier guarded systems/preflight work. Scientific/paired seeds `8100`, `48131`, `48132`, and `48133` are forbidden for this engineering probe.

## Root cause derived from the installed PyTorch grouped_mm contract

The failed repair1 W1 tensor had logical multiplication shape `[expert, 512, 338]` and contiguous BF16 stride `[512*338, 338, 1]`. Its row stride was `338 * 2 = 676` bytes, not a multiple of the grouped GEMM 16-byte alignment requirement.

Repair2 changes **physical layout only**, not logical dimensions, values, router semantics, expert count, top-k, or trainable parameter count:

1. W1 is stored in native Linear orientation `[expert, 338, 512]` and passed as a transpose view `[expert, 512, 338]`. The transpose has strides `[338*512, 1, 512]`; the non-unit matrix stride is `512 * 2 = 1024` BF16 bytes and is 16-byte aligned.
2. PyTorch grouped_mm creates padded-stride output storage, but GELU materializes logical width 338 contiguously. That would recreate a 676-byte row stride before W2. Repair2 therefore pads only the **physical activation storage** to width 344 and slices back to logical width 338, yielding stride `[344, 1]` = 688 bytes per row while preserving exact logical values and gradients.
3. W2 stays logical `[expert, 338, 512]` contiguous; its row stride is `512 * 2 = 1024` bytes and is already aligned.
4. FP32 master parameters remain the same logical shapes/numel as the production MoE. No dummy parameters, hidden-width change, expert-count change, or scientific architecture change is allowed.

The zero-credit implementation includes a Python mirror of the grouped_mm 16-byte stride predicate and must reproduce the repair1 invalid layout while accepting all repair2 layouts before any H100 can be considered.

## Fresh engineering namespace

- engineering seed: `2026090902`
- trigger title: `[modal-cortex-s-100m-systems-microbench-v1-repair2]`
- phase: `microbench-v1-repair2`
- Modal app: `cortex-s-v0-100m-systems-microbench-v1-repair2`
- result root: `/vol/cortex-s-v0/100m-systems-microbench-v1-repair2`
- maximum H100 function timeout: 15 minutes
- H100 dispatches allowed in this namespace: **one**
- 2B/full-training functions in this app: **none**

Before choosing the seed, issue search and commit-message search found no prior `2026090902` use. GitHub code search was temporarily unavailable, so the code itself additionally declares `2026090901` and `910001` consumed and rejects all known scientific seeds. Any evidence of prior `2026090902` consumption discovered before a paid dispatch invalidates this preregistration and requires a new seed/namespace rather than a retry.

## Frozen corpus gate

Exactly the historical 100M/2B corpus must be mounted at:

`/vol/data/tam100m-2b-curated-v1`

Required fingerprints:

- `train.bin` SHA-256: `93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b`
- `val.bin` SHA-256: `ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760`
- `meta.json` SHA-256: `14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3`
- train tokens: exactly 2,000,000,000 uint16 GPT-2 tokens
- validation tokens: exactly 5,000,000
- corpus assembly version: 3
- corpus seed: 8100 (data construction metadata only; it is not used as the repair2 model/benchmark seed)

The zero-GPU function must also PASS the exact stride-layout regression before any H100 allocation.

## Frozen systems comparison

The repair2 H100 benchmark remains a production-shape engineering comparison, not a language-quality experiment:

- same 100M CORTEX-S configuration and parameter count as the existing 2B protocol;
- sequence length 512;
- microbatch 64;
- gradient accumulation 2;
- 65,536 token exposures per optimizer step;
- one compile/first-step trigger;
- three additional warmup steps;
- 20 measured optimizer steps per full-model variant;
- BF16 grouped GEMMs;
- legacy production sparse-MoE implementation versus stride-aligned grouped-mm candidate;
- identical repair2 seed and minibatch stream for the two variants.

The candidate must preserve exact top-k routing, logical expert values, total trainable parameter count, and production hidden width 338.

## Gates

Before timing, semantic probe must satisfy:

- exact top-k router choices;
- absolute pretraining-loss delta <= `0.02` between legacy and repair2 candidate;
- exact total trainable parameter count;
- the repair2 grouped-mm layout contract passes.

Systems result gates:

- both legacy and grouped candidate complete compile/warmup/measurement with finite loss;
- grouped candidate peak allocated VRAM <= 70 GiB;
- grouped full-model training throughput / legacy throughput >= `1.20x` for `PROMISING_SYSTEMS_CANDIDATE`;
- speedup < `1.10x` => `STOP_GROUPED_PATH`;
- `1.10x <= speedup < 1.20x` => `INSUFFICIENT_SYSTEMS_GAIN`.

Any runtime/compile/semantic/layout failure is an engineering STOP. It must not be retried with seed `2026090902` or the repair2 namespace after H100 dispatch has begun.

## Authorization boundaries

This branch/PR authorizes **zero paid compute**. Before merge, fresh full-repo CI, the CORTEX-S CPU gate, and the existing 100M/2B zero-credit gate must all pass on one exact final head. The repair2 CPU regression itself must be included in those test runs.

After an explicitly authorized exact-head merge, live `main` and the repair2 trigger namespace must be audited again. Only a separately explicit continuation beyond that boundary may create one repair2 trigger issue. Once the H100 function starts, seed `2026090902` and the repair2 namespace are consumed regardless of PASS, FAIL, timeout, or infrastructure error after allocation.

Even a `PROMISING_SYSTEMS_CANDIDATE` result **cannot authorize the 2B run**. It only permits a separately reviewed zero-credit production-integration step and then a new exact-source preflight under the broader 100M/2B governance protocol.
