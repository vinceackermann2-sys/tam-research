# CORTEX-S v0 — 100M / 2B memory-lean grouped v4 preregistration

Status: **zero-credit implementation stage only; no v4 H100 dispatch is authorized by this document or its PR**.

## Evidence carried forward

The scientific comparison remains the same adaptive paired historical-control experiment against repository issue #140. The completed Transformer used seed `8100`, 101,803,520 trainable parameters, context 512, global batch 128, a nominal 2B-token budget and 30,518 full optimizer steps / 2,000,027,648 literal token exposures. Its final NLL was `2.7115590302149455` and reported training throughput was `321151.5755348581 tok/s`.

The exact historical corpus is still frozen by CPU fingerprint issue #767:

- train SHA-256 `93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b`;
- validation SHA-256 `ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760`;
- metadata SHA-256 `14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3`.

All earlier paid/system attempts remain consumed and may not be rerun or reinterpreted.

## Consumed grouped v3 preflight

Issue #802, workflow `34345899535`, job `102447296828`, exact source `3a9538228d100a40818030a64b5b9eaf7311f94a`, and engineering seed `2026090905` are permanently consumed.

The v3 zero-GPU corpus/production gate passed. The one H100 calibration measured the exact 101,778,112-parameter grouped production graph:

- production backend: `physical_padded_grouped_bf16`;
- logical expert width: `338`;
- physical runtime width: `344`;
- compiled execution, `max-autotune-no-cudagraphs`;
- compile time: `366.727893589 s`;
- 40 measured optimizer steps / 2,621,440 tokens;
- measured time: `10.839793787 s`;
- measured throughput: `241,834.8588 tok/s`;
- projected pure training time: `8,270.1063 s`;
- frozen 1.10 overhead + compile projected full envelope: `9,463.8449 s`;
- frozen full-envelope gate: `<= 8,500 s` — **FAIL**;
- peak VRAM: `45.4400 GiB` — PASS;
- `full_run_authorized=false`.

No `[modal-cortex-s-100m-2b-full-v3-grouped]` run was launched. Paired seed `8100` and reserved independent seeds `48131`, `48132`, `48133` remain untouched.

The v3 result is `ENGINEERING_PREFLIGHT_ONLY`. It contains no CORTEX-S 2B language-quality result.

## First-principles v4 change

v3 proved that physical padding plus grouped GEMMs works, but its hot path still performs avoidable memory traffic. v4 changes only how already-required BF16 expert inputs are materialized:

1. **Cast before top-2 gather.** v3 gathers two FP32 token rows per token and then casts the doubled packed tensor to BF16. v4 casts the single normalized token matrix to BF16 first, then gathers the same stable top-2 rows. The grouped GEMM receives the same BF16 values, while the gather moves half as many bytes per selected element.
2. **Cast before physical padding.** v3 pads logical FP32 expert matrices from width 338 to width 344 and then casts the larger matrices to BF16. v4 casts the logical matrices to BF16 first and pads six exact zero channels in BF16. The grouped GEMM again receives the same BF16 values with less temporary traffic.
3. **Lean routing-plan materialization.** The stable expert sort is unchanged. Instead of allocating an expanded token-id tensor and sorted expert-id copy, v4 derives each token id from flattened top-k position `order // top_k` and computes counts directly from the unsorted expert ids. Token/expert assignments, stable row order, weights and offsets are identical.

The architecture is otherwise frozen:

- 101,778,112 trainable parameters;
- d_model 512, 24 layers, 16 heads;
- recurrent state 128/layer;
- 8 experts, top-2 routing;
- logical expert hidden 338, physical grouped width 344;
- 4/24 full-attention layers;
- same optimizer, LR, weight decay, warmup, batch shape, context and corpus;
- same grouped BF16 expert arithmetic;
- no new trainable parameters and no change to routing semantics.

This is a systems optimization hypothesis, not an architecture-quality result.

## Frozen v4 engineering protocol

Fresh engineering seed: `2026090906`.

Consumed engineering seeds that are forbidden for reuse: `910001`, `2026090901`, `2026090902`, `2026090903`, `2026090904`, `2026090905`.

Scientific/control seeds forbidden during v4 preflight: `8100`, `48131`, `48132`, `48133`.

Fresh single-use namespaces:

- preflight trigger: `[modal-cortex-s-100m-2b-preflight-v4-memory-lean]`;
- preflight phase: `preflight-v4-memory-lean`;
- preflight root: `/vol/cortex-s-v0/100m-2b/preflight-v4-memory-lean`;
- full trigger, only if a separately authorized exact-source preflight passes: `[modal-cortex-s-100m-2b-full-v4-memory-lean]`;
- full phase: `full-v4-memory-lean`;
- paired run root: `/vol/cortex-s-v0/100m-2b/paired-seed8100-v4-memory-lean`.

The preflight uses exactly 40 measured production optimizer steps after the existing compile/warmup procedure. It must pass every existing finite-numerics, exact-parameter-count, positive-throughput and <=70 GiB peak-VRAM gate.

The budget gate is **not relaxed**:

- `MAX_PROJECTED_FULL_SECONDS = 8500`;
- projection formula remains `2,000,000,000 / measured_tps * 1.10 + compile_seconds`;
- `HARD_FULL_TIMEOUT_SECONDS = 10000`;
- same frozen pricing snapshot and user-credit envelope.

For orientation only, if v4 compile time were exactly the v3 value, the 8,500-second gate would require about `270,494 tok/s`, roughly 11.85% above v3's measured throughput. This is not a replacement threshold: eligibility is determined only by the frozen full projection formula using the actual v4 measured throughput and compile time.

## Authorization boundary

This implementation and its PR authorize **zero GPU spend only**. Before any v4 H100 allocation:

1. fresh full-repository CI must pass on one exact final head;
2. fresh CORTEX-S CPU gate must pass on that same exact head;
3. fresh 100M/2B zero-credit gate must pass on that same exact head;
4. any merge must use exactly that successful head and only after authorization;
5. live `main` must be audited again after merge;
6. the v4 trigger/result namespace and seed `2026090906` must still be unused;
7. a separate explicit authorization for the bounded H100 v4 preflight is required.

The preflight itself cannot launch full training. Even a preflight PASS only makes a separately triggered exact-source paired run eligible. It does not authorize creating that full-run issue automatically.

No v4 result can by itself support a breakthrough, AGI, SSI, alignment, continual-learning, or independent-replication claim. A completed seed-8100 run, if ever separately authorized, would remain `PAIRED_HISTORICAL_CONTROL_ADAPTIVE_EXPERIMENT` evidence and would still require fresh-seed replication before stronger scientific claims.
