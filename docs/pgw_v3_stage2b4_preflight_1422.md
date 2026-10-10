# PGW-v3 Stage-2B4 — derivative capacity and observed CPU walltime preflight

**Classification target:** `PGW_V3_STAGE2B4_CPU_DERIVATIVE_AND_RESOURCE_PREFLIGHT_READY` **only after original exact-head CI and original merged-main CI PASS.** This is **not model training, replication, active-matched capacity, equal compute, learned retrieval or breakthrough evidence**.

## Immutable lineage and scope

Frozen implementation [#1422](https://github.com/vinceackermann2-sys/tam-research/issues/1422) follows design [#1417](https://github.com/vinceackermann2-sys/tam-research/issues/1417). PGW Stage-2B3 [#1411](https://github.com/vinceackermann2-sys/tam-research/issues/1411) / PR #1414 merged `61f57a702c7cd02019397b2631ac052d4098a1a1`, original CI #38072261210 and merged-main CI #38072621912 PASS. Implementation freeze main `2057ebcefede8e72cdf117a19deba4ce2af64d28`; branch `research/pgw-v3-stage2b4-cpu-capacity-timing-v1` created from audited live main `33f9e159fb801d7fd55cbc51a522fc28ba6deaf4` after unrelated reduced-attention/CORTEX additions. Other research, CI, frozen PGW Stage-0→2B3 modules, and negative PGW-v2 #1181 / PGW-core #1201/#1213 remain unchanged.

## Actual tested arms and fixtures

All inputs come from the existing verified Stage-1B oracle, using one present-key and one absent-key query (B=2) at each of the three completed-chunk delay lengths:

| Delay chunks | Full input tokens |
|---|---:|
| 1 | 80 |
| 2 | 88 |
| 4 | 104 |

All inputs contain eight completed WRITE chunks, controlled overwrites and unrelated interference, followed by delay and a final READ/QUERY. The answer labels are external, independently oracle-verified and never appended to input. This is **tiny structural fixture coverage**, not a future full scientific training corpus.

<escape>Arms:</escape> `pgw_hybrid`, `pgw_utility_only`, `pgw_surprise_only`, `pgw_fixed_random`, `pgw_recency`, `pgw_no_workspace`, `full_no_global`, `full_global`, and `chunk_only`.

The six PGW arms have identical initial parameter tensors copied from one hybrid model (20,354 instantiated per arm). The two full-causal Transformer variants and the chunk-only negative control have identical initial parameter tensors copied from one frozen Stage-2B2 reference (20,347 instantiated per arm), with the Stage-2B3 global-position variant adding **no parameters**. PyTorch RNG is forked for structural fixture initialization and restored; local value 1422 is not a scientific seed.

## Structural gradient measurements

`gradient_activity_preflight()` executes one **independent** answer-only cross-entropy `backward()`, clears gradients, then one **independent** auxiliary predictor `backward()` for each of all 27 arm/delay combinations. Neither loss has an auxiliary coefficient or optimizer. The auxiliary loss reuses the frozen Stage-2B0 predictor objective on the eight WRITE chunks; its representation differs by architecture.

For each phase, report parameter elements where `grad is not None`, elements with nonzero finite gradient, and nonzero counts by predictor, utility, other workspace, and non-workspace grouping; also compute the union over the two probes. Model `state_dict` is checked byte-for-byte unchanged after both backward passes. These counts constitute **empirical lower bounds for those particular inputs and random initial weights**, not analytical active parameter capacity or a matched-control certificate. Equal nominal counts do not prove fair complexity.

## Measured CPU timing diagnostic

`cpu_derivative_timing_preflight(arm,delay,warmups=3,repeats=10)` is a manually callable, **no-optimizer** timing utility, with bounded warmups 0–5 and repeats 1–20. It measures two separately defined CPU phases with Python `time.perf_counter_ns` and reports median and nearest-rank p95 milliseconds:

- **Forward answer:** `torch.no_grad(): model(tokens, anchors)`.
- **Forward + answer loss + backward:** the complete `F.cross_entropy(model(tokens,anchors), target).backward()` derivative computation. It is **not** a backward-only timing and deliberately includes an additional answer forward.

Batch B=2, CPU, model.eval, float32. `torch.set_num_threads(1)` is temporarily applied and restored. Model parameters are verified unchanged; PyTorch/Python/platform identity is recorded. The CI smoke tests invoke two selected arms (`pgw_hybrid` and `full_global`) at delay1 with **0 warmups/2 repeats**, solely to ensure the timing implementation works; those noisy, low-sample smoke measurements are **not** publishable per-architecture benchmark numbers. The full default timings are available as bounded preflight operations but are NOT automatically run for all nine arms or three lengths by CI. This is deliberate to prevent large and unreviewed CPU runs.

**Memory:** native PyTorch peak resident tensor memory is **unknown / not measured**. The implementation does not invent a number from `tracemalloc` or cumulative process-RSS high-water. There is no CUDA/profiler GPU metric.

**Compute:** frozen Stage-2B3 `attention_operation_counts` accompanies each row with ideal/dense attention QK/AV pairs and PGW write/read comparison counts. Those are **not whole-model FLOPs** and are never substituted for measured CPU timings. Projection/FFN/normalization/softmax, sequential recurrent overhead and train-step FLOPs are excluded from pair counts. Timing is machine-specific and can be dominated by Python/OS noise.

## Hard scientific stop

Passing CI permits only a zero-training resource/gradient preflight report. Before any actual CPU model optimizer step, a separately preregistered and owner-authorized experiment must freeze full collision-audited train/validation/test corpora and SHA256 manifests; true active-parameter/compute matching or explicit unmatched-control labels; consistent auxiliary lambda, optimizer, batch and token schedules; CPU measurement environment and equal budgets; three independent paired scientific seeds with durable reservations, retries=0/no resume; preregistered present-only/latest-write exact-match gains, per-cell performance and missing-only checks that defeat the trivial 50% always-NOT_FOUND baseline. No H100/Modal paid authority arises from this stage.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`breakthrough_claim_allowed=false`  
`scale_up_authorized=false`
