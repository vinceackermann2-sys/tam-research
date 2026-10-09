# PGW-v3 Stage-1B: positional-deconfounded retrieval oracle

**Classification:** `PGW_V3_STAGE1B_POSITION_DECONFOUNDED_ORACLE_IMPLEMENTATION_NO_TRAINING`

- Design issue [#1372](https://github.com/vinceackermann2-sys/tam-research/issues/1372)
- Source-bound implementation preregistration [#1374](https://github.com/vinceackermann2-sys/tam-research/issues/1374)
- Original first PR source-main: `395e650e6398c876db90f5f6a76fdf2a15b1b48d` (advanced after issue creation via unrelated CHM-specific additions)
- Branch: `research/pgw-v3-stage1b-positional-oracle-1372-v1`
- Historical Stage-1 oracle: [#1368](https://github.com/vinceackermann2-sys/tam-research/issues/1368), PR #1371, merge `3a9213470fcc04f4130c5a8e56c97d28f2bfd753`, both exact-head and merged-main original CI PASS.

## What changed

Stage-1 fixed the most recent target-key write at chunk index 2. That made position copying a potential shortcut. **This Stage-1B data fixture is not compatible with its previous fixed-position data assumptions**; rather than mutating frozen Stage-1 science, the new data generator lives in `tam_research/pgw_v3_stage1b/`.

Stage-1B uses eight complete event WRITE chunks, then 1/2/4 delay chunks and a final READ chunk. Each chunk is 8 token IDs. All target labels are kept **outside** the model's input stream, which contains only WRITE(key,value), DIST, READ(key), and QUERY commands plus filler. The 8-bit input domain is unchanged. `NOT_FOUND=64` is a valid **answer**, not a token the input stream contains.

The full factorial is:

| Variable | Levels |
|---|---|
| Last tracked write position in first 8 chunks | 2, 3, 4, 5, 6 |
| Older same-key writes | 0, 1, 2 |
| Delay length, completed chunks | 1, 2, 4 |
| Interfering later writes to other keys | yes / no |
| Queried key present | yes / no |
| Data split | train / validation / test |

That is **180 scenario cells per split**, 540 split-cell combinations. `fixture_panel(per_cell=2)` audits **1,080 input examples**.

When the queried key is missing, a *decoy key* has the same historical write pattern (including the varying last tracked position) while the queried key never appears as WRITE. The final event at index 7 always writes a different key. Thus copying chunk 2 or the most recent global WRITE cannot solve all cases; for missing keys the independent reference oracle returns 64.

At least three other keys appear alongside the tracked key. Every WRITE chunk in a generated example has a distinct value ID, eliminating accidental equality between stale/current target writes or competing keys. Delay interference writes target an unrelated key, never the queried key. The final read contains no answer and no future write; completed earlier writes alone determine the answer.

## Integrity, determinism and independent reference

`make_example` uses only the public namespace `PGW_V3_STAGE1B_POSITIONAL_ORACLE_20261009_V1`, split, sample index and factorial cell to derive SHA256 counter-style deterministic draws. It neither accesses nor mutates Python's global PRNG. Samples are hashed by exact token-stream bytes. `assert_panel_disjoint` fails on observed duplicate fingerprints, including cross-split duplicates, and requires valid hashes.

`oracle_latest_value_or_not_found` independently parses the flat stream into causal WRITE updates and returns the last value for the final READ key, or NOT_FOUND. It rejects incomplete chunks, bad key/value/filler/query IDs, future READ placement, invalid commands, and missing READ. Unlike Stage-1, **missing-key READ is intentionally valid**.

Three explicitly naive **algorithmic** controls are frozen and tested against adversarial cases: copy chunk 2, latest WRITE regardless of key, most frequent WRITE value (ties resolved by smallest ID). The strict oracle answers all valid samples by construction. These are **not trained models**, and a pass demonstrates correct data/metadata and flawed-heuristic counterexamples, not useful neural memory.

## Stopping and scientific boundaries

This stage is only an integrity benchmark. It has no embedding model, optimizer, training step, GPU runner, gradient loss or use of scientific seeds. No test here establishes whether PGW-v3 can *learn* key-addressed retrieval, defeat strong baselines, or improve next-token NLL.

Before any CPU learning, separately preregister:
1. A model/token adapter that couples the frozen Stage-0 workspace to token embeddings and an answer head, and a predictor-specific objective if it is to learn (surprise is currently detached and receives no gradient).
2. A verified **full** train/validation/test stream manifest, collision audit, split-wise corpus fingerprint, three fresh science seeds, exact data/token and optimizer budgets, fixed wall-time/compute ceilings, evaluation intervals and preregistered thresholds.
3. **Active**, as well as instantiated, parameter accounting across hybrid, utility-only, surprise-only, recency, random, no-workspace and Transformer baseline arms. No inactive padding can masquerade as model-capacity parity.
4. Gradient/causality tests and task-specific metrics broken down by target position, overwrite count, distractors and delays. All negative endpoints preserved.
5. Separate explicit authorization prior to any training or scientific GPU consumption.

Prior PGW #1181/#1201/#1213 negative mechanisms remain negative and cannot be upgraded by passing an oracle test. No breakthrough, SOTA, scale-up or production conclusion is allowed.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`scale_up_authorized=false`  
`breakthrough_claim_allowed=false`
