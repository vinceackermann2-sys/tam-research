# PGW-v3 Stage-1 — deterministic retrieval oracle, no training

Issue: [#1368](https://github.com/vinceackermann2-sys/tam-research/issues/1368)  
Proposal: [#1354](https://github.com/vinceackermann2-sys/tam-research/issues/1354)  
Stage-0: [#1359](https://github.com/vinceackermann2-sys/tam-research/issues/1359), implementation PR #1363, merge `288c3e86d92c6e84c656d604405f765ec64f6f43`, merged-main CI #37896660270 PASS.  
Pinned Stage-1 original source-main: `b35851896d56835b4f29fdfca13ddd1603592f1d`.

**Purpose:** establish a validated, deterministic input-prefix generator and separate causal answer oracle for a later PGW learning experiment. This is **not a neural model test**. There is no PyTorch dependency in the new data module, no optimizer, no gradient descent, no Modal, no GPU, and no new scientific seed.

## Grammar and exact sample structure

Each example is a flat sequence of 8-token chunks:

- `WRITE=1`, `READ=2`, `DIST=3`, `QUERY=4`;
- key token IDs 16–23 (8 keys), value IDs 32–63 (32 values), filler IDs 160–255;
- `[WRITE, key, value, filler×5]`;
- `[DIST, filler×7]`;
- `[READ, key, QUERY, filler×5]`, with the READ value withheld and supplied only as an external target.

Exactly three initial WRITE chunks are followed by `d ∈ {1,2,4}` complete distractor chunks, then a final READ chunk. The queried key has `w+1` distinct historical write values where `w ∈ {0,1,2}`; its final write is at prefix chunk index 2 (zero-based). When interference is enabled, distractor chunks write unrelated keys, never the queried key. When disabled, they contain `DIST` and filler only.

All 18 `3 overwrite counts × 2 interference modes × 3 delays` cells exist for each of three splits (`train`, `validation`, `test`). Samples are deterministically derived from namespace `PGW_V3_STAGE1_ORACLE_20261009_V1`, split name, fixture index and cell via SHA256, without mutable RNG. The generator computes the expected answer directly from its final target-write construction; a **separate strict interpreter** parses the entire valid causal prefix to independently recover the last written value. Both must agree.

## Correctness/ablation checks

The tests exercise the entire 54 split/cell combinations; they check exact length, symbols, the latest-write value, unique old/new target values, lack of answer token in the READ suffix, and strict malformed/unbound query rejection.

Counterfactual checks: mutating a stale target-key value does not change the oracle; mutating the latest target write does change the oracle; changes to unrelated-key writes, filler and future-after-query filler do not affect the answer; changing the queried key follows the last write to that *other key* where it exists.

`fixture_panel(per_cell=8)` includes 432 examples and validates unique sample fingerprints across the observed panel. **This is not a mathematical promise of zero collisions across an arbitrarily large future training dataset.** Future run-control must audit *the complete actual corpus*, fail closed on cross-split duplicates and preserve exact manifest hashes. Fixture IDs are **not** trainable-model scientific seeds.

## Important scientific limitation — avoid a positional shortcut

In this first *integrity fixture only*, the latest target-key write is **always in chunk #2**, irrespective of overwrites or distracting information. A trivial position-copying model could exploit that fact without implementing genuine key-addressed retrieval. Accordingly, **accuracy on this fixture must not be used as evidence of key-value memory or PGW-v3 utility**.

Before any CPU training or comparative claim, independently preregister a **positional-deconfounded v2 generator** that varies the queried key's last write position, interleaves both target and nontarget writes, introduces missing-key and changed-query-key controls, and tests deliberately adversarial recency baselines. Freeze non-overlapping train/validation/test partitions, active-parameter equality, optimizer, objectives, metrics, thresholds, data mixture, sample budget and three fresh scientific seeds before inspecting training outcomes.

## Scope and hard stop

This PR adds only:
- `tam_research/pgw_v3_stage1/__init__.py`
- `tam_research/pgw_v3_stage1/oracle.py`
- `tests/test_pgw_v3_stage1_oracle_1368.py`
- `docs/pgw_v3_stage1_oracle_1368.md`

No historical PGW-v0/v1/v2/core scientific evidence changes. Frozen mechanism results #1181 and #1201/#1213 remain negative, even if these oracle tests pass. A correct deterministic interpreter is necessary test infrastructure, **not** proof of architecture quality.

`cpu_training_authorized=false`  
`gpu_authorized=false`  
`scientific_seed_consumption_authorized=false`  
`scale_up_authorized=false`  
`breakthrough_claim_allowed=false`
