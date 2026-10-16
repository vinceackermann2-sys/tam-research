# PGW-v3: deterministic traversal is a separate scientific seal

**Class:** CPU-only regression + prospective protocol note, 2026-10-16.
**Parent source:** live main `9afed9493aeedb75f91ab1fac87c9c2086cb50b5`.
**Source of prior immutable roots:** issue #1454, which points back to the
pre-sealed Stage-2B5 roots in issue #1429. Do not revise those roots.

## Distinct claims

Stage-2B5 `_digest()` sorts its serialized record lines before SHA-256.
Thus its four frozen roots seal record **multisets**, not iteration order.
Two sample permutations can share the exact frozen corpus roots. That does
not by itself corrupt scoring: Stage-2B5's held-out evaluator joins
predictions by sample ID and computes aggregate metrics. But physical
traversal order (and batch formation) can change learning trajectories if
future training is ever separately authorized. An original content-seal
PASS must not be misreported as proof of sequence-level reproducibility.

The new additive regression test asserts the exact source grammar traversal
of all 5,760 rows: split; position; overwrite count; interference; delay;
missing flag; replicate index. It verifies the original sealed roots,
unique IDs, and RNG invariance; swapping two rows is a negative control
that preserves Stage-2B5's content digest but changes an independent,
domain-separated, rank-and-ID-sensitive digest. Corrupting a record changes
the frozen content digest. No existing root is rewritten.

The rank-sensitive digest is currently **diagnostic only**. Before a paid
scientific experiment, preregister an exact training sample schedule,
batching and seed policy, source-bound sequence digest(s), data-loader
implementation, and storage of actual batch IDs. Do not elevate the new
digest into an immutable historical seal post hoc.

## What this does NOT close

- Stage-2B6 issue #1454 remains unmerged; independently regenerated token
  bytes/oracles, externally supplied-manifest validation, and verified
  scoring require separate exact-head PR/CI.
- Position masking/long-range causality, information parity of auxiliary
  prediction objectives, combined-loss active-gradient counts, matched
  parameter capacity, CPU walltime and *whole step* compute all remain
  prerequisites to a fair PGW vs Transformer scientific comparison.
- The original 5,760-sample corpus is synthetic; structural checks cannot
  establish learned retrieval, general language value, or a breakthrough.

## Safe validation and governance

`python -m pytest -q tests/test_pgw_v3_stage2b5_sequence_contract_20261016.py`

Only standard-library hashing and the frozen deterministic corpus generator
are used; **no torch model, optimizer, GPU, Colab, Modal, scientific seed or
paid API**. CPU time/memory for this new test must be measured, not
represented by unverified estimates. Preserve original PR-head first CI
attempt; never rerun/amend failed or cancelled heads. Review additive scope
and live main drift before any pinned merge. Treat a passing regression as
an ordering **contract**, not experiment authorization.

**Decision:** PGW remains BLOCKED_PENDING_APPROVED_FAIR_SCIENTIFIC_PROTOCOL.
