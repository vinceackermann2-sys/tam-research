# Original user SI architecture — revisable world memory (RWM1)

**2026-10-21. Engineering repair only. No science, GPU, heldout data or
superiority claim.**

## Why

The 21 October component diagnostic of the user-selected
`SIOriginalWorldScopedV2` (commit `9298c117`) used three trained 62,366-parameter
checkpoints and 56 executed feedback events each. It found:

- **D1:** revised facts return stale values. Duplicate keys append, and cosine
  argmax returns the first: revised world-A accuracy was 8/16 in every seed.
- **D2:** the native export omits replay: 0/56 replay records were restored.

Reading the code also showed:

- **D3:** memory and replay share one global FIFO bound, so one world's writes
  evict another world's facts.
- **D4:** memory keys ignore the executed action.
- **D5:** any nonempty world always returns its nearest record, even for an
  unrelated context.
- **D6:** stored keys carry no record of the backbone weights that produced them.

The "V3" name is already used by `experiment/si-original-factual-sensor-replay-v3-20261020`
(sensor-level replay gradients). That branch is a different change, which the
study did not select. This repair is therefore named RWM1 and subclasses V2
directly.

## What changes

`SIOriginalRevisableWorldMemoryV1` in
`tam_research/si_original_revisable_world_memory_v1.py` replaces only the
memory and replay **storage objects**. The inherited V1/V2 ingest validation,
backbone, fast-weight rule, forward path and every neural parameter and buffer
are unchanged. State dicts load strictly in both directions between V2 and RWM1.

- **Identity:** a record is identified by (world, action, key). Two keys are the
  same context iff their max absolute difference is ≤ 2e-6. Writing an existing
  identity replaces its value (latest wins).
- **Revision metadata:** each record keeps `writes`, `changes`, `first_write` and
  `last_write`. A bounded per-world audit log stores superseded-value digests
  and is never read by retrieval.
- **Read:** the default is identity-only. A miss returns zeros, and `forward`
  reports `memory_hit`. An optional cosine threshold exists for descriptive
  measurement. When several actions share one context, the action-free forward
  read returns the latest executed observation.
- **Isolation:** each world has its own memory and replay quota and evicts only
  its own oldest-written record. V2's `max_worlds` bounds the shared registry.
- **Replay:** replay stays an append-only log of what happened. It iterates in
  global write order, so `consolidate_verified_replay` selects the same batch as
  V2 when no quota overflows.
- **Lifetime:** `save_lifetime`/`load_lifetime` store weights, fast state,
  memory, audit log, replay, skills, counters and configuration. The file loads
  with `torch.load(weights_only=True)`. Import validates everything before
  mutating anything. It refuses V1/V2 schemas, and it refuses a
  representation-fingerprint mismatch, because stored keys cannot be reindexed
  without raw histories. Saving never overwrites an existing file.

## Mechanism of the ingest hook

V1's `ingest_executed` calls `memory.write(world, prior, delta)` and then
`replay.append(PublicExperience(...))`, and the replay record carries the
executed action and reward. In RWM1, `write` validates and *stages* the memory
write, and `WorldReplay.append` commits the memory write and the replay record
together under one write index. The staged slot is cleared before and after
every ingest, so a rejected event leaves all state unchanged (test T13).

## Tests and limits

`tests/test_si_original_revisable_world_memory_v1.py` (T1–T22) uses only
deterministic untrained fixtures. The inherited V1/V2 tests are unchanged.
The trained-checkpoint diagnostic and its preregistered gates live in the
research package (`research_si_rwm1_20261021`).

Passing these tests is a storage-mechanics result. It is **not** evidence of
continual learning, better decisions, planning or near-match generalization.
Identity matching assumes the same computation path. Batched queries can differ
from single-example keys by float rounding. Provenance is still caller-attested.
