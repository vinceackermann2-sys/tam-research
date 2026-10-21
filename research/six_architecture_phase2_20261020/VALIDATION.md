# Validation snapshot

Final21 October language confirmation is complete. The full audit at12:08UTC
passed95 immutable source references and29 retained checkpoint sets. Maximum
actual language-model size is99,837,440; maximum committed exposure count is
299,999,232 and maximum committed-plus-conservative-loss total is322,224,128.
All reservation digests/bounds and the single idempotent parent-loss ledger
reconcile. Conv7's300M failed gate is independently recalculated from recorded
NLL/throughput/parent metrics, and no later training stage was released.

TERMINAL_OPTIMIZER_AUDIT.json independently reconstructs all four final optimizer
files, verifies individual chunk and concatenated hashes, all finite model/Adam
tensors, every Adam step, RNG presence, exact frozen-constructor parameter counts
and seed/exposure equality. These checks perform zero optimizer updates.
Terminal test contains2,096,640 target positions per model; the unused partial
block in the2,097,152-token array is disclosed in EXECUTION_NOTES.md.

Terminal inference and PIQA1,838/LAMBADA5,153 are complete for all four models.
The analysis checks checkpoint identities/budgets, source and pinned task hashes,
PIQA normalization/aggregate accuracy, task-row alignment and all timing settings.
Only equal committed-budget models receive paired task/test comparisons; no
post-hoc terminal score changes the frozen training gate. Five reporting
regressions passed again in0.44s. See TERMINAL_SUMMARY.json, AUDIT_CURRENT.json
and the final remote source/result/claim audit under l4_panel.

The earlier validation history below records checks performed during the study.

The phase-two source/cap/equivalence suite passed3 checks on2026-10-20:
actual parameter counts and token-floor arithmetic; causal active convolution
and live gradients; exact scaled CPW output equivalence to the pinned native
source, including future-token isolation. This follows the separate phase-one
51-check validation record. Model efficacy is judged from experiments, not the
unit checks. Five additional reporting checks pass: intermediate checkpoints
cannot enter scientific comparisons; seed/device mismatches are excluded;
different terminal budgets cannot receive a paired test comparison; and lost
reservation charges and token-block alignment are enforced. Current gates stay
separate from the interrupted trial, and training completion alone cannot mark
the full language confirmation complete before terminal benchmarks finish.

The15:26UTC AUDIT_CURRENT.json snapshot verifies58 immutable source references
and23 available checkpoint sets, including complete optimizer chunks for the
100M-token A100 Transformer and all FOUR100M-token L4 models. Both the
individual chunk hashes and concatenated checkpoint hashes match. Intermediate
L4 checkpoints can later be intentionally superseded; retain their manifests
and pruning records and regenerate the audit for the final snapshot.

The L4 corpus split counts/document digests/token hashes exactly match the A100
corpus. Actual language-model counts are99.56–99.84M. Completed scientific A100
and L4 endpoints contain99,999,744 optimizer exposures each. Conv7 passed the
100M gate, CPW local failed, and the three-model300M stage was then running. The
nearest-rounding oversized Conv7 preflight was rejected
before an optimizer update. All checked trained models remain below both caps.

The frozen-state RLT extractor reproduced native terminal logits exactly in
all six integration fixtures. The new probe's erased-carry controls remain at
chance for inaccessible history. Its OOD follow-up failures are included rather
than replaced. SI non-head weights and direct logits are bitwise unchanged by
head repair; its stale-gradient coverage caveat is explicit in execution notes.

Operational checks verified authenticated binary upload/download roundtrips at
64MiB, atomic CLI acknowledgements, and a complete real optimizer backup before
the L4 controller advanced. A runtime restoration plan has been validated in
read-only mode; offline recovery fixtures verify that an uncommitted reservation
is charged once, its full details and orphaned result remain archived, and a
corrupt recorded durable checkpoint blocks recovery. The cap audit now checks
reservation digests/bounds and reconciles lost-reservation ledgers with lineage
charges. A full new-VM restoration is reserved for an actual interruption.

Five native SI V2 world-scoped regression checks also passed in8.11s during the
continual-learning follow-up. They verify update/isolation/serialization
mechanics, not retained task improvement. Private CLI process fixtures passed
deadline/descendant cleanup and success-marker checks after Windows inherited
PIPE handles stalled an exporter. The current watcher uses bounded file-backed
CLI output, per-digest ACK deduplication and an independent keepalive. Training
sources and protocols did not change during those transport repairs.
