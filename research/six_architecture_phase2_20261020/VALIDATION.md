# Validation snapshot

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
100M gate, CPW local failed, and the three-model300M stage is running. The
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
