# Validation snapshot

The phase-two source/cap/equivalence suite passed3 checks on2026-10-20:
actual parameter counts and token-floor arithmetic; causal active convolution
and live gradients; exact scaled CPW output equivalence to the pinned native
source, including future-token isolation. This follows the separate phase-one
51-check validation record. Model efficacy is judged from experiments, not the
unit checks.

AUDIT_CURRENT.json currently verifies39 immutable source references and20
available checkpoint sets, including complete optimizer chunks for the100M-
token A100 Transformer and the first20M-token L4 Transformer segment. Both the
individual chunk hashes and concatenated checkpoint hashes match. Intermediate
L4 checkpoints can later be intentionally superseded; retain their manifests
and pruning records and regenerate the audit for the final snapshot.

The L4 corpus split counts/document digests/token hashes exactly match the A100
corpus. Actual language-model counts are99.56–99.84M. Completed scientific A100
endpoints contain99,999,744 optimizer exposures each; larger confirmation is
still running. The nearest-rounding oversized Conv7 preflight was rejected
before an optimizer update. All checked trained models remain below both caps.

The frozen-state RLT extractor reproduced native terminal logits exactly in
all six integration fixtures. The new probe's erased-carry controls remain at
chance for inaccessible history. Its OOD follow-up failures are included rather
than replaced. SI non-head weights and direct logits are bitwise unchanged by
head repair; its stale-gradient coverage caveat is explicit in execution notes.

Operational checks verified authenticated binary upload/download roundtrips at
64MiB, atomic CLI acknowledgements, and a complete real optimizer backup before
the L4 controller advanced. A runtime restoration plan has been validated in
read-only mode; a full new-VM restoration is reserved for an actual interruption.
