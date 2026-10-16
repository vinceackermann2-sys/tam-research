# PGW-v3 Stage-2B6: immutable corpus, authenticated scoring and fairness blockers

**Track:** PGW-v3/SUBCONAI only. Source-bound research issue #1454.
**Parent:** live main `bb0ef600af7a2a51287f11e5178513187169c225` (after PR #1465).
**Change:** four additive PGW-only files. No source-model, protocol, Modal,
Colab, workflows, other architecture tracks or old branches modified.

## Fail-closed root and identity attestation

The immutable hashes were sealed in issue #1429 and restated in #1454:
- train `0174103c2ef1cf14e5dec0df23498d994062dabd0e4151d02f7d2ba4f055d9a3`
- validation `c6b4fb7bf1943755510bb24fb98053a0ccf45cc83ecb4b51c821193845668611`
- test `3b3ab8e8b5d68337ad1608053541838f02369033b6c95e538a7266b330d8c7d3`
- all `36a40585fe3deb2c4fc268c32eb93092ebc46f7809b772cdbc223d0185d940ef`.

`verify_sealed_corpus(manifest)` reconstructs every sample (5,760 records,
522,240 input tokens), checks 2,880/1,440/1,440 split sizes and 180
factorial cells per split with complete replicate-index sets, plus
row IDs, token and semantic uniqueness, oracle labels and root SHA
recomputation. It **independently hashes the regenerated token bytes**
instead of trusting a generator-provided SHA. The Stage-2B5 sealed SHA
sorts rows and is order-insensitive; this gate *also* rejects manifests
whose traversal differs from the frozen grammar. Neither source-expected
hash nor inputs are rewritten on mismatch. Python RNG state must be
unchanged.

`evaluate_verified_predictions(manifest, split=..., predictions=...)`
runs the complete attestation **before** Stage-2B5's strict
sample-ID-based heldout scorer. It does not perform any model inference,
learning, or batch construction. Direct calls to the old
`evaluate_complete_predictions()` remain untrusted; future actual
scientific consumers must be audited to use the verified wrapper, rather
than claiming a global security gate simply because this API exists.

Negative regressions include tampered answers, IDs, split strata, stale
generator-supplied token SHA after legal filler-token mutation, wrong roots,
permuted records with unchanged old root, and incomplete prediction lists.
These tests are structural checks, not new learned empirical results.

## Fair control readback and outstanding gates

`readonly_research_gate()` compares PGW-hybrid and full-global reference
on Stage-2B4's three delays (1/2/4), using the tiny frozen Stage-1B
gradient fixture. Instantiated parameters: 20,354 vs 20,347. Reports
CE-only and auxiliary-only gradient activity and their union; the union
is **not** equal to combined-loss gradient count, and cannot establish
capacity equivalence. Reports unchanged model weights, protected RNG, and
explicit science blockers.

Still BLOCKED: analytical/actual combined-loss active-parameter parity,
auxiliary prediction information-objective parity, position/causality
controls, measured complete training-step CPU/GPU compute and walltime,
optimization/loss lambda and data-batch-order preregistration, negative
results and fair matched Transformer evaluation, independent replication,
durable one-time seed/job reservations, explicit source-bound user
authorization and capped verified compute funds. Structural synthetic
retrieval cannot justify a breakthrough; exploratory negative results
are not a terminal failure without appropriate controlled evidence.

## Safe test/CI contract

`python -m pytest -q tests/test_pgw_v3_stage2b6_readiness_1454.py`

Includes CPU backward **probes** on frozen random initializations to
measure gradient reachability, never an optimizer step. No GPU/Colab/Modal
or research seeds consumed. Runtime and peak memory are not claimed
until independently measured. No paid execution is authorized.

New isolated branch → reviewable PR → ORIGINAL first-attempt exact-head
CI PASS → live-main path drift audit → SHA-pinned merge → ORIGINAL first
postmerge main CI PASS. If any CI head fails/cancels, freeze it and make
a new successor (never rerun, amend or replay). Any candidate scientific
run requires separate durable source/seed/cost reservation and explicit
owner authorization. Passing this structural gate is NOT permission to
train or scale.
