# Original SI / PCMA-v1 — historical attempt-ledger reconnaissance

**2026-10-18 · Read-only engineering checkpoint · NOT science · No seeds reserved/consumed.**

## Frozen baseline and branch isolation
Stacked only on original screenshot-SI PCMA preflight draft PR #1509 (itself descending from #1508/#1507/#1505/#1504/#1503/#1502/#1501), not any RLT/PGW/CPW/CHM/Conv7 architecture line. Immutable original `tam_research/si_original_hybrid_v0.py` blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` remains unchanged. Original unchanged-hybrid v26 NO-GO and v25 quarantined duplicate 117214 remain authoritative. No `main` changes or merges.

## Problem found in scientific readiness
The first preflight #1509 explicitly listed a few documented consumed IDs, e.g. v25 117211–117214 and v26 118211–118213. Those are NOT an exhaustive history of the original SI experiments. Earlier versions contain successful attempts, timed-out attempts, import/numerical failures, dev-only checks, and prospectively registered replacements that cannot silently be used again.

Examples of historical *claimed* attempted/consumed cases in GitHub's source documents:
- v7: 99611 interrupted with 0-byte result; 99612, 99613 and prospectively registered 99614 are separately documented.
- v8: 100211 interrupted at command timeout; 100212/100213 and replacement 100214 documented.
- v11: 103211 import failure, 103212 evaluation pretraining failure; 103213/103214/103215 documented as later separate valid engineering attempts.
- v23: 115213 timed out without a result, while 115211/115212 and separate extension 115214 are documented.
- v24: 116211 timed out; 116212/116213 were NOT attempted in original panel; separate 116214/116215/116216 extension attempted.
- v25: 117212 produced complete JSON but the child exit status was not captured; **CONSUMED** and not clean. 117214 was subsequently reused improperly and is **quarantined** (do not repair by rerunning).
- v26: development seed118010 and confirmatory 118211/118212/118213 documented as used once; result remains NO-GO.

These statements are *documentary claims*, not an independent attestation to the presence and exit status of every raw attempt. The source documents and associated historical raw evidence must be checked directly before permitting a scientific run.

## Implemented read-only audit
`tam_research/si_pcma_v1_attempt_audit.py` scans every `docs/SI_ORIGINAL_*.md` file classified as a protocol, results document, or addendum. It extracts 4–6 digit numbers from lines mentioning seeds/attempts/replacements/retries/consumption, expands short explicitly printed ranges, and records a document-relative path, line number, Git-compatible blob SHA, and short evidence snippet. This inventory is deliberately **conservative**: a training-budget integer or model step count on the same line may be collected as a **POTENTIAL_SEED_ONLY** false positive. It is never promoted to `consumed=true` merely because of a textual match.

Two independent notions MUST NOT be conflated:
- **Documented potential conflict**: identifier was printed near seed/attempt language in a historical SI artifact. It is blocked from automatic reuse and requires manual source-level adjudication.
- **Proven globally available scientific seed**: would require a separate complete audit of GitHub issues/comments, historical PR heads/failed or canceled CI, all local/remote attempt reservations and raw logs, across *all* research branches/workspaces. The current connector-based document scan CANNOT establish that, nor can any negative grep.

A new-looking integer still returns `BLOCKED_UNVERIFIED`. Both known consumed anchors and unmatched IDs are denied; no method here creates reservations or triggers experiments. The `si_pcma_v1_science_preflight.py` gate can now run this source scan during local review, but **always** withholds execution authority.

## Remaining experimental blockers — no implicit approval
1. Cross-project global consumed/failed/canceled-attempt ledger and independently audited durable reservation facility.
2. Source-complete frozen protocol with training data, independent heldout worlds, predetermined seed IDs, sample and compute budgets, model checkpoints and hashed results destination.
3. Sufficient fair development training for SI hybrid, same-PCMA Transformer and compact GRU, plus a higher-compute Transformer and strongest classical controls. The existing ≤32-step smoke helper is not scientific training.
4. Fair active-training/inference FLOPs and memory/latency profiling beyond partial PyTorch operator estimates.
5. Separate-world-process/public-feedback-only evaluation, causal probe-vs-no-probe experiment and world-level clustered uncertainty.
6. Prespecified ALL-family stop/no-go thresholds and actual user approval. A performance failure must remain a failure; an infrastructure failure must never be counted as an architecture failure or used to rerun a consumed seed.

**Conclusion:** The repository-only attempt audit improves collision detection but is INCOMPLETE. It does not supply scientific seed availability, proof of architectural gain, or permission to launch CPU/GPU research. PCMA-v1 remains UNKNOWN; original unchanged SI v26 remains NO-GO.
