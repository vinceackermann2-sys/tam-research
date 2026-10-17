# PCMA-v1 durable memory checkpoint — 2026-10-17

**Status: Engineering only. Not scientific confirmation. No paid compute, no new scientific attempt.**

## Exact source lineage
This PR stacks on original SI/PCMA neural-two-step draft PR #1507. The full chain is #1507 → #1505 → #1504 → #1503 → #1502 → #1501 → original screenshot-inspired SI draft #1499. The canonical original SI model `tam_research/si_original_hybrid_v0.py`, blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`, must remain unchanged. The old unchanged 64,614-parameter SI's v26 NO-GO is authoritative; consumed/quarantined v25 seed 117214 remains excluded under audit-only #1500. Nothing is merged to `main`.

## What genuinely changed
1. `tam_research/si_pcma_v1_persistence.py` checkpoints the *completed* public factual pair memory and its bounded rank-2 fast residual: first public symbol, actual chosen first action, logical time, observed two-action net reward, factual provenance, adapter coefficients and update budget. It **does not** save unobserved world state, hidden rules, future or unchosen action outcomes, current pending actions or the trained model's weights.
2. Freeze model binding: a SHA256 digest of every deployed frozen neural parameter (name, dtype, shape, raw bytes) is stored. A fresh policy must be initialized with identical trusted neural weights, action continuation, ablation flags, resource limits and adapter settings before it may load online memory.
3. Crash-safe local checkpoint writing: canonical UTF-8 JSON, bounded at **2 MiB**, an unkeyed SHA256 accidental-corruption check and same-directory `os.replace` with fsync for best-effort POSIX durability. Uses NO Python pickle or dynamic arbitrary object construction.
4. Session-boundary-only checkpointing: refuse if the first or second action is unresolved, short-context window is nonempty, or the per-session adaptation counter has not reset. The policy must complete a real session and call `restart_session()` before saving. Checkpoint loading follows the same closed-boundary contract.
5. Strict validation and two-phase restoration: reject invalid record shapes/ordering, duplicate provenance, mismatched exact network checkpoint, invalid fees (where recorded), invalid size, nonfinite rank coefficients, incompatible ablations or unauthorized extra fields **before modifying** the policy. Auxiliary provenance metadata is bounded, while monotonically increasing factual time rejects replay of old events even after memory eviction.
6. Tests: saving after real chosen OOD-world actions, restoring into a *fresh Python interpreter*, checking same memory/action-selection behavior, refusing corrupt payloads, model mismatch, other architecture, bad clock/replayed history, adaptation/retention mismatch and mid-session checkpoint, plus finite bounded storage.

## Scientific status and limitations
- **New demonstrated property sought:** factual PCMA online state can be restored after a *policy process restart*. No model training, no hidden world-state replay and no scientific score are part of this checkpoint. The world process itself must continue operating or an independently reviewed world-side durability mechanism must be provided for a truly restartable complete system.
- **Not an authentication/security protocol:** plain SHA256 verifies accidental corruption, not malicious modification (any writer can recompute it). The checkpoint may reveal user history; permissions `0600` are set on new files but full encryption/key management, signed provenance, rollback prevention across storage replacement and adversarial sandboxing are NOT provided.
- **Not model checkpointing:** pretrained weights must be saved in a separate trusted process and loaded before memory restoration; only their digest is included. Do not load arbitrary neural pickle files.
- **Reset-ablation contract:** after `restart_session(erase_persistent=True)`, the current implementation deliberately refuses exporting a new checkpoint if the policy and adapter clocks disagree; use a fresh policy/session to model an independent reset, or add a jointly agreed reset ledger to a future version.
- **No heldout comparison:** the durability checks use developer-world fixture integers only, never scientific seeds; all OOD evaluation remains blocked from development fitting. Positive engineering tests cannot be cited as evidence of SI performance.

## Required next gates
Before any scientific claim or independent heldout run: frozen attempt manifest with genuinely unused seed reservation and immutable hashes; trained development-only hybrid/Transformer/GRU with a fair compute budget; a higher-compute Transformer control, strong nonprivileged classical policies; a process-separated public-feedback boundary; real value-of-information/no-probe comparisons with world-cluster uncertainty; explicit net reward, memory necessity, post-restart retention, change recovery and worst-family failure thresholds.

**Scientific verdict: original unchanged SI v26 NO-GO; PCMA-v1 still UNKNOWN. No breakthrough or scaling authority.**
