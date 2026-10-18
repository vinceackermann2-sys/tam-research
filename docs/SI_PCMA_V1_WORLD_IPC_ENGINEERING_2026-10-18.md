# SI PCMA-v1 — development world/process separation checkpoint

**2026-10-18 · Engineering only · No heldout or scientific run · No model advancement**

## Root provenance
Stacks on original screenshot-inspired SI PCMA draft PR #1510, itself on #1509/#1508/#1507/#1505/#1504/#1503/#1502/#1501 and original SI v26. Preserves canonical 298-line source blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`. The unchanged 64,614-parameter original SI hybrid **v26 NO-GO** and v25 invalid repeat of already consumed 117214 remain binding. No changes to main, other architecture branches or their scientific ledgers.

## Problem and new code
Previous `run_ood_sessions` constructed `OODRuleWorld` in the *same Python interpreter* as the policy. Even though policy callbacks received only public observations/actual chosen transitions, a policy could accidentally or deliberately inspect module globals/private instance state and shortcut the task; Python underscores are not a security boundary.

New `tam_research/si_pcma_v1_world_worker.py` isolates the hidden OOD world *instance* in a trusted child Python interpreter. New `si_pcma_v1_world_ipc.py` communicates through strictly schema-checked JSON-line requests/replies:
- Parent starts one child for one *development-only* world. The worker refuses `--split heldout` even outside the parent and does not expose the hidden mapping, latent context, RNG state or training/evaluation oracle.
- Only `start`/observation, `step` with one chosen integer action, `restart` after complete session, and `close` are valid. Unknown/private/extra commands receive constant error codes. Neither peek nor counterfactual outcome is exposed.
- Each reply is a validated public observation or **one executed factual** transition with source/destination public symbols, chosen action, observed reward, actual paid-probe fee, monotonic time, provenance ID and done flag. Parent refuses incorrect action/fees, replay, bad step budget and private extra fields.
- Worker persists its hidden world across development *session* restarts, not process restarts. Session traces match the original single-process world for identical development seeds/actions, including stable, shift, noisy and 16-step long contexts.
- Bounded command and reply sizes, limited per-call timeout, subprocess context cleanup, and stripped nonessential environment variables reduce accidental exposure/resource waste. No science seeds are reserved; developer seed integers in CI are test fixtures only.

## Honest trust and isolation limitations
**Not a malicious-code sandbox**. The current worker is a child process running under the same OS user, so a hostile policy may inspect OS process command lines (including development fixture seed), filesystem source, local process state if permissions permit, or modify code before launch. Both development and heldout rule tables remain present in the public source repository, and the toy mapping has only 6 possible values. This is suitable for detecting API-level privileged-object/feedback leakage and separating normal trusted policy callbacks from live world state, **not** for a strong adversarial hidden-test contest or a security claim. A stronger future runner needs separate OS user/container, secret heldout generator configuration and randomization not disclosed to policy, OS/network/IPC restrictions, independently signed source/checkpoint, and a tamper-evident run ledger.

This implementation is deliberately **development-only** and raises on `OODWorldConfig(split="heldout")` *before* spawning any process. It is NOT a scientific runner and does not return aggregate promotion statistics. The old in-process runner remains for comparison of correctness, but must not be represented as isolated scientific evaluation.

## Independent scientific execution remains BLOCKED
Before any frozen scientific attempt: complete cross-branch/global attempt ledger and unique fresh seed reservations; externally authored unseen rule worlds; strong classical and higher-compute Transformer comparators; locked neural training budgets and checkpoint hashes; equal training/inference compute accounting; actual sensor acquisition causal experiment; world-grouped uncertainty, effect-size and worst-family stop gates. Only after independent review and explicit compute/science approval may a separately designed *scientific* evaluator run.

The development tests do not demonstrate any SI capability, information-acquisition value, robust hidden-law transfer, long-horizon planning or architecture-specific advantage.

**Verdict: PCMA unknown; unchanged original SI v26 NO-GO.**
