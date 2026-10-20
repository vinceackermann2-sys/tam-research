# Continual-learning evidence and scope

Continual-learning advantage is not established for these six architectures.
Memory storage, state carry and an implemented update rule are narrower claims
than learning successive tasks while retaining prior capabilities.

The user's selected SI Original V2 implements world-scoped online fast updates,
factual memory and replay consolidation. On20 October2026, the five native
tests in `tests/test_si_original_world_scoped_v2.py` passed on CPU in8.11s:
own-world fast-state change, cross-world isolation, fast/memory state roundtrip,
invalid-feedback rejection, and bounded capacity/checkpoint validation. These
are deterministic engineering fixtures without an optimizer or scientific
efficacy comparison. Changing a state does not establish a better action.

The earlier three-seed SI pilot exercised factual feedback and replay, but its
development world had no accumulated memory/fast records. Its no-memory/no-fast
ablations therefore do not identify a continual-learning benefit. That pilot
saved neural weights without its external lifetime state. The later planning
head repair deliberately disabled fast adaptation and memory, so its100% finite
cue-task result is not evidence for continual learning. Its persistent objects
were empty. Native `export_persistent_state` exports factual memory and fast
weights but does not include the `replay` buffer; a full lifetime restoration
must save replay and learning/representation version information separately.

CPW's current near100M language arm is sequence-only, without AFM. Its learning
curve cannot validate the separate fast-memory proposal. Prior corrected-binding
controls did not learn reliably, leaving AFM efficacy unresolved. PGW slot
collapse, CHM address/copy failures and bounded-RLT length/overwrite failures
also limit their current memory claims. Conv7's measured language result does
not establish protection against forgetting during later task training.

A positive continual-learning verdict requires a prospective sequential-task
study: measure acquisition on A, introduce B/C, retest A without training on the
evaluation cases, measure conflicting-fact replacement and old-fact retention,
ablate memory/fast updates/replay, and verify complete restart persistence. Use
learnable feedback-matched controls, multiple seeds, counted external-state
capacity and training exposures. No such complete controlled study is claimed
by the current language or planning-head experiments. The present conclusion
is unvalidated continual-learning utility, rather than proof of impossibility.
