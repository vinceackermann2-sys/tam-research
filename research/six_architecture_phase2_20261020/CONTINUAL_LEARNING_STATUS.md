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

The prospective21 October memory-revision diagnostic now tests the native
path with all three trained62,366-parameter checkpoints. Each receives56 actual
one-step public feedback events across worlds A/B/C, with616 additional raw
before/after positions and zero optimizer updates. All16 A associations remain
correct after learning B/C; A logits are bitwise unchanged. All48 world-scoped
associations initially match their executed transition deltas. After eight A
contexts produce revised outcomes, every revised association still retrieves
its old delta: current A association accuracy falls to8/16 (50%) in every seed.
The other eight A contexts and B/C remain correct. Memory occupancy56 is below
capacity128. Native memory appends duplicate keys and cosine argmax selects the
first tied key; it does not replace the old record for this changing-fact use
case. A latest-record dictionary using the same executed feedback retains the
current deltas by construction. This is a component correctness control, not a
trained neural competitor or proof about action accuracy.

Native weight+fast/memory restoration preserves logits, state and retrieval
exactly, but restores0/56 replay records. A separate explicit full diagnostic
checkpoint restores56/56. External storage at that point is2,688 memory floats,
150 world-fast floats and2,688 replay-latent floats, plus metadata and rewards.
See SI_MEMORY_REVISION_PROTOCOL.md and results/si_memory_revision. Repair
duplicate-key revision semantics and full lifetime persistence before spending
on a large continual-learning claim. These measurements do not test whether
replay subsequently improves decisions or protects an old task during neural
training; the full sequential-task efficacy question remains open.

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
