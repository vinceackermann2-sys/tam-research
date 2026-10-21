# SI native memory revision and persistence diagnostic

Prospective component diagnostic, frozen21 October2026 before execution.
Use the unmodified user-selected SIOriginalWorldScopedV2 at source commit
9298c117e736676269107374a402519e45e52f71, and the three already trained
62,366-parameter repair checkpoints, seeds202610301/302/303. No optimizer,
extra policy fitting, search tuning, or checkpoint selection. This is not a
sequential-task action-accuracy or general continual-learning benchmark.

Select16 distinct public five-modality cue histories by advancing the original
_public_batch generator from namespace79,000,000, rejecting exact duplicates
before any model evaluation. The same histories occur in worlds A/B/C.
Execute action0 in a deterministic public one-step fixture, append the actually
observed terminal text100+reward, and call native ingest_executed. First world A
returns reward0 for all16 contexts; world B returns1 and C returns0. Then revise
the environment's reward for the first8 A contexts to1, execute action0 again,
and ingest the new factual outcomes. Each record is an actual fixture execution,
not an unexecuted counterfactual or label inserted into memory. Total56 feedback
events and616 raw before/after modality positions per checkpoint, in addition
to its736,000 prior accounted positions. The fixture is development-only.

At A acquisition, after B, after C, and after A revision, query native factual
memory with the exact frozen backbone state. Compare its returned transition
delta with the latest publicly executed delta for each world/context. Report
exact-association accuracy (max absolute difference<=2e-6), delta error, entry
counts and unchanged-A logits across B/C ingestion. The positive control is a
world/context dictionary retaining the latest delta from precisely the same
executed feedback; it receives no extra outcome. It is an engineering
learnability control, not a parameter/compute-matched neural competitor.

Native memory capacity128 is not exceeded (56 rows), so this does not test
eviction. Duplicate keys and conflicting outcomes are deliberate, to assay a
changing-fact use case within a world. A transition memory may choose other
semantics, so failure does not imply that all episodic-memory uses are invalid.
Fast weights remain active during behavioral roundtrip/isolation comparisons;
no intervention on fast state is presented as causal action utility.

Export native persistent state plus weights and restore a fresh model. Verify
all three worlds' logits/state/retrieval roundtrip within2e-6, and count replay
records before/after native import. Save a separate full diagnostic checkpoint
including replay and restore it explicitly, without changing native APIs.
Report external-state scalar storage and serialized checkpoint hashes.

No pass/fail language-model gate changes. An update failure or missing replay
is a concrete repair target before any claim of continual-learning utility.
Successful storage/isolation/roundtrip remains narrower than acquiring later
tasks, resisting forgetting, or improving decisions.
