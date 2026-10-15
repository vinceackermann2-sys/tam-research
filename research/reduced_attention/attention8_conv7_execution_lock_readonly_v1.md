# Conv7 source-bound review lock (CPU only)

This is a read-only planning artifact, not a preregistration or authority for paid execution. The lock manifest covers 13 exact Git blobs, all three comparator model names/parameter counts, 200,015,872 synthetic planned training exposures per model and the 30,518-step full-science cosine schedule prefix, final shared 819,200-token heldout evaluation, and the frozen expected corpus SHA256 hashes.

The offline validator checks source bytes and cross-validates the lock against the current unapproved proposal, including matching optimizer geometry, the three draft quality/system gates, and the consumed scientific seeds 60231 and 60232. Mutation tests in pytest temporary copies ensure a modified source file, evaluation size, seed, cost ceiling or authorization flag is rejected. The script never creates a GPU job, remote reservation or result artifact.

This contract verifies hash *expectations*, not the real corpus files on Modal storage; it does not prove remote reservation linearizability, backend billing limits, H100 compilation, model quality or wall-clock speed. The forthcoming paid runner must receive its own source/tree/harness review, unique job ID/engineering seed, real durable reservation, strict budget/timeout, and explicit user authorization. Scientific replication and 250M/5B scaling remain unauthorized.
