# Conv7 offline exclusive journal simulation

**Synthetic CPU-only study. No paid-run authorization.** Extends the merged in-memory single-attempt rehearsal into a caller-supplied local sandbox directory. No Modal app, workflow, real reservation, GPU launch, or scientific seed is involved.

The journal creates immutable reservation, model-start and terminal markers using filesystem O_EXCL (create only if absent), file fsync and directory fsync. Interrupted or corrupted markers block further simulated action. Duplicate concurrent reservations, starts, and terminal writes can have at most one winner. The three models retain their exact order: Transformer, original attention-8, then Conv7.

An open STARTED marker after a simulated crash cannot restart. A terminal ERROR prevents all subsequent model attempts. Even complete synthetic traces explicitly deny scientific evidence and paid authorization.

This demonstrates a **local filesystem** concurrency property only. It does not prove remote Modal Volume consistency, atomic cross-worker reservation, backend durability, strict billing caps, or correctness of a live training harness. A real H100 launcher would require separate source/tree/harness binding, a unique fresh engineering seed, exact approval, externally durable reservation before GPU allocation, budget ceiling, real dataset SHA256 verification, retries=0, no-resume, and result verification. The currently consumed scientific seed 60232 is never reused. Nothing in this PR authorizes GPU execution or scaling.
