# New paired L4 panel after infrastructure interruption

The A100 runtime disappeared after two100M-token endpoints. Both metrics are
retained; only the full-attention optimizer checkpoint was exported. Conv7/CPW
outcomes are unknown, never scored as architecture failures. The missing A100
assignment was confirmed, and an A100 allocation request returned HTTP503. One
L4 was available through the same existing entitlement; no purchase was made.

This is a NEW paired seed202610321 for ALL FOUR arms on the SAME L4, preserving
the selected phase-two equations/counts, source corpus/splits, batch/context,
optimizer and2B-horizon LR. This is not a silent replacement of unknown results.
The A100 evidence remains a separate trial. Same-seed comparisons across arms
use identical batches; native module construction still consumes RNG differently.
Do not compare absolute training TPS between A100 and L4.

Scientific milestones/gates remain100M/300M/1B/2B as in the phase-two protocol.
Every milestone interval is subdivided into <=25M-input-exposure segments,
round-robin across active arms. Segment boundary evaluations are retained but
ONLY the predeclared scientific milestones determine continuation; no best
checkpoint selection. Before each segment, an exclusive reservation must be
saved locally and acknowledged to the runtime. Before the next segment, its
full optimizer checkpoint must be saved locally, each64MiB chunk SHA256 checked
and the concatenated checkpoint hash verified, then acknowledged to the runtime.
An independent keepalive runs during downloads and checkpoint waiting.

If a runtime disappears, charge the whole locally reserved but uncommitted
segment to that lineage's spent exposure counter, then resume ONLY a verified
checkpoint. Replayed data positions count again through this charge. Hard cap:
actual committed optimizer exposures + all lost-segment reserved charges
<=2,000,000,000, and instantiated parameters<=100,000,000. The terminal stage
may consequently end below2B committed updates. Record this explicitly and
compare equal committed budgets where available. No automatic ambiguous retry
without the conservative charge. A fresh seed is a separate disclosed lineage.

Full-source corpus preparation reproduces the prior DATA hashes, including
independent validation/test. L4 must support BF16 and flash SDPA; otherwise stop
before language training. Actual hardware and package versions are recorded.
Tests remain closed until terminal selection. All terminal benchmark/scoring
protocols remain unchanged and require the same hardware for cost comparisons.

Mid-milestone checkpoint binaries may be pruned only after a newer full
checkpoint for the same lineage is verified locally. Retain all scientific
milestone checkpoints, immutable manifests, source hashes and interruption
charges. No weights or raw corpus are committed to Git.
