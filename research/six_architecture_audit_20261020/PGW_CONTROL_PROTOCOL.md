# New finite PGW learnability control

Frozen before execution. The first screen's one-layer full-attention control
failed retrieval. At its QUERY anchor the attention query is computed from a
fixed marker embedding; prefix-value attention cannot directly condition its
relative allocation on the preceding query key in that same layer. This is a
weak control for multi-token key/value binding, not an impossibility theorem.

Use a fresh three-layer, width64, heads4, FF256 causal Transformer with learned
absolute positions, no auxiliary loss. This is a **strong positive control**,
not a parameter/compute-matched architecture superiority baseline. For each new
seed202610231/232/233:5000 updates, batch24, AdamW .001, wd.01, clip1, fixed LR.
Native position/overwrite/interference/missing factors cycle in balanced batches;
delays1/2/4 cycle per update. New training indices start20,000,000; development
indices21,000,000 use three complete factor repetitions per delay.

All three strong controls must achieve >=80% present retrieval AND >=80% missing
rejection at every delay at the fixed final update. Only then train the unchanged
native PGW hybrid on the identical examples/update budget and answer CE alone.
Otherwise stop the candidate extension as inconclusive. This changes the first
screen's objective by removing auxiliary targets from every new arm. It does not
retroactively repair or replace the first screen. Native inactive predictor
capacity and model-size/compute differences must be disclosed.

Record query swaps separately: for each fixed development prefix, vary the READ
key across all eight legal keys and derive targets with the strict public oracle.
All-eight accuracy is a correlated prefix-level diagnostic, not eight independent
examples. No tuning, retry, resume or selected checkpoint. Save every attempt,
source/data/checkpoint hashes, actual parameter/exposure count, wall time and
gradient coverage. Hard ceilings100M parameters/2B exposures and15 minutes/model.
These are new development cases, not an independent confirmation or breakthrough.
