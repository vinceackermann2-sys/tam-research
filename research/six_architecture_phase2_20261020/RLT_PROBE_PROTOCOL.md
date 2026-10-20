# Frozen-state readout probe after the retention screen

This is a new prospective diagnostic prompted by observed state separation in
the failed long-distance models. It is not part of the original retention gate
and cannot turn that original failure into an unmodified-model success.

Use all six completed native/public-prior checkpoints, frozen completely.
Generate512 examples at each length128/256/512/1024 for TRAIN (namespace76M),
and512 per length for fresh EVALUATION (namespace77M). Each uses256 independent
filler prefixes, paired opposite initial SET bits and a final QUERY. Shuffle
rows independently of their labels before feature extraction, avoiding any
batch-row numerical parity shortcut. Current-chunk encoder inputs after the
first chunk are identical within each pair.

Extract the actual final32-dimensional recurrent state. The heavy decoder has
no feedback to this bounded branch's carry, so previous-chunk decoder outputs
can be omitted when collecting states; verify terminal-logit equivalence to the
native full forward on an integration fixture. Fit a fresh Linear(32,2) head
after per-feature TRAIN-only standardization (std clamped at1e-6),1000 full-batch
AdamW updates, LR.03, no decay. No backbone, original decoder, or normalization
weights change. Seed=model seed+1337. No tuning or checkpoint selection.

Compare the frozen original decoder and fitted state-only probe on fresh
evaluation data at all four lengths. Also apply the same probe to the exact
checkpoint with interchunk carry erased; require50% within paired numerical
tolerance. A successful probe shows information is cheaply recoverable from
the stored state; it is extra supervised readout training on long contexts,
not zero-shot length extrapolation, language-model superiority or an
architecture-only breakthrough. A failed probe does not prove all nonlinear
readouts impossible. Gate: both classes>=95% at every length, all3 seeds per arm.

Each composite has27,712 frozen +66 new trainable parameters. Count983,040 raw
feature-source positions plus2,048,000 repeated state positions =3,031,040
training positions/model. Evaluation and integration forwards are separate.
Save all fitted heads, train standardization buffers, optimizer and source hashes.
