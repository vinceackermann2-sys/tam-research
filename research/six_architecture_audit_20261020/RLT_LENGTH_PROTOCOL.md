# Frozen length extrapolation diagnostic

After the already-declared carry/bias4/reset training completes, load its six
carry and bias4 checkpoints. No training, selection or parameter changes. Test
length256/512/1024, with the forced SET at position0 and QUERY at the last
position. All intermediate tokens are native distractors. This isolates longer
retention, not complex overwrite generalization. Use128 noise-prefix clusters
per length, each paired with SET0 and SET1:256 balanced examples. New fixed seed
202610242+length. Hash each panel.

Evaluate each checkpoint with carried state enabled and with the same weights'
cross-chunk state erased. Report both classes, NLL and per-example predictions.
The reset ablation has identical final local input for each opposite-label pair,
so its balanced accuracy must be50%; assert this as a leakage diagnostic.
>=95% in each class at gap1023, together with causal erasure, would support this
specific retention extrapolation. It would not establish language quality,
general algorithmic competence or original published RLT fidelity. All outcomes
are exploratory development diagnostics; no fresh training exposures are added.
