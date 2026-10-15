# BA gate — achieved synthetic last-write memory, still not breakthrough

Frozen BA GitHub Actions run: https://github.com/vinceackermann2-sys/tam-research/actions/runs/38081374598, attempt 1, success.

Evidence commit: `742fba9f9852e517efaba96743d4d3e2bb3a6f94`. Job: `rlt-cpu-carry-intervention-20261010-ba`. Evidence JSON and **120,530-byte trained carry.pt** are both committed in Git; SHA-256 `74fe2bb1cfb74242aaba3d0eb8ac5df13c1a421a7d50054a0d7dbf503fb4149b`. Git blob `d9d536b1b626412e20d57042dd7d1730571be276`. The workflow's staged-content guard, remote hash readback and independent Actions artifact upload completed successfully. No rerun or amendment.

BA carried model was 27,712 parameters and ran 15 s short-control plus 90 s long-phase CPU training with fresh seed. Balanced heldout accuracy 256 examples/distance was **100% at all five gaps [1,4,16,64,127]**, with binary NLL under 0.00052 throughout. For 16 counterfactual pairs/gap, flipping the final earlier SET (opposite true label, unchanged suffix) yielded 100% *both* correct. At gap127, the mean recurrent-state L2 separation between the opposing paired inputs diminished from ~2.788 after chunk1 to ~1.691 after chunk16; final binary-logit mean absolute difference was ~7.947. This is genuine model-specific memory access under its configured local attention, not an AW or AY historical checkpoint reconstruction.

Do not claim seed robustness: AW succeeded at intermediate distances but not 127; AY collapsed at all long distances to one class; BA succeeded even at 127. These are single-seed engineering screens with differing seeds/protocols. Seed sensitivity and transfer beyond the synthetic SET/QUERY language remain unknown. The prior 15M real-language results favored Transformer.

## Next zero-GPU inference-only BB gate

Use the **existing BA-trained checkpoint** without retraining. Compare three precisely same-weight variants: unchanged carry-RLT, full per-chunk recurrent reset, and state erasure at token 120 only. Test two *fresh independently generated, balanced heldout* evaluation seeds (512 samples/gap) for distances [1,4,16,64,127] and opposite-target suffix-identical counterfactual pairs. The two ablations' final logits must be identical when the true last write is inaccessible (16/64/127); the carried model may differ. Record class-conditional accuracy, binary NLL and logit differences, preserve negative evidence. Source/data/regression gates precede execution. No GPU, new training, scientific replication, larger model or breakthrough claim.
