# Conv7 Colab Pro backend migration audit (no GPU)

The historical Conv7 three-way pilot and source-lock describe a Modal H100 80GB run, while the current merged repository has a new WSL-hosted Colab Pro issue worker. They are not interchangeable. This audit is for planning only and may not be used as authorization.

GitHub issue #1443 reports one successful Tesla T4 CUDA smoke test with the Colab Pro worker, plus verified teardown. That already-consumed job ID must not be rerun. A T4 CUDA smoke is not 200M-token training and says nothing about the actual Conv7 quality gate.

The reviewed worker accepts only an explicitly approved owner-authored job issue with a script in colab_jobs or its earlier probe. Its default GPU allowlist is CPU/T4, and default/max timeouts are 180/300 seconds including provisioning. The proposed 200M-token-per-model pilot has no standalone Colab runner or migrated and verified frozen training/validation dataset. Colab compute-unit billing is not controlled by a guaranteed hard dollar limit here, and it is unknown whether the user's WSL worker is currently online.

This CPU-only preflight pins the merged worker and bridge-guide Git blobs and the previous draft/proposal blobs. It checks the three model names, 3,052 optimizer steps and 200,015,872 exposures per model, absence of assigned engineering seed/budget, and deny-only paid authorization flags. It reports eight independent blockers. Adversarial CPU tests mutate temporary copies of pinned files or add a fake runner to ensure future changes force a new audit.

No real data movement, Colab session, GitHub issue trigger, approved job label, GPU allocation, scientific seed reservation, training, replication, or benchmark is initiated. A separate source-bound Colab job, data/fairness plan, credible runtime/compute-unit budget and explicit GPU authorization remain necessary. The previously consumed scientific seeds 60231/60232 and T4 probe issue #1443 remain frozen.
