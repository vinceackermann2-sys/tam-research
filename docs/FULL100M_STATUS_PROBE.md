# Full-100M Modal status probe

Purpose: diagnose the production `tam-research-full100m` launch without starting, retrying, or duplicating any training.

Trigger an issue whose title starts with `[modal-status-full100m]` after this change is merged. The workflow only runs Modal CLI metadata/storage queries from the GitHub runner:

- `modal app list --json`
- `modal app history tam-research-full100m --json`
- recent app logs (non-following)
- Volume listings for the 100M pretraining data, post-training data, and full100m run root

The probe allocates no Modal Function and no GPU. Its issue comment is intended to answer whether the production app is still active, stopped, or failed and whether durable data/checkpoint artifacts exist before any retry decision.
