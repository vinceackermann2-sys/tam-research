# SI adaptive horizon v1 — execution interruption and replacement-seed record

**Date:** 2026-10-16. **CPU-only engineering** experiment, frozen source `fd47e2e38cb5d5b5a1c41951589e2a6e9bb3ba72` and protocol `docs/SI_ORIGINAL_ADAPTIVE_HORIZON_V1_PROTOCOL_2026-10-16.md`. No Modal/Colab/GPU job.

The original three confirmation seeds were `95111, 95112, 95113`. The local execution command ran all seeds in order but was terminated by the local 120-second command deadline while `95113` was running. Seed `95113` did not produce a complete, parseable JSON result (0-byte output). Treat it as **consumed/incomplete**, not a model failure or success. Do not rerun this seed or invent its score. The tests for causal prefix invariance and nonzero finite learned-gate gradients passed before training.

**Before executing a new seed**, designate `95114` as the single replacement engineering attempt. It uses exactly the same frozen code, hyperparameters, training source, evaluation and controls. No tuning of window, penalty or other model settings after looking at 95111/95112. The 3-completed-seed aggregate will comprise 95111, 95112, and 95114 **clearly labeled a replacement due to local runner interruption**. This does not meet a strict no-amendment scientific-replication standard; consequently report results as exploratory engineering evidence, not strict preregistered scientific replication.

One attempt only at `95114`. Do not accidentally run the interrupted seed again. Persist JSON and all negative results. This addendum is written **before** invoking the 95114 run.
