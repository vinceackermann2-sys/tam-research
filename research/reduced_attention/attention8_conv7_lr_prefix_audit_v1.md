# Conv7 zero-GPU full-horizon cosine audit

**Status: not paid-run authorization.** No Modal runner, GPU allocation, engineering seed, or training script.

The previous 200M engineering panel ended its learning-rate cosine at 3,052 optimizer steps; the failed 2B scientific Pair-1 used 30,518 steps. Both used the same peak learning rate but *different schedules*. The new proposal explicitly requires the **first 3,052 steps of the 30,518-step cosine**, using the actual tam_research.train.cosine_lr implementation, not a fresh shortened schedule.

| Update | Zero-based step | Full-2B prefix learning rate | Old short-200M learning rate |
|---|---:|---:|---:|
| First | 0 | 0.0000004918032787 | 0.0000049180327869 |
| End of long warmup | 609 | 0.0003000000000000 | 0.0002782475488964 |
| First post-warmup | 610 | 0.0003000000000000 | 0.0002781703019503 |
| Final pilot update | 3051 | **0.0002955864947336** | **0.0000300000744682** |

The long warmup is 610 steps versus the short schedule's 61. At the 200M endpoint the 2B-horizon prefix is about **9.85x larger** than the old short-horizon LR. This explains a non-equivalence in the experiment designs; **it does not prove that Conv7 will improve quality or efficiency**.

Added CPU tests import the original optimizer function and check learning rates, 2B/200M exposure geometry, shared three-model evaluation stream, the 819,200 validation-token evaluation geometry, and the still-unfunded protocol.

The engineering seed, spend cap and run authorization remain unset. Any future executable experiment must be separately preregistered and authorized, reserve its attempt durably before GPU allocation, have retries=0, no resume, primary Modal only, and never reuse seed 60232. A pilot pass is not a scientific replication or breakthrough.
