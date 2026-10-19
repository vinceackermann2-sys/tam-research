# Original SI / PCMA-v1 — Independent published memory benchmark gate

**Date: 2026-10-19 · DRAFT / engineering-only · No scientific seeds or claims.**

## Why this branch exists

The previous same-project handcrafted 27-state history-necessity environment showed only a **0.0003515 (0.076%)** development-only MSE benefit of a last-feedback model over a 3×4 symbol lookup, with the strongest constant-action baseline still performing better on average. Source: original SI PCMA draft PR #1515, successful reporting in GitHub Actions run `39215009963`, and the PR's permanently recorded negative commentary. This was enough to **STOP SCALING THAT TOY**. Merely inventing a harder custom environment would risk post-hoc benchmark construction.

The next gate must use a **pre-existing, independently authored external benchmark**, not a generator tuned until PCMA seems to win.

## External source and frozen package

[POPGym: Partially Observable Process Gym](https://github.com/proroklab/popgym) was published by Morad et al. at ICLR 2023. The underlying source and its unmodified Gymnasium environments are controlled by the independent `proroklab/popgym` project, not by our SI research branch. PyPI release `popgym==1.0.7` is explicitly pinned. Its wheel was released on **2026-06-11**, sha256 `23be7f754f3167f58c4ac844426f3e0f7230035bedfcd5d85b004524eef21bb6`. The first-party POPGym project warns that its legacy RLlib baseline packages are difficult to maintain; we install only the core environments and do not silently downgrade our research training stack.

Published, predeclared selection for this external positive control:

| Environment | Public information memory requirement | Task-specific cheap reference |
| --- | --- | --- |
| `RepeatFirstEasy` | Remember the *first* observed suit despite later cards | Store the initial public suit |
| `RepeatPreviousEasy(k=4)` | Recall a card suit from four observations earlier | A four-item FIFO history buffer |
| `CountRecallEasy` | Count prior publicly observed card colors | Two integer cumulative counters |

Every reference controller receives exactly the public Gymnasium `observation` value and produces a permitted action; it never receives `get_state()`, the Markovian hidden-state wrapper, future/unchosen reward, `info`, the environment object or a privileged oracle. A no-memory negative control sees **only the current public observation**: current suit on the two repetition tasks, and current-card match for count recall. The explicit-memory controller is algorithmic and task-aware; it has **no learned parameters**, which makes it a deliberately strong, inexpensive lower-cost sanity comparison. All controllers use the exact same 8 development episode IDs `21..28` and the same number of action opportunities per episode. Episodes reset the policy's state independently, so there is no cross-episode data leakage.

The benchmark task rewards are returned unmodified and averaged **by episode**, not by action (actions within an episode are correlated). The positive-control script runs three tasks, 8 public episodes each, both policies on the same reset ID: `3 × 8 × 2 = 48` environment episodes on CPU. It prints the returns to the GitHub PR log and refuses to promote the benchmark if the exact-memory controller does not reach ≥0.98 mean normalized return and exceed the current-only control by at least 0.05 on *each* environment. These are engineering smoke assertions, **not preregistered thresholds for PCMA superiority**.

## What this would establish

If the smoke check succeeds, this external dataset contains a **detectable, solvable public-history signal** under those simple controls. That is a better foundation for an evaluation than the earlier overfit tiny generator. The task-specific classical policy may approach perfection; this is GOOD evidence that there is no intrinsic reason to invoke a 64k-parameter SI model for basic memory recall. Full POPGym contains other, more complex partially observed tasks and its own published neural baselines.

This check **does not run or train the original SI/PCMA hybrid, Transformer, GRU, or cheap learned regressors** on POPGym, and it DOES NOT say they would generalize. The original factual two-action PCMA API has only three public symbols and guess/probe actions, while POPGym offers different observation/action spaces, rewards, and episode lengths. The input/output heads, memory write semantics, action-conditioned world model and training objective would all require a **separate, fair, frozen architecture adapter** before such a test could count. Pretending that an external classic FIFO policy is a trained PCMA result is disallowed.

## Mandatory next STOP/GO boundary

1. This initial **positive control** must pass on the untouched external tasks without using any scientific seed or paid GPU.
2. A subsequent **single frozen adapter design** must feed identical public observations, past action/reward, elapsed step, and action budget to original hybrid + PCMA, Transformer + identical PCMA, GRU, and cheap classical references. No hidden-state wrapper or privileged rollout in training.
3. Pre-register substantial actual CPU/GPU **development training** and independent episode IDs before model selection, with the same optimizer schedules, source manifests, checkpoint hashes, and complete compute accounting. Models may not reuse the smoke episodes as sealed heldout.
4. Externally independently selected hidden tasks and independent unique-attempt ledger must be locked before any confirmatory seed is consumed. Precommit a meaningful improvement margin over the **best cheap control** and matched-compute Transformer, world/episode-level uncertainty, and a hard scale/stop decision. With near-perfect task-specific classical controls, failing to beat them should yield **NO-GO for an architectural-superiority claim on these tasks**, not benchmark retuning.
5. If no viable cost-matched, nontrivial external memory/control task is found, **STOP the PCMA-v1 research line** rather than indefinitely adding toy worlds.

## Explicit current verdict

- Original unchanged screenshot-inspired SI v26: **scientific NO-GO**.
- Previous SI PCMA custom history world: **NO-SCALE** on its observed developer data.
- Published POPGym benchmark smoke: **engineering-only** until the exact-head CI result.
- Neural SI/PCMA successor: **UNKNOWN**, not breakthrough.
- No scientific seeds, scientific CPU trials, paid Modal/Colab/GPU, merges, or cross-project changes authorized.
- Canonical original hybrid Git blob is still `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`.
