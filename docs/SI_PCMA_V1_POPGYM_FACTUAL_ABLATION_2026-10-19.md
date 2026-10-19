# SI PCMA-v1 — frozen public-factual online-state ablation on published POPGym

**19 October 2026 — development only, nonconfirmatory — not a new scientific attempt**

## Lineage and immutable evidence
Stacks on #1517, which directly trained original `OriginalSIHybrid` and original `TransformerControl` with 48 public-memory-imitation updates on external POPGym, and previously on #1516, #1515, etc. Canonical untouched screenshot-inspired SI source Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`. Original v26 scientific **NO-GO**, custom history-world **NO-SCALE**, and #1517's negative 48-update neural memory acquisition remain binding. No merge, main edits, science attempts or paid compute.

## Core design and what has not changed
On `popgym==1.0.7`, exact original two 64.7k-parameter backbones are fitted once each with **identical code/data and fixed 48 CPU AdamW updates** as #1517. Frozen development-training episode IDs 101–112/task; development-validation 201–208/task. The original model sources and initial seed fixture 70631 are unchanged, as are public three-channel observations, zero action inputs, fixed 3-to-27 output head, 51-step length, three published tasks and task-specific classical public-memory ceiling **+1.0**.

Unlike #1517, the **same frozen model** is then deployed under FOUR separate online-memory conditions on each developer validation episode:

| Condition | Bounded factual reward memory | Rank-2 online adapter |
| --- | --- | --- |
| none | Disabled | Disabled |
| memory_only | Enabled | Disabled |
| adapter_only | Disabled | Enabled |
| both | Enabled | Enabled |

## PCMA-mechanism port (important scientific limitation)
POPGym RepeatFirst/RepeatPrevious/CountRecall does NOT implement the original SI PCMA 3-symbol/4-action paid-probe and two-step factual protocol. This pilot therefore uses **an explicitly labelled port** of the *public observed factual reward memory + bounded rank-2 online residual* mechanisms to up-to-27-action published tasks. It does **not** pretend to be a byte-for-byte execution of `TwoStepPCMAState` or claim that factual memory survives an independent POPGym-world reset.

POPGym reward-to-unit-feedback accounting follows the **actual published environment denominator**: 51 reward-bearing steps for RepeatFirst/CountRecall, but **48** for RepeatPreviousEasy(k=4), which grants zero rewards during the earliest three decisions. This correction was required by the first CI execution's fail-closed reward bound, **before any ablation outcomes were produced**. This is source-verified normalization, not tuning to positive/negative model scores.

For every first-action candidate, the same state code uses only the task ID, **currently public** card observation, candidate action and elapsed public step; it retrieves up to eight **earlier actually observed same public-input/action rewards** (stored in a 128-record FIFO). The rank-2 adapter gets only those public features and **actual reward from the chosen action**, never the result of an unexecuted action, private simulator state, privileged `info` or tomorrow's observation. Its auxiliary gradient updates are capped at **8 per episode** at fixed LR 0.05, rank 2, blend 0.5; the memory cap is 128. The state is discarded between independent POPGym episodes to avoid false same-world persistence/leakage. These hyperparameters are **development precommits**, not tuned to score.

The neural base produces strictly causally masked class logits on the public sequence; the preexisting #1517 prefix-invariance and observation-equality contracts remain. Online corrections are computed one executed step at a time, *after* the previous step's actual chosen reward arrives. Real actions are separately executed for each arm in independent environments reset to the same development fixture episode ID, and only real external rewards count. The exact classical **+1.0** return still sets a ceiling. This is not a compute/FLOPs matched experiment; the extra adapter consumes additional inference/update compute.

## Falsifiable metrics and STOP conditions
The branch-specific PR GitHub CI prints **24 descriptive rows**: (hybrid/Transformer) × (none/memory_only/adapter_only/both) × (three task families), mean actual episode return and public-expert accuracy, counted actions, stored factual records and adapter updates. Frozen model weights, training data, validation IDs and task selection remain identical across ablations. No heldout configuration is available in this file, no science seeds are reserved, and it always reports `permitted_architectural_claim=false` and `active_flops_matched=false`.

Predeclared interpretation:
- If online PCMA port does not improve meaningfully over *the same backbone with no PCMA*, record **negative development evidence for this memory/adapter instantiation**. No post-result retuning of blend, memory, rank, actions or task is permitted.
- If both online architectures remain below perfect cheap classical public-memory performance, **NO claim of SI advantage** on these tasks regardless of ablation lift.
- Any hybrid-vs-Transformer difference is *descriptive* only, not a scientific architecture winner: unequal actual computation, weak 48-update teacher imitation, tiny published easy tasks and no independent protected heldout.
- Even apparent lift from this port would NOT validate persistent cross-session world-law acquisition or economically useful probe purchase (the original PCMA hypotheses). A real, sealed SI experiment must evaluate those original mechanisms on an independently authored nontrivial task with a matched-PCMA Transformer, stronger classical belief/TD controls, audited compute, disjoint unseen worlds and precommitted failure/success thresholds.

**Current scientific verdict remains unchanged: original SI v26 NO-GO; custom PCMA world NO-SCALE; PCMA successor UNKNOWN.**
