# Original SI PCMA-v1: history-necessity world — structural development milestone

**2026-10-19 · Nonconfirmatory engineering only · No scientific trials, seeds, compute or breakthrough claims**

## Provenance and negative evidence
This draft extends original screenshot-inspired SI/PCMA strong factual control draft PR #1513, stacked on #1511/#1510/#1509/#1508/#1507/#1505/#1504/#1503/#1502/#1501 and the untouched original SI v26. The original SI source blob remains `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f`. The unchanged 64,614-parameter original hybrid v26 **NO-GO** and invalid repeated consumed v25 seed117214 quarantine remain authoritative. No `main` changes, merges or other-project edits.

## The specific scientific design problem
The prior `OODRuleWorld` uses just three public symbols and three hidden static mapping permutations in each split. A 3×4 public-symbol/action lookup plus a little action-value learning can plausibly solve this tiny task as well as, or better than, a much larger neural architecture. A neural win on such a benchmark cannot establish long-term causal memory or special SI learning dynamics. This new **additional, development-only** world is a falsification-oriented structural alternative, not a replacement or retuning of the frozen old v26 experiment.

## New history-dependent dynamics
`tam_research/si_pcma_v1_history_world.py` adds `HistoryNecessaryDevelopmentWorld` with a hidden **context (3) × phase (3) × mode (3)** state, plus an independently drawn hidden reward offset. Observation contains only `(context + phase) mod 3`, corrupted independently in ordinary observations; a paid probe consumes a full action, costs **0.1**, emits zero immediate reward and provides a trusted version of the *same* public symbol. It does not reveal hidden mode, context, phase, offset, reward of an unchosen action or future state.

An executed guess's reward is 1 only when the guess matches `(2×context + phase + mode + offset) mod 3`. Guess actions advance hidden context and phase differently based on the chosen action. In shift worlds, hidden mode changes after nine global actions; noise rate increases in noisy worlds; long worlds have sixteen actions per session. Context, phase, mode and offset persist across session restarts.

**Mechanical testable consequence:** the exact same public symbol and remaining action budget can correspond to hidden `(context=0,phase=0)` versus `(context=1,phase=2)` with distinct correct guesses, at fixed mode and offset. Thus no memoryless public-symbol/action lookup represents the correct guess everywhere. Past executed actions, observations and factual rewards may carry information about latent state. This is a **necessary challenge**, not evidence that the latent state is always identifiable or that either architecture can learn it.

## Factual training interface and independent IPC
`randomized_history_development_pairs` uses the same *uniform known first-action propensity = 1/4*, same immutable public second-action continuation, same actual chosen two-step return and same `RandomizedFactualPair` schema as existing PCMA neural and cheap-control development data. `randomized_pairs_to_causal_samples` therefore works for both, and neither receives oracle labels or information about unexecuted alternatives.

The existing public-only `si_pcma_v1_world_worker.py` and `si_pcma_v1_world_ipc.py` now accept an explicit `world_kind="history"` in addition to the unchanged default `"ood"`. Both are **development only** and reject heldout configs. The policy sees only `PublicObservation` / `FactualTransition`; the child's private hidden state cannot be retrieved through the JSONL protocol. Unit tests compare exact development traces against a trusted same-process reference across stable/shift/noisy/long scenarios, including probes, and assert correct rewards for same-public-observation but different hidden latent states.

## Methodological and security limitations
- **Still a small authored toy.** The hidden state is 27 combinations, the public and action alphabets remain tiny, and the generator is public code. It is better than a static lookup benchmark for a particular structural question, but it is NOT independently authored heldout intelligence work. In fact, a simple classical filter, finite-state model or TD/PSR controller might dominate it; that is a desired falsifier.
- **No proof that history helps empirically yet.** A collision between two private states shows a current-symbol table cannot be universally correct; it does *not* prove a history-conditioned model can disambiguate all states, use a probe profitably, or generalize from limited data. A statistical conditional predictive-information check on strictly held-in development trajectories is necessary before proposing any scientific run.
- **No science-ready seed plan.** All numbers are deterministic developer fixtures; neither `heldout` creation nor scientific execution is supported here. Historical attempt ledger is still globally incomplete.
- **Same-OS process ≠ secure sandbox.** A malicious local policy may inspect process args/source/environment, even though ordinary callbacks see only public facts. A future independently designed heldout generator must be privately housed in a restricted process with authenticated receipts.
- **No training budget parity.** Existing neural routines remain ≤32-update developer checks, and PyTorch profiling is incomplete for active FLOPs. Equal samples or nominal parameters alone are not valid compute matching.
- **No model performance result**: this branch must not report heldout scores, rank models, optimize parameters on the new structural world, declare information advantage or promote PCMA.

## Decisive next phase / stopping rule
Before spending **any new scientific seed**, stop expanding the benchmark catalogue and establish: a conditional-history value necessity check (and cheap finite-state/Bayesian control) on development data; genuinely frozen development training/cost budgets for hybrid+same-PCMA Transformer+strong classical baselines; independent unseen task generator and sealed heldout validator; world-level uncertainty and explicit no-go margins. If a cheap history-based classical policy solves the revised task or the probe has no positive net value after opportunity cost, record that as evidence AGAINST the proposed architectural claim rather than tuning away the failure.

**Current verdict: original unchanged v26 NO-GO; PCMA-v1 still UNKNOWN.**
