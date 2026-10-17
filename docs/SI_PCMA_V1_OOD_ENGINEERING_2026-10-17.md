# Original SI PCMA-v1 — OOD world and baseline engineering checkpoint

**2026-10-17 · Nonconfirmatory · Draft · Zero new scientific seeds**

## Parent lineage and authoritative negative result
Stacked on draft PR #1502 (factual-only learning), itself on #1501 and original SI v26 PR #1499. No `main` modifications or other architecture track changes. The immutable original screenshot-inspired model `tam_research/si_original_hybrid_v0.py` Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` stays untouched.

Original 64k hybrid v26 stopped for failed generalization and losses to a Transformer/cheap lookup. We do not reopen its old thresholds or count its synthetic attempts anew. V25 consumed seed 117214 / quarantined duplicate remain excluded. CI contract tests here are NOT scientific confirmation.

## New task hypothesis and falsifiability
**Hypothesis:** Cross-session factual memory plus bounded online adaptation permits an agent to (i) recover a concealed per-world action mapping; (ii) use a costly probe when its factual value exceeds the fee; and (iii) detect or recover from hidden changes. To demonstrate *hybrid-specific value*, the canonical SI model MUST also outperform a capacity- and compute-accounted Transformer with the same PCMA memory, adaptation, probes and feedback. The simple controls remain in the promotion comparison.

The independent new world contains a hidden context in {0,1,2} and an unknown permutation mapping that context to the rewarding guess among actions {0,1,2}. **Action 3** buys a trusted observation of the current context for fee 0.10, yields no immediate reward, and leaves the context unchanged; guesses return binary reward and advance context. Ordinary context symbols are noisy and available to all agents. All policies receive only public observation and **executed** factual transitions; hidden mapping, context, change point and unchosen action results are absent.

### Explicit environment partitions
- **Development** mapping permutations: `(0,1,2), (1,2,0), (2,0,1)` (cyclic/even).
- **Heldout** mapping permutations: `(0,2,1), (2,1,0), (1,0,2)` (reflected/odd), with zero overlap.
- Scenario **stable**: mapping stationary, ordinary observation corruption probability 0.12, 6 actions/session.
- Scenario **shift**: hidden mapping rotates at global time >=9, 6 actions/session; unseen change marker not provided to policy.
- Scenario **noisy**: ordinary observation corruption probability 0.50, 6 actions/session.
- Scenario **long**: horizon **16 actions/session** versus 6 in stable, 0.12 corruption. This replaces the prior dummy long scenario only for the new OOD world, without modifying old PCMA v1 world code.

These task choices are development engineering decisions. They do NOT predeclare the final scientific test distribution, learning budget, evaluation seeds, or advancement thresholds. All fixture integers in unit tests are engineering constants, NOT preregistered or validated scientific seeds.

## New control methods
1. Uniform randomized action including paid probe; fixed no-probe guess.
2. Cheap recency-window action-conditional Beta-smoothed empirical values, optionally epsilon exploration and fixed real-fee probe schedule.
3. Nonprivileged exponentially updated Q values with a one-sided factual negative-surprise accumulation/reset control.
4. Existing original source controls: factual exact-match value lookup and online tabular TD(0), unchanged.
5. Existing original hybrid and matched Transformer interfaces, frozen-neural+PCMA adapters, and error-reliability controller from PR #1502, unchanged and capable of calling the OOD runner.

A new **development-only** data collector refuses any heldout-rule configuration before collecting training rows. It records rewards ONLY for the action chosen and executed, namespaces logical times, and builds query reward targets separately from public input tokens. Each policy's rollout uses its own world. Same initial seed does NOT guarantee identical state/observation noise trajectories if actions differ, so report that caveat when comparing paired means later.

## Outstanding hard prerequisites before genuine scientific experiment
- **No scientific run is authorized or undertaken here.** Future frozen protocol must specify exact repository source blob hashes, pristine unused seeds checked against *all* GitHub consumed-attempt logs, CPU budget, data volumes, training recipe, stratified OOD scenarios, error bars/paired statistics, success/failure gates and retries=0.
- Add meaningful causal necessity tests showing the paid probe beats a good no-probe policy under some uncertainty regime, with value-of-information and total fee measured. A forced/periodic probe is not a learned information-acquisition controller.
- Train true outcome-supervised neural controllers on *development-only* data; compare fully trained, adapted hybrid and equally enabled matched Transformer and GRU, strong classical recency/change detection, memory-only/adapter-only, shuffled memory, action-blind retrieval, zero probe and privileged **explicitly labeled** upper bound. Equal parameter totals do not imply equal active FLOPs.
- **Hardening:** Python's private attributes are an API convention, NOT a security boundary. Evaluator must never pass the world to controller or execute arbitrary untrusted policy code in the same process. Separate processes or an explicit narrow RPC tool boundary should be adopted for a stronger causal leakage guarantee. Unit tests check nominal API boundaries, not malice.
- Avoid cherry-picking a benchmark where rules trivially collapse to lookup or fixed decision. No new v27 tweaks to unchanged old hybrid, no performance promotion based on isolated toy task successes; require reproducible real multi-step task evidence and independent replication.
- Profile effective latency, FLOPs proxy, storage and updates; register worst-case degradation threshold for any retained memories and unseen shifts.

## What CI can and cannot establish
CI can check rule split disjointness, sequential factual information accounting, observability, paid probe fee, persistence and 16-step horizon, cheap controls and heldout training-data denial. It cannot establish adaptive advantage, safe deployment, superintelligence, or a breakthrough.

**Decision until the next independently frozen, properly controlled experiment: UNKNOWN; DO NOT SCALE.**
