# Original SI / PCMA-v1 — History-Information Falsification Gate

**19 October 2026 · Development diagnostic only · NOT scientific approval or a breakthrough.**

## Lineage and immutable negative result
This isolated draft stacks on original screenshot-inspired SI PCMA history-necessity world PR #1514, following #1513/#1511/#1510/#1509/#1508/#1507/#1505/#1504/#1503/#1502/#1501 and original unchanged hybrid v26. Canonical original Git blob `575c17dc2f39a29d930d82179aa0c96dcbdc7d0f` must remain untouched. **Original v26 remains scientific NO-GO**, and the v25 invalid duplicate of consumed seed117214 remains quarantined. No merge, `main` change, new scientific seed, or paid compute.

## First real falsification question
The history-necessity world proves that the *same current public symbol* can correspond to private states requiring different guesses, so a memoryless lookup cannot be correct in every latent state. This alone **does not** establish that factual histories are useful for prediction, let alone that the original hybrid has an advantage. A last-feedback lookup or cheap regularized history model might exploit the full relevant signal without neural sequence modeling.

New `tam_research/si_pcma_v1_history_information.py` implements:

- A **last-feedback backoff** factual predictor: public symbol, candidate first action, immediately previous *executed* action, prior public symbols and prior observed reward. It learns real chosen first-arm two-action net return, smoothing toward the same symbol-only 3×4 factual baseline. No hidden context, mode, phase, target reward or alternate arm enters its inputs.
- `FrozenLastFeedbackTwoStepPolicy` runs that cheap predictor through the exact same first/second decision phase, paid probe cost, fixed second-stage public continuation, bounded factual pair memory, rank-2 online adaptation budget and restart/ablation controls as every PCMA neural contender. It can use the separate-process **development-only** world IPC.
- `development_history_information_audit`: prevalidate that both train and validation partitions are `development` worlds with **disjoint underlying world seed IDs** and no duplicate configurations, *before building any world*. Randomize actually executed first arms separately in the two groups, with known 0.25 propensity and the same frozen public second-stage decision. Train four frozen cheap predictors ONLY on development-training factual rows: constant per action, 3×4 symbol/action lookup, public-history ridge, and last-feedback backoff. Compute paired factual-return mean squared error on disjoint development-validation worlds. Average **within each world, then equally across worlds** to avoid counting correlated actions as independent replications.
- `DevelopmentInformationAudit` reports per-world descriptive scores and historical sample counts but **always** has `passed_scientific_gate=false`, `may_evaluate_heldout=false`, and `may_claim_architecture_gain=false`. There are no p-values, confidence limits, preregistered thresholds, neural evaluations or protected hidden-task observations in this diagnostic.
- Contract tests: synthetic example where identical public symbol/first action have different target returns according to earlier *factual* feedback; prospective current/future target never becomes a predictor feature; strict train/validation world-disjointness; failure before any heldout world creation; exact row counts; no action-level pseudoreplication; reproducibility on developer fixture integers; cheap history controller's full public-only IPC rollout and PCMA memory/update accounting.

## What constitutes a useful EARLY negative finding
An observed single fixture run is not enough to call anything a breakthrough or failure. A **credible reason to STOP scaling this successor** will be failure to show stable, meaningful conditional-history predictive value on multiple independent development worlds over **both** symbol-only lookup and the cheap existing history ridge; a trivial finite-state/belief-state/TD controller explaining the effect; or a nonpositive *net* probe benefit after its full action opportunity cost. Such a development outcome would disqualify the current task/mechanism as worth an expensive confirmatory campaign; it would not prove universal impossibility of artificial superintelligence.

Do not retrospectively tune the task or baselines based on a bad result to manufacture an advantage. In particular, setting harder hidden rules does not automatically create learnable evidence; the task may become **unidentifiable** from the public history. Failure to beat cheap controls is scientifically meaningful negative evidence.

## When can the user receive a defensible verdict?
There are **two levels**:

1. **Development triage:** once this disjoint-world information audit is executed and cheap finite-history/belief baselines are inspected, one can make a conservative **STOP (not worth confirmatory science)** versus **WORTH A SEALED TEST** decision. Current CI validates only that the audit code works; it is **NOT yet an observed signal or stop/go result**.
2. **Scientific PCMA verdict:** only after freezing adequate actual neural training data/updates and checkpoints; a real active-compute-accounted hybrid vs same-PCMA Transformer plus higher-compute Transformer; the strongest cheap controllers, no-probe/memory/adapter ablations; externally independently authored heldout worlds; completed global consumed-attempt ledger and prospective one-use seeds; predeclared world-cluster uncertainty and practically significant STOP/GO margins; and independent execute-once evaluation. No scientific attempts have been authorized or run yet, so **a calendar completion date would be invented**. Passing developer tests is never evidence of architectural superiority.

Original unchanged v26 already has **NO-GO**. PCMA remains **UNKNOWN** unless early triage justifies retiring this successor.

## Security and provenance limitations
The developer world process uses public JSONL feedback but shares OS account with the controller; it is not a sandbox. The data generator is public code and can be overfit. No part of this PR reserves seeds, accesses private heldout worlds or trains a model with a confirmatory budget. Any world score here is **descriptive, exploratory, and potentially distribution-dependent**.


## Fixed CPU-only developer screen in PR CI
This isolated PR's CI includes an additional **read-only, development-only** step invoking `python -m tam_research.si_pcma_v1_history_information`. It uses 8 predefined development-training worlds (fixture seeds 11–18) and 8 nonoverlapping development-validation worlds (31–38), cycling through stable/shift/noisy/long scenarios, 2 sessions each, fixed first-arm randomizers 23/24. It logs the **same actual** validation-first-arm factual two-step outcomes for all four cheap forecasters, prints per-independent-world MSE and unweighted world means, and hardcodes the warning `NOT scientific validation; NOT a novel seed reservation; NOT a PCMA verdict`.

These developer fixture IDs are NOT reserved or available as scientific seeds and this report does NOT train or evaluate neural PCMA. The workflow step is conditional on this exact experimental branch name; it is not a scheduled research job and does not touch `main`. Its values are exploratory and cannot support a scientific breakthrough claim, statistical significance, or promotion. In particular, no independent world distribution, complete cost-parity audit, preregistered gain threshold or external heldout trial exists. No benchmark tuning may be done in response to this first descriptive report.
