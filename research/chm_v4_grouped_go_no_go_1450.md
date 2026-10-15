# CHM-v4 #1450 — prospective evidence and stop/go contract

This is **pre-experiment statistical and governance code only**. It imports neither the V4 dataset nor any trainable model, reads no model results, and opens no development/test entities. All tests use fabricated numbers labeled `UNREAL-...`; no GPU, scientific seed or training run is executed.

## Evidence ladder: what would count as progress?

- **A — engineering:** Frozen V4 benchmark #1425, hard visible read #1428, TRAIN-only stage interventions #1434, and neural one-step gradient wiring #1435. A purpose-built rule reading 2048/2048 TRAIN cases is *by construction*, not model generalization.
- **B — memorization capacity:** #1436 source-locked 256-step TRAIN-only microfit may, if authorized and executed once, pass only if all **24/24 positive TRAIN rotations** are answered and the one independent no-match entity is rejected. Failure is a concrete mechanism diagnosis, not proof the entire CHM research idea is futile. Success permits future model work but is not a breakthrough.
- **C — model selection:** Future preregistered independent DEVELOPMENT entities; no touch on frozen V3 scored panels or reserved V4 TEST before source lock. No posthoc test tuning.
- **D — first independent TEST:** Only a new approved, source-bound first-attempt study may open the reserved **32 independent TEST entities per family**. Within each entity eight counterfactual rotations are correlated; they contribute one cluster-mean. Compare model accuracy to exact 1/8 value-blind chance and a Transformer using the **same model-visible tokenizer, data, optimizer, loss/supervision budget and measured compute**. Original pointer uses additional source-address CE; a conventional decoder lacking this supervision is *not* a fair superiority control. The fairness fields are externally verified attestations, **not proofs produced by setting booleans true**.
- **E — breakthrough investigation:** Require at least one fresh, strictly independent replication with no reused scored panels, evidence of statistically persuasive cross-entity improvement over a comparably supervised/compute-matched strong Transformer, reported wall-clock and FLOP budgets, realistic-language robustness, ablations confirming the proposed architectural mechanism, and credible cost/utility advantage. Only after this could stronger claims be considered.

## Locked statistical definitions (hypothetical screen)

Pure function `prospective_grouped_screen()` is **NOT a scorer or an authorization**. It only accepts future externally produced `PositiveEntityCluster` and `NegativeEntityCluster` summary records. Each positive family must contain exactly 32 distinct entities, with eight per-entity rotations covering CODE-18001..18008 exactly once and paired correctness indicators for both the model and fair Transformer. Do not treat 256 rotations per family as 256 independent observations.

Family-by-family define independent-entity mean model accuracy `a`, and mean paired accuracy difference `d` versus matched Transformer. With one-sided global α=0.05, use **seven Bonferroni-corrected directional claims**: three `a > 1/8`, three `d > 0`, and one no-match false-positive rate `FPR < 0.05`. For n=32 entities, the explicit conservative Hoeffding lower bounds are:

- `lower_accuracy = max(0, a - sqrt(log(7/.05)/(2*n)))`
- `lower_paired_difference = max(-1, d - sqrt(2*log(7/.05)/n))`

These are mathematically conservative bounded-independent-entity checks, **not estimates of statistical power or evidence that data were randomly sampled**. Exact binomial one-sided Clopper-Pearson upper bound is applied to one false-positive flag **per independent no-match entity**, not to 8 repeated negative variants.

### Critical power/feasibility limit

The current V4 TEST reserves only 32 independent negative entities. Even at **zero false positives** they cannot verify FPR below 5% at one-sided 95% confidence: the *uncorrected* 95% upper bound is `1-0.05**(1/32)≈8.94%`, and the simultaneous seven-claim bound is even higher. The tool is designed to return **screen failed** when this seven-claim gate is mathematically impossible. Do not inspect/rescore existing 32 test entities to repair it. A larger future independent no-match cohort would require its own prospectively frozen source/manifest, review and authority. At least 59 error-free independent negatives are necessary without multiplicity correction, and more with seven comparisons.

### What counts as a failure verdict?

**Technical stop:** terminate a *specific architecture candidate* when it repeatedly fails prospectively pinned source-address/copy/gate TRAIN capacity checks under bounded, separately reviewed, independent engineering attempts, after excluding broken data/protocol/compute artifacts. A failed run is immutable and consumed; its exact seed/source cannot be rerun or retuned retroactively.

**Research-level stop:** recommendation that continuing CHM is *not worth it* requires a preregistered finite research budget, sufficiently powered fresh independent evaluations, fair alternative baselines, demonstrated lack of meaningful quality/compute/utility advantage, and a documented opportunity-cost assessment. A single failed microfit, one negative synthetic report, or absence of evidence is NOT automatically decisive.

No automatic GPU/Modal run, Colab approval label, paid scientific seed, 100M experiment, or breakthrough verdict is created here. The CHM-v3 `CHM_V3_100M_DAEC_STAGE_C_STOP` and scientific seed `2013161` remain frozen.

Governance: isolated additive source branch→original first attempt CI PASS→re-audited main→exact-head pinned merge→independently verified merged-main first CI PASS. No failed/cancelled head is rerun or amended.
