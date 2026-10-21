# Six-architecture measured verdict

None has established a breakthrough. Do not advance any current tested recipe
to larger language training. Conv7 is a useful local-mixing component with a
small language-loss gain; its300M quality/cost gate and terminal tasks do not
justify scaling this implementation. SI has concrete inexpensive repair targets.
PGW/CHM need valid mechanism/control improvements, and bounded RLT needs robust
extrapolation/overwrite behavior before scale is sensible.

This is an allocation decision about the pinned implementations and executed
recipes, not proof that the architecture families are impossible. The study
combined the earlier50 completed runs with new near100M language training and
targeted source-pinned small-model experiments. It did not reproduce every
historical repository experiment or compare against all modern architectures.

| Architecture | Decision now | What stops progress | Evidence needed to reopen larger work |
|---|---|---|---|
| Conv7 | Stop the current language recipe at300M; optional small systems experiments | Its development lead shrank from.036435 at100M to.005562 at300M, with3.34% lower timed training throughput. PIQA/LAMBADA are below the Transformer. Batch8 prefill is slower, although batch1 and naive generation are faster. | A prospective bounded test that demonstrates a useful quality/cost or longer-context advantage, then independent-seed replication. Kernel optimization alone does not establish better task learning. |
| CPW local | Drop this local recipe as the next large language-training target; stop100M | Development NLL5.034013 versus4.520965 at100M despite1.0624x throughput. Fifteen preceding tokens plus the current token cannot encode distant context; the distant-prefix intervention is exactly invariant. The loss deficit is not causally attributed entirely to this window. | A valid long-range-memory configuration and learned positive retrieval controls. CPW-AFM/world variants were not tested by the local panel and remain unvalidated. |
| PGW | Hold large scaling; repair workspace allocation and controls | Three checkpoint traces send93–99.5% of writes to one of four slots. Direct target-value read contributions attenuate, while key/routing paths prevent treating that coefficient as a full information bound. Query-specific binding controls did not learn reliably. | A learned control that passes the same task and an intervention showing allocation repair improves memory-specific performance. Slot-collapse traces alone do not prove the family fails. |
| CHM | Drop the current pointer prototype as a large-training target | Gold first-hop context barely repairs poor unseen-identity/payload readout in three broad-trained checkpoints. Missing-fact rejection also fails. Flat soft attention has not established hierarchical acceleration. | Reliable unseen-address copying and rejection with valid controls; then a native hierarchical cost benchmark. Gold-label diagnostics are consumed development interventions, not deployable accuracy. |
| Bounded RLT branch | Stop scaling the tested memory/readout recipe; hold a general-memory claim | Native and gate-prior recipes each pass only1/3 full length gates. Frozen-state probes decode the first-SET task through1024 in all3 native seeds, but all6 new extrapolation/overwrite gates fail;4096 is near chance. | A repair that survives new lengths, last-write gaps and seeds without task-specific extra labels. The published nonlinear RLT is a separate unreproduced model. |
| SI Original V2 | Continue only cheap revision/persistence and multistep utility work; hold full-system scaling | Planning repair solves a finite task already solved by the direct policy. Native storage isolates worlds, but all8 revised associations return stale deltas in all3 seeds. Native export restores0/56 replay records; explicit full checkpoints restore56/56. | Resolve duplicate-key revision and full lifetime persistence, then show useful A/B/C task acquisition/retention and multistep decisions beyond direct-policy/latest-record controls. Storage correctness alone is not continual-learning efficacy. |

## Language allocation gate

All four selected models have99.56–99.84M actual parameters. At300M, Conv7
requires development NLL at least.03 below Transformer, OR throughput at least
1.03x with NLL no worse than+.015; it must also improve its parent by.005.
It meets the parent condition but neither Transformer condition. The gate was
frozen before the endpoint. No1B or2B continuation was launched after it failed.

| Model | Committed training exposures | Development NLL | Terminal test NLL | PIQA primary | LAMBADA exact word |
|---|---:|---:|---:|---:|---:|
| Transformer |299,999,232|4.136642|4.162973|55.93%|8.21%|
| Eight-attention parent |299,999,232|4.197558|4.223946|55.50%|7.51%|
| Conv7 |299,999,232|4.131080|4.157327|55.06%|6.79%|
| CPW local |99,999,744|5.034013|5.051848|53.75%|0.00%|

CPW's lower training budget prevents a matched terminal comparison against the
300M models. Its stop is supported by the matched100M development comparison.
The parent also has22,224,896 conservatively charged lost exposures, so its total
charged upper bound is322,224,128. Maximum actual model size is99,837,440.
No selected model exceeds100M parameters or2B charged training exposures.

Conv7-minus-Transformer test NLL is-.005646, descriptive block interval
[-.007026,-.004300]. Its PIQA difference is-.871 percentage points with paired
example interval[-2.448,+.871], and LAMBADA is-1.417 points[-1.979,-.873].
These intervals measure examples/blocks in one paired trained seed; they do
not measure training-seed robustness or establish a general superiority result.
Test coverage is2,096,640 target positions; PIQA uses1,838 questions and LAMBADA
5,153 examples. No contamination audit was performed.

## Mechanism and inference checks

Conv7 improves its sparse-attention parent by.066478 development NLL. Removing
its trained convolution gains raises NLL from4.131080 to6.197049. The branch is
used by the trained model; erasure is not a fresh-training counterfactual.
At context512/batch8, Conv7 prefill is36.169ms versus32.770ms for Transformer.
At context512/batch1 it is12.692ms versus13.531ms. Eight-token generation from
a128-token prompt is85.57 versus76.29 new tokens/s without KV caching.
This is mixed implementation evidence, not an optimized production-serving win.

CPW's final logit is exactly unchanged when inputs more than15 steps back are
altered. Its0% LAMBADA result is consistent with a severe local-context problem,
but that observation does not isolate all causes of its language-learning gap.

Continual learning remains unproven in the six designs. SI's exact association
and serialization checks establish isolation and expose revision/replay defects;
they do not show acquisition and retention of useful decisions on new tasks.
RLT preserves recoverable information in a constrained task, with failed OOD
follow-ups. See CONTINUAL_LEARNING_STATUS.md for the distinction and measurements.

## Evidence and limits

L4_RESULTS.md, L4_SUMMARY.json and l4_language_gate_300M.png report only matched
scientific training endpoints. TERMINAL_BENCHMARKS.md and TERMINAL_SUMMARY.json
contain all inference settings, task scores and paired example comparisons.
RESULTS.md contains CPU mechanism/repair results. Prospective protocols and
executed model/training/evaluation sources remain unchanged.

All final language checkpoints include model, Adam and RNG state. Per-chunk/
whole-file SHA256s, finite tensors, exact optimizer steps and actual constructor
counts are checked in TERMINAL_OPTIMIZER_AUDIT.json. AUDIT_CURRENT.json reconciles
immutable sources, dataset/benchmark pins, checkpoints, reservation digests and
the single idempotent lost-exposure charge. Unsuccessful/unknown A100 and runtime
attempts remain recorded; missing evidence is not relabeled a model failure.
All four terminal optimizer files also passed server size/SHA256 checks in the
existing unpublished draft archive; asset identifiers and exact hashes are in
TERMINAL_CHECKPOINT_CACHE.json. Older recovery assets remain preserved. Final
remote checks reproduce all13 scientific source/protocol hashes and all three
completed result-file hashes, including original protocol claims. The finished
L4 runtime was released after these proofs were exported.

The current recipes were not individually hyperparameter-tuned. They were
measured on one large paired seed and a fixed2B-horizon learning schedule.
Nothing here establishes full-system novelty, frontier competitiveness, or
family-wide impossibility. The practical recommendation is to stop large runs
and spend any next research effort on the specific falsifiable repairs above.
