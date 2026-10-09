# Conv7 three-model engineering: zero-GPU source-bound readiness

This audit is **not authorization to train**. It is a local, read-only check on the merged draft engineering proposal. It neither imports Modal nor constructs, trains or launches a GPU model. It **cannot** reserve seeds or spend credits.

## Frozen source and safety

The readiness code checks the exact Git blob hashes of the original model builder, candidate prototype, Transformer and attention-8 parent, shared learning-rate helper, and draft experiment JSON. Any change to the locked files causes a fail-closed check; updating the lock requires a new reviewed PR and original CI attempt.

It cross-checks the candidate's CPU-only parameter contract (101,803,536 parameters, +16 versus the Transformer), eight attention layers [3,6,9,12,15,18,21,24], the three-way comparator list, exposure geometry, batch sizes, shared validation-stream intent and **exact full-horizon learning-rate prefix**.

A successful report must still say **CONV7_THREE_WAY_CPU_READINESS_PASS_PAID_EXECUTION_BLOCKED**, with no engineering seed, no strict GPU spending cap, and no paid execution authorization. The result is a source/contract readiness signal only, **not evidence about convergence, H100 performance, scientific quality, or breakthrough viability**.

## What remains blocked

Before any H100 launch, separately freeze and review an execution harness with source/tree/harness identifiers, a unique engineering seed, strict cost ceiling and one-shot durable reservation before the first paid GPU allocation, retries=0 and no resume. Test cancellation/OOM/partial-result behavior without spending GPU credits. The user must expressly authorize the paid run for its exact source and seed. A future engineering pass would still require independent authorization for 2B scientific training, replication or scaling.

The prior consumed scientific seed **60232** and its verified SCREEN_FAIL result remain immutable.
