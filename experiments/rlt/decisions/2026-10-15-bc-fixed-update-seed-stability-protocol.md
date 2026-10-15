# BC stage gate — fixed-optimizer-update seed stability protocol

Date: 2026-10-15. Parent evidence: RLT AX–BB draft PR #1416, isolated topic `exp/rlt-ax-bounded-transformer`. Initially **source + pure preflight only**, with no paid CPU/GPU execution implicitly approved by the document.

## Why repeat differently?

AW carry learned gaps 16 and 64 but failed at 127; AY carry collapsed at gaps 16,64,127; BA learned every tested gap, including 127, and BB established through same-weights erasure interventions that BA's long-gap classification actually depends on the persistent recurrent state. These three **nonidentical** engineering runs used different model/data seeds and **equal wall-time**, resulting in different update counts. None independently estimates the probability of successful training. Checkpoint preservation was broken for AY, fixed for BA.

The next falsifiable hypothesis is that one fixed 27,712-parameter carry-RLT architecture has **material initialization sensitivity under an identical gradient-update and training-data budget**. Do not claim replication of AW/AY/BA: this is an exploratory engineering panel.

## Locked BC panel protocol

- One new unique engineering job ID: `rlt-cpu-fixed-update-seed-stability-20261015-bc`.
- Three fresh initialization seeds: 20671081, 20671082, 20671083. Shared easy data stream 20681081 and long data stream 20691081, same sequence examples at the same update number. All seeds independently reinitialize model and optimizer.
- 27,712 parameters, carry-RLT, 8-token chunks, sequence length 128, batch 8, AdamW wd 0.01, clipping norm1, learning rate 0.003 for **exactly 2,100** 8-token easy updates; learning rate 0.001 for **exactly 1,200** 128-token long updates. Evaluate intermediate snapshots after steps 400,800,1200; *no early stopping or selection*.
- Distance bins 1,4,16,64,127. At each milestone, the same unseen seed 20701081 balanced holdout (128 examples per class, 256/bin) is used only for reporting, never fitting or selecting the final step.
- Locked per-seed pass: at final 1200 updates, both classes independently >=95% correct at **each** gap 16,64,127. Report all three outcomes, accuracy, class-specific accuracies, balanced binary NLL, updates, tokens/second, and checkpoint hash, including a failing model. The panel is not a statistically reliable population success-rate estimate.
- No new GPU, no scaling, no scientific replication, no protected seeds.
- Every single trained case writes a unique `.pt` checkpoint and per-case JSON using exclusive creation. The terminal result must include all three, plus a durable provenance and a SHA-256 manifest; ignored `.pt` files must be force-staged and independently read back from the remote Git ref. The attempt is nonretryable, including failures.

## Pre-registered interpretation

If 3/3 pass, the design has limited evidence of reliable learning under these three selected initializations and fixed steps, still not language competitiveness. If 0–2 pass, there is direct evidence of optimizer/seed fragility under this protocol: pursue readout/gate initialization interventions as separately registered experiments rather than selecting the winning seed. Do not decide breakthrough/abandonment solely from toy-task success or failure. Real held-out language NLL and efficient training/inference over a strong equal-budget Transformer remain the decisive unknowns.

## First gate

Add source + nontraining correctness tests on topic branch, require green separate GitHub Actions run and source-bound immutable reservation **before** launching the new CPU panel. Read live RLT/AX/main refs before any write. Never mutate the old results or direct-write to `main`.
