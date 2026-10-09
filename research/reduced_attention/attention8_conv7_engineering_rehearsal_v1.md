# Conv7 engineering screen: synthetic-only decision rehearsal

**This is a CPU-only test of decision logic, not training.** No model has been run, and all example NLLs and throughput figures in the unit tests are *invented synthetic values solely for branch-coverage tests*. These values must never be reported as experimental measurements.

The three-way draft engineering plan compares Transformer, original attention-8 and attention-8 plus causal Conv7 at ~200M token exposures each, with the same complete optimizer-step count, data/evaluation geometry, intended seed and **full-2B cosine schedule prefix**. The script \`tam_research/cortex_attention8_conv7_engineering_rehearsal_v1.py\` reuses the draft threshold values and accepts only records explicitly labeled \`SYNTHETIC_ONLY_NO_TRAINING\`.

## Fail-closed behavior

- Reject missing or extra models, model-identity mismatch, non-COMPLETE runs, mismatched token or optimizer steps, wrong scheduler horizon, differing synthetic seed/evaluation stream, missing or non-finite metrics.
- When all records are syntactically consistent, evaluate three distinct draft gates: candidate Transformer NLL delta <= +0.015, improvement over original attention-8 >= 0.010 NLL, and training TPS ratio vs Transformer >= 1.03.
- Even with all gates passing, classify the result as \`SYNTHETIC_ENGINEERING_SCREEN_PASS_NOT_SCIENTIFIC\`, **never** as scientific evidence or a real engineering result.
- If any draft authority flag is turned on, or the engineering seed or spend cap becomes populated, the synthetic rehearsal raises a fail-closed error; it cannot authorize training by itself.

This module has no Modal client, PyTorch training entrypoint, GPU jobs, file mutations, networking or job triggers. It uses only the previously merged design proposal and in-memory mock records.

## Next research gate

Only after CPU CI succeeds may the decision checks be reused as a **design reference** for a separate durable, source-bound engineering runner. The current mock classifier is explicitly not a validator of real experiment files; actual result provenance, job reservation, GPU spending, and dataset fingerprints would require additional hard checks. A new seed, hard budget and paid authority must be explicitly approved before any real H100 work. Replication and 250M/5B remain unauthorized. The previous scientific seed 60232 is permanently consumed.
