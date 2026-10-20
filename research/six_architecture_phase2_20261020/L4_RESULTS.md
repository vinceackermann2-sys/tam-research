# L4 paired language confirmation

Seed: 202610321. Training complete: False.

Only scientific endpoints enter the comparison table. Segment backup metrics remain in L4_LANGUAGE_RESULTS.csv.

| Candidate | Exposures | Development NLL minus Transformer | Training speed / Transformer | Gate |
|---|---:|---:|---:|---|
| attention8 | 99,999,744 | +0.057621 | 1.0702 | control |
| conv7 | 99,999,744 | -0.036435 | 0.9640 | pass |
| cpw_local | 99,999,744 | +0.513048 | 1.0624 | fail |

| Terminal model | Exposures | Test NLL | Matched terminal Transformer budget |
|---|---:|---:|---|
| cpw_local | 99,999,744 | 5.051848 | False |

One paired seed; token-block bootstrap intervals are descriptive.
Intermediate backup validations do not select checkpoints or authorize continuation.
Terminal models stopped at different budgets are not matched-budget comparisons.
A100 measurements belong to the separate interrupted trial.
Training throughput excludes input preparation, transfers, evaluation and backup time.
