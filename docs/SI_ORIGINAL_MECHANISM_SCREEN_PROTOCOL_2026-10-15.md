# SI original hybrid — frozen exploratory component screen

Date 2026-10-15. Code committed as `scripts/si_original_mechanism_screen.py` at `3c0ead6a2d36e5de92dd5f9ec4ea7394b988da49` **before** three CPU engineering seeds 94211 / 94212 / 94213. A two-step implementation smoke check used separate seed 94200; exclude it from the results. We ran this screen after observing the Gate-1 secondary abrupt-switch improvement.

**Hypothesis:** one or more of recurrent state mixing, sparse expert routing, and iterative refinement can explain the hybrid's better response to regime changes despite losing the locked primary stable-world endpoint.

Run `python -m scripts.si_original_mechanism_screen --steps 300 --batch 16 --seed SEED` for each of 94211, 94212, 94213. Same action-dominant world generator, paired data streams, training and evaluations as Gate-1. Compare full hybrid, all-attention/no-recurrence, dense-FFN/no-MoE, no-refinement, and full Transformer.

**Primary for this exploratory screen:** report per-seed late switch MSE and full-sequence stable MSE together (avoids optimizing away the downside). Secondary: parameter count, CPU wall time, other OOD results and route use where possible. No tuning based on this result.

**Important limitation:** these are deletions/substitutions with **unequal trained/active parameters, FLOPs, dispatch patterns, and sometimes different initialization streams**; do not call an improvement causal attribution, statistical evidence of breakthrough, or GPU approval. A better candidate must still pass independent matched-capacity/residual/compute-controlled experiments using additional unconsumed seeds and task families. Do not rerun this as scientific replication; no Modal credit authorized or spent.
