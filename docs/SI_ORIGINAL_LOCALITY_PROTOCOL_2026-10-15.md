# SI original hybrid — controlled local-attention follow-up

Phase-1 action-world secondary OOD switch improvement persisted when removing recurrence, MoE and refinement separately (unequal-parameter ablations). This motivates a **controlled local-versus-global attention mask comparison** to test an alternative explanation, the causal context radius.

**Frozen script before confirmation:** `scripts/si_original_locality_probe.py` committed at `3b2ddb84e50da905f7537135163c1052cb6ae601`. Independent CPU seeds `94311,94312,94313`; smoke implementation seed 94300 ran for 2 steps only and is excluded. Train all models for 300 steps, batch 16, same per-seed world streams and evaluation.

- `local_dense`: SI hybrid scaffold, three causal attention layers window=6, dense FFN, no recurrent mixer, no MoE, no refinement.
- `global_dense`: **identical weights, parameter count, initialization and training** to local_dense, except causal attention window=None (full prefix). These two isolate the mask/receptive-field choice under matched data/model/optimizer.
- `original_hybrid`: original attention/recurrent/sparse/refine design for context.
- `transformer`: previous full-attention dense Transformer for context.

**Predeclared primary for the local/global comparison:** mean over three seeds of *abrupt-switch late MSE*, together with per-seed differences and stable whole-sequence MSE as guardrail. Report parameters and CPU latency; this is a **secondary-mechanism screen**, not a promotion to scientific breakthrough. An apparent gain can arise from inductive bias or optimization; neither local attention nor reference CPU implementation proves scalable efficiency.

Compare local vs global first (matched parameters), not to the larger Transformer (unequal parameters). Keep the lower-quality cases and no-training baselines. No new Modal run approved, no seed rerun, no data adjustment based on results.
