# PGW-v3 Stage-0: utility-addressed workspace structural prototype

Classification: `PGW_V3_STAGE0_STRUCTURAL_CONTRACT_NO_TRAINING`
Frozen proposal: [#1354](https://github.com/vinceackermann2-sys/tam-research/issues/1354); exact Stage-0 implementation contract: [#1359](https://github.com/vinceackermann2-sys/tam-research/issues/1359).
Source branch: `research/pgw-v3-stage0-utility-addressed-1354`, pinned to live main `d2c8ca9ba8022dcb0148d0a1d814b93dbab99c1e`.

## Historical evidence and claim boundary

PGW-v2 mechanism panel #1181 did **not** support the workspace, surprise routing, or token read addressing on its frozen endpoint. PGW core panel #1201/trigger #1213 did **not** support the predictor (incremental ΔNLL +0.002627589 versus required +0.005) or cross-chunk carry (ΔNLL -0.000668723). This does not change.

The PGW-only workspace-path diagnostic #1346/#1351 established a mathematical cross-chunk causal path; original exact-head and merged-main CI passed. That diagnostic does not establish learnability or utility. No CHM, CPW, TAM, RLT, CORTEX, or attention-width result may be reassigned to PGW.

## Frozen Stage-0 module, geometry and allowed tests

This is an **isolated residual workspace module**, **not** a full LM or Transformer competitor. It consumes contextual features `[B,T,32]` from a hypothetical causal local substrate; it returns *only* the workspace-read residual. There is no tokenizer, dataset, optimizer, LM head or GPU runner.

| Field | Stage-0 value |
|---|---|
| `d_model` | 32 |
| `key_width` | 8 |
| `value_width` | 16 |
| `predictor_rank` | 8 |
| `chunk_size` | 8 |
| `workspace_slots` | 4 |
| `selected_events` | 2 |
| `surprise_weight` | 0.5 |
| `utility_temperature` | 1 |
| instantiated parameters (each mode) | 2,209 |

Controls: `hybrid`, `utility_only`, `surprise_only`, `fixed_random`, `recency`, `no_workspace`. Modes instantiate the **same** tensors and may be initialized identically, but **active parameter counts differ**. Do not present them as active-parameter-matched scientific baselines.

For one completed 8-token chunk:
1. Prediction from **previous token within chunk only** (zero at chunk start) gives detached normalized latent surprise; per-chunk standardized surprise. **No predictive auxiliary loss is implemented, so the predictor is untrained in Stage-0.**
2. Learned utility = linear projection of current token latent. In hybrid mode rank by `utility + 0.5*surprise`; select hard top-2 events. Utilities are computed only for the just-completed chunk; nothing from a future chunk.
3. Query *prior* workspace keys before writing any current-chunk events, so a current-chunk write can affect only later chunks. First-chunk output is exactly zero.
4. A selected event produces an event-key and event-value projection. Hard argmax slot selection with softmax straight-through gradient chooses a slot, and sigmoid gate overwrites key/value using `slot <- slot + ST_assignment * sigmoid(gate) * (event - slot)`.
5. Later tokens read the bounded slots by content-query softmax, project memory value back to 32 features, and output a residual. `no_workspace` yields an exact zero residual.
6. `hybrid` / `utility_only` use a straight-through event multiplier `1 + soft_selected - stopgrad(soft_selected)` to permit a utility gradient through hard event selection. This gradient path is only a structural test, **not evidence that the router learns correct retrieval utility**.
7. `fixed_random` is deterministic and input-independent, using only local Python PRNG seeded `135400 + chunk_size*100 + k`; `recency` selects the last two events. Unit test RNG seeds `1354..1359` are test fixtures, **not scientific run seeds**.

Required CPU checks: equal instantiated parameters/shapes across arms; bounded exact top-k, content-independent fixed control; causal completed-chunk lag; no-workspace invariance; future perturbation cannot change earlier output; downstream-only gradient to prior input and utility score; manual same-key slot overwrite; invalid dimensions/shapes rejected.

## Before ANY learning experiment

Create a fresh preregistration and separate branch that freezes:
- token/event grammar and model input adapter for WRITE(key,value), READ(key), distractors, multiple overwrites, delayed query, train/test separation;
- data generation seeds, evaluated delays (1/2/4+ chunks), intervention controls, reference deterministic oracle;
- precise trainable predictor target/objective and whether gradient through surprise is enabled;
- baseline parameter geometry, **active** parameter count for every arm (including the frozen PGW-v2 control), initialization, optimizer, token budget, evaluation interval, compute/time cap, thresholds and failure handling;
- independent pure-routing utility-only, surprise-only, recency, fixed-random, no-memory and full-attention reference, with at least three genuinely fresh training seeds;
- stop if frozen retrieval/overwrite accuracy gates fail; no post-result tuning or reinterpretation.

A synthetic success is at most a *specific capability signal*, not a foundation-model gain. A 25M/10M H100 experiment requires a separately authorized, source-bound, equal-token, equal-**active**-parameter and equal-compute preregistered design. No H100 is authorized here.

`gpu_authorized=false`
`cpu_training_authorized=false`
`scientific_seed_consumption_authorized=false`
`breakthrough_claim_allowed=false`
`scale_up_authorized=false`
