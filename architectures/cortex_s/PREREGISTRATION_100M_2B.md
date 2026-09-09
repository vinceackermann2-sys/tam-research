# CORTEX-S v0 — 100M / 2B paired historical-control preregistration v3-grouped

Status: **frozen before the grouped production H100 preflight**.

This document supersedes the paid execution mechanics of v2 while preserving the same scientific comparison. It does not retroactively turn any engineering run into scientific evidence.

## Why v3 exists

The original preflight `#761` is consumed and stopped in the CPU corpus gate. The repaired v2 preflight `#769` is also consumed. Its zero-GPU corpus gate passed, but engineering calibration seed `910001` measured only `133,384.4585 tok/s`, projecting a conservative full envelope of `16,676.53 s` against the frozen `8,500 s` gate. Therefore no v2 full run was launched.

Subsequent systems work used engineering-only seeds and single-use namespaces. Seeds `2026090901`, `2026090902`, `2026090903`, and `2026090904` are consumed. None are scientific seeds.

Repair4 issue `#799`, workflow `34339214619`, job `102425759653`, exact source `9f4196b10674c6eb0454dc2d3459a7c283f16637`, established a promising systems candidate on one H100:

- physical zero-padding from logical expert width `338` to runtime width `344`;
- exact trainable parameter count preserved at `101,778,112`;
- compiled grouped forward/backward operator probe: PASS;
- exact routing probe: PASS;
- semantic loss delta: `0.004322052001953125` against frozen `0.02` gate;
- legacy production-shape throughput: `139,895.4161 tok/s`;
- grouped production-shape throughput: `182,118.5159 tok/s`;
- measured speedup: `1.3018190376x`, above the preregistered `1.20x` promising gate;
- grouped peak VRAM: `45.2683 GiB`, below the `70 GiB` gate.

Repair4 explicitly returned `full_training_authorized=false` and `next_stage_authorized=false`. It is **engineering systems evidence only**. This v3 integration is a separately preregistered production stage.

## Classification

The eventual seed-8100 comparison remains an **adaptive paired historical-control experiment**. The historical Transformer result was known before CORTEX-S development. Even a win cannot by itself be called a breakthrough.

Reserved independent scientific seeds `48131`, `48132`, and `48133` remain untouched.

## Frozen historical control

Repository issue `#140`, Transformer 100M, seed `8100`:

- parameters: `101,803,520`;
- final NLL: `2.7115590302149455`;
- final perplexity: `15.053722884190283`;
- training seconds: `6227.609573988244`;
- total compute seconds including compile: `6478.327432424761`;
- reported training throughput: `321151.5755348581 tok/s`;
- peak VRAM: about `11.896 GiB`.

No second Transformer will be trained.

## Frozen paired CORTEX-S scientific run

If and only if the new grouped production preflight passes, a separately triggered full stage may use:

- paired seed: `8100`;
- nominal training-token budget: `2,000,000,000`;
- validation tokens: `5,000,000`;
- context: `512`;
- microbatch: `64`;
- gradient accumulation: `2`;
- global batch: `128`;
- AdamW betas `(0.9, 0.95)`;
- peak LR `3e-4`;
- weight decay `0.1`;
- warmup fraction `0.02`;
- CUDA bf16 autocast;
- same immutable corpus bytes as the historical control.

### Token accounting

The historical trainer and CORTEX-S use full optimizer batches:

- tokens per step = `64 × 512 × 2 = 65,536`;
- steps = `ceil(2,000,000,000 / 65,536) = 30,518`;
- literal full-batch exposures = `2,000,027,648`;
- nominal overshoot = `27,648` tokens (`~0.00138%`).

The full run must therefore execute the same `30,518` optimizer steps. We will report both the nominal 2B budget and literal exposures.

## Frozen corpus

CPU fingerprint evidence remains issue `#767`, workflow `34271912952`:

- train SHA-256 `93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b`;
- validation SHA-256 `ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760`;
- metadata SHA-256 `14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3`;
- train bytes `4,000,000,000`;
- validation bytes `10,000,000`;
- metadata assembly v3, seed 8100, GPT-2 tokenizer, uint16.

The grouped v3 zero-GPU stage must re-hash these exact existing bytes before any H100 allocation.

## Frozen CORTEX-S architecture

Scientific architecture semantics are unchanged from v2:

- vocabulary `50,257`;
- width `512`;
- layers `24`;
- heads `16`;
- recurrent state `128` per layer;
- experts `8`;
- top-k `2`;
- logical expert hidden width `338`;
- full attention every `6` layers = `4/24` attention layers;
- trainable parameters `101,778,112`;
- selected expert-token assignments `25%` of dense expert assignments.

The production systems backend is now frozen as `physical_padded_grouped_bf16`. It materializes six zero runtime channels, giving physical expert width `344`, but those channels add **zero trainable parameters** and are not a new learned architecture component. On CPU/unsupported devices the code uses an equivalent grouped reference path.

The models are matched on total trainable parameters, not active FLOPs.

## Grouped production integration gate

Before the next H100 allocation, repository CI must prove on the exact branch head that:

1. the grouped production 100M builder has exactly `101,778,112` trainable parameters;
2. all 24 MoE blocks use the production grouped backend;
3. logical width remains `338` and physical runtime width remains `344`;
4. a real 100M CPU forward is finite;
5. grouped router statistics remain JSON-serializable without synchronizing counts in the hot training path;
6. existing CORTEX-S correctness tests still pass;
7. paid launcher syntax and unique v3 namespaces/triggers are locked;
8. normal full-repository CI, the CORTEX-S CPU gate, and the 100M/2B zero-credit gate all pass on the exact head.

Only after those checks may the branch be merged.

## H100 production preflight v3-grouped

Fresh engineering calibration seed: `2026090905`. It becomes consumed at the first H100 dispatch and may never be reused. It is not scientific evidence.

Unique namespaces and triggers:

- app: `cortex-s-v0-100m-2b-v3-grouped`;
- preflight root: `/vol/cortex-s-v0/100m-2b/preflight-v3-grouped`;
- paired full root: `/vol/cortex-s-v0/100m-2b/paired-seed8100-v3-grouped`;
- preflight issue title: `[modal-cortex-s-100m-2b-preflight-v3-grouped]`;
- full issue title: `[modal-cortex-s-100m-2b-full-v3-grouped]`.

The workflow is issue-triggered only; consumed v1/v2 issue titles are no longer accepted.

The preflight uses the actual grouped production builder inside the normal 100M trainer, with the same context, microbatch, accumulation, optimizer and compile mode intended for the full run. It may not start full training itself.

PASS requires all existing trainer gates:

- finite training loss and gradients;
- exact parameter count `101,778,112`;
- positive throughput;
- peak H100 allocation `<= 70 GiB`;
- conservative projected full envelope `<= 8,500 s`;
- exact source SHA unchanged;
- production backend metadata exactly `physical_padded_grouped_bf16`.

A PASS makes a **separate** `full-v3-grouped` issue eligible. It does not launch it automatically. A FAIL stops paid progression.

## Credit guard

Frozen pricing snapshot:

- H100 `$0.001097/s`;
- CPU `$0.00003942/core/s`;
- memory `$0.00000667/GiB/s`.

The full function keeps the `10,000 s` hard timeout. Protocol validation requires the conservative hard-timeout estimate to remain below the stated `$29` credit envelope. Live billing may differ; the primary gate is measured wall time.

## Checkpoint durability and no automatic retry

The full grouped stage retains atomic `latest.pt` checkpoints every nominal 200M reported tokens and explicitly commits the Modal Volume after each checkpoint. A checkpoint is salvage/evidence only.

`automatic_resume_authorized=false`: a failed full dispatch is consumed and will not be retried, rerun, or resumed without a separately governed recovery decision.

## Result interpretation

Primary quality comparison remains held-out NLL after the same 30,518 full optimizer steps. Report NLL/PPL, training throughput, training/total compute time, peak VRAM, router utilization, parameter mismatch, nominal token budget, optimizer steps and literal exposures.

If CORTEX-S does not beat historical NLL `2.7115590302149455`, say so and stop paid scientific progression. If it does beat it, classify the result only as promising `PAIRED_HISTORICAL_CONTROL_ADAPTIVE_EXPERIMENT` evidence. `breakthrough_claim_allowed=false` remains frozen and independent fresh-seed replication is still required.

## State-policy limitation

Training samples random 512-token windows and resets recurrent state for each independently sampled sequence. This experiment does not establish continual learning across batches and cannot support a continual-learning claim.
