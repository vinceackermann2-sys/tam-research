# CORTEX-S v0 — 100M / 2B paired historical-control preregistration v2

Status: **frozen after CPU fingerprint recovery and before any H100 calibration**.

The original paid preflight trigger `#761` is consumed and will never be rerun. It stopped in the CPU-only corpus gate because the original protocol had frozen SHA-256 values that had not actually been established by the historical Transformer run. No H100 calibration and no CORTEX-S training step occurred in `#761`.

Issue `#767` then ran a new single-use CPU-only fingerprint diagnostic. It verified the existing historical corpus metadata and exact byte sizes and observed the real train/validation/meta SHA-256 values. That diagnostic explicitly did not authorize H100 or full training. This v2 document freezes those observed values before the first H100 calibration.

No reserved fresh CORTEX-S scientific seed has been consumed.

## Classification

This is an **adaptive paired historical-control experiment**, not a blind fresh-seed trial.

The repository's matched 100M Transformer was completed earlier and its outcome was available before CORTEX-S was scaled. Therefore even a strong CORTEX-S result cannot by itself be called a breakthrough. The purpose of this run is to decide whether the architecture warrants an independent replication.

## Frozen comparison

Historical control: repository issue `#140`, Transformer 100M, seed `8100`.

CORTEX-S paired run:

- seed: `8100`, pairing the random-window generator convention with the historical control;
- nominal training-token budget: `2,000,000,000`;
- validation tokens: `5,000,000`;
- context: `512`;
- microbatch: `64`;
- gradient accumulation: `2`;
- global batch: `128`;
- optimizer: AdamW, betas `(0.9, 0.95)`;
- peak LR: `3e-4`;
- weight decay: `0.1`;
- warmup fraction: `0.02`;
- precision: CUDA bf16 autocast;
- same immutable pretraining directory and exact corpus bytes as the Transformer;
- same final validation seed/batch convention as the repository trainer.

No second Transformer will be trained. Reusing the completed control is materially cheaper and avoids introducing a second control implementation.

### Important token-accounting correction

The repository trainer implements the 2B budget with full optimizer batches:

- tokens per optimizer step = `64 × 512 × 2 = 65,536`;
- optimizer steps = `ceil(2,000,000,000 / 65,536) = 30,518`;
- literal full-batch token exposures = `30,518 × 65,536 = 2,000,027,648`.

The historical Transformer reports a capped `tokens_seen = 2,000,000,000`, but its final optimizer step was a normal full batch. Therefore the fairest comparison is to run CORTEX-S for the same `30,518` full optimizer steps. Both models have a nominal 2B budget and `2,000,027,648` literal full-batch exposures; the overshoot is only `27,648` tokens (`~0.00138%`). We will not falsely describe the run as literally exactly 2,000,000,000 exposures.

## Frozen corpus fingerprints

CPU-only fingerprint-v2 evidence: issue `#767`, workflow run `34271912952`.

- train SHA-256: `93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b`;
- validation SHA-256: `ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760`;
- metadata SHA-256: `14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3`;
- train bytes: `4,000,000,000`;
- validation bytes: `10,000,000`;
- metadata contract: assembly v3, seed 8100, GPT-2 tokenizer, uint16, 2B train tokens and 5M validation tokens.

These values were observed from the already-existing corpus; the corpus was not rebuilt or changed.

## Frozen 100M CORTEX-S architecture

- vocabulary: `50,257`;
- width: `512`;
- layers: `24`;
- heads: `16`;
- recurrent state: `128` per layer;
- experts: `8`;
- top-k: `2`;
- expert hidden width: `338`;
- full attention every `6` layers, therefore `4/24` attention layers;
- total trainable parameters: `101,778,112`;
- historical Transformer parameters: `101,803,520`;
- absolute mismatch: `25,408` parameters, about `0.025%`;
- theoretical selected expert-token assignments: `25%` of dense expert assignments.

The models are matched on **total trainable parameters**, not active FLOPs. Any speed/efficiency advantage must be measured; it is not assumed from sparsity.

## State policy limitation

Training uses random 512-token windows. Recurrent state is reset at each independently sampled training sequence. This run tests the CORTEX-S architecture inside a normal language-model training regime; it does **not** establish continual learning across batches and cannot support a continual-learning claim.

## Zero-credit gates before H100

Before any H100 function is permitted:

1. instantiate the real 100M CORTEX-S on CPU and measure actual parameter count;
2. require parameter mismatch <= `0.5%`;
3. run a real 100M forward pass on CPU and require finite logits;
4. run same-mechanism forward/backward on a reduced CPU model and require finite gradients;
5. syntax-check the paid launcher and trainer;
6. require normal repository CI and both dedicated CORTEX-S CPU workflows to pass;
7. re-hash the exact existing corpus against the issue `#767` observed fingerprints before allocating H100.

## H100 calibration gate v2

Calibration seed `910001` is engineering-only and becomes consumed by the first H100 preflight-v2. It is not scientific evidence.

The preflight uses the exact production 100M graph, context, microbatch, gradient accumulation and optimizer. It may try the historical Transformer's compile mode; if compilation fails it may make one eager fallback measurement. It may never start the full paired run itself.

Full progression requires all of:

- finite loss and gradients;
- exact expected parameter count;
- positive measured throughput;
- peak H100 allocation <= `70 GiB`;
- conservative projected full compute <= `8,500 seconds`;
- exact source SHA unchanged between preflight and full launch.

The consumed v1 preflight namespace is not reused. v2 uses:

- preflight root: `/vol/cortex-s-v0/100m-2b/preflight-v2`;
- paired run root: `/vol/cortex-s-v0/100m-2b/paired-seed8100-v2`;
- preflight issue title: `[modal-cortex-s-100m-2b-preflight-v2]`;
- full issue title: `[modal-cortex-s-100m-2b-full-v2]`.

## Credit guard

Pricing snapshot recorded in protocol code:

- H100: `$0.001097/s`;
- conservative Starter CPU assumption: `$0.00003942/core/s`;
- conservative Starter memory assumption: `$0.00000667/GiB/s`.

The full function has a hard timeout of `10,000 seconds`. The protocol validates that this conservative hard-timeout compute estimate is below the stated `$29` credit envelope. Live billing can differ, so the primary fail-safe is the measured calibration plus wall-clock ceiling.

## Checkpoint durability

The full run writes an atomic `latest.pt` every nominal 200M tokens. The Modal wrapper now explicitly commits the mounted Volume after each completed checkpoint, because uncommitted Volume writes are not guaranteed durable outside the running container.

A durable checkpoint is **salvage/evidence only**. It does not authorize an automatic retry, rerun, or cross-dispatch resume. If the single full attempt terminates before completion, the experiment stops under the existing governance until a separate recovery decision is explicitly authorized.

## Primary result

Primary quality comparison is held-out language NLL after the same 30,518 full optimizer steps and matched training-byte/random-window protocol. Historical Transformer reference:

- NLL: `2.7115590302149455`;
- perplexity: `15.053722884190283`;
- training throughput: `321151.5755348581 tok/s`;
- training seconds: `6227.609573988244`;
- total compute seconds including compile: `6478.327432424761`;
- peak VRAM: about `11.896 GiB`.

Report CORTEX-S final NLL/PPL, training throughput, total compute time, peak VRAM, actual router utilization, parameter mismatch, nominal token budget, optimizer steps and literal full-batch exposures.

## Interpretation gate

A CORTEX-S result is interesting enough for independent replication if it is finite and its matched-step held-out NLL is strictly below the historical Transformer's `2.711559...` while measured systems cost remains practical.

Even then, the paired adaptive run remains `PAIRED_HISTORICAL_CONTROL_ADAPTIVE_EXPERIMENT` with `breakthrough_claim_allowed = false`. It is at most promising candidate evidence that warrants a fresh independent protocol.

If it loses on matched-step NLL or the preflight fails the budget/systems gate, stop paid progression. Do not spend reserved fresh seeds just to rescue the result.

## Fresh replication remains untouched

Reserved seeds `48131`, `48132`, `48133` remain unused. They may only be used after this development comparison justifies a separately frozen independent protocol, including a modern recurrent/state-space baseline and additional data/domain checks.
