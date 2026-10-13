# Five-track continuation with Colab as the future compute backend

Prepared 2026-10-13 from main `107342cb15fa53e81948869eac450f2c69d1419a`
and read-only remote branch inspection. The owner's clarified architectures are
**CONV7, PGW, CPW, RLT, CHM-v4**. CPW is separate from PGW and CHM; never assign
another project's evidence to it. GitHub issue/API reads were denied in this
cloud task, so the state below is frozen repository evidence, not a fresh
verification of issue approvals or callbacks. Other ChatGPT project chats are
not available to this task.

Implementation and CPU correctness checks are authorized. New training,
replication, scaling, scored heldouts and paid probes still need a concrete
approved job with explicit limits. The worker remains CPU/T4, 180 seconds,
one attempted issue per invocation, maximum accepted timeout 300 seconds.
Future execution uses Colab; no Modal jobs or fallback are launched.

| Track | Latest recoverable state | Next permitted preparation / blocker |
|---|---|---|
| CONV7 | Cortex attention-8 plus 16 causal depthwise conv7 mixers. Main includes PR #1423 data-stream parity. New source-lock branch `research/cortex-attention8-conv7-execution-lock-readonly-v1` at `6e0173fb719af30f78df9f8197f01dd03bf3b4b3` is unmerged. | CPU readiness, geometry and synthetic sampler checks. Proposed 3-arm 200M screen requires H100 80GB/BF16, a new seed/job ID, corpus transfer and explicit paid approval; it cannot fit T4 memory or this deadline. |
| PGW | Main includes Stage-2B5 static corpus/metrics, issue #1429; source branch `research/pgw-v3-stage2b5-corpus-metrics-v1` at `311b18c4d618b79121574d23aeb0f91f17c55cd9`. | CPU corpus integrity: 5,760 samples / 522,240 tokens and algorithmic controls. No optimizer, training seed, checkpoint or learned claim exists. Training fairness, seeds, optimizer and budgets are still unfrozen. |
| CPW | Canonical separate line `experiment/cpw-v0-brain-predictive-workspace-20261005` at `aa8bdafa47e27a91ce64ba3771f1515603f815f9`; latest `audit/cpw-binding-gradient-1335` at `7b0579cf08304779f1df8267f032e944a27caf6a`. Both unmerged into main. | Corrected #1307 eight-slot binding task; #1335 random-init five-arm gradient audit, no updates. Review/import its dependency closure into main first. Engineering init 1335001/source 1345001 are not scientific seeds. |
| RLT | Latest BB decision on `exp/rlt-ax-bounded-transformer` at `7cff4302caed515b19f683ba1a0f9be10a065659`; `exp/rlt-colab` at `b1bac447f9327ec15e0b17f60df03519988aba6e` is older AW. Unmerged into main. | Preregister an independent CPU training-stability panel and richer inference tasks; do not replay AW–BB. Three-way 315+ second training already exceeds both current 180 seconds and 300-second accepted maximum. |
| CHM-v4 | Main #1431 factorized source/copy/gate audit, following #1430 TRAIN-only gradient engineering. DEV16/TEST32 remain unopened. | Prepared first Colab CPU engineering job calls unchanged `train_only_factorized_audit()`: 2,048 TRAIN fixtures, no updates/forwards/heldouts/scientific seeds. Needs reviewed main script, cloud storage and a live local worker. |

## Frozen results and scientific gates

- **CONV7:** predecessor #1300/#1331 is SCREEN_FAIL: parent NLL
  2.7680209398269655 vs Transformer 2.722934193611145, delta +0.0450867462
  against +0.015; systems ratio 1.074995 passed. Seeds 60231/60232 consumed.
  CONV7 itself is untrained. Proposal keeps 3,052 updates / 200,015,872 tokens,
  micro64×accum2/context512 and the prefix of the 30,518-step 2B cosine, warmup610,
  AdamW LR3e-4/betas(.9,.95)/weight-decay.1/clip1. Draft selection gates are not
  an execution approval. Historical 42.91–44.37 GiB allocations exceed T4 memory;
  BF16/SM90 hardware cannot silently become T4 FP16.
- **PGW:** #1181/#1201/#1213 remain UNSUPPORTED. Stage-2B5 algorithmic control
  accuracies are integrity evidence only. Proposed later training requires ≥3
  fresh paired seeds, matched capacity/compute and frozen data/loss; draft gates
  include ≥80% macro/present, ≥90% missing, ≥5-point mean advantage and positive
  paired delta in ≥2/3 seeds. Current authorization flags deny training/scaling.
- **CPW:** old v4/v5 recall was query-blind/position-confounded (#1301). Corrected
  eight slots and identical-prefix four-query groups must remain intact. Frozen
  provisional #1330 screen requires three new seeds, five arms, 1,500 updates
  at micro32×accum2, 512 examples/delay plus 512 query groups; Transformer long≥80%
  and delay256≥70%, sequence-only≤5%; candidate ≥2/3 rows with long≥80%, delay256≥70%,
  +40 points and lower NLL than sequence-only per long delay, all-four≥70%.
  This readout does not certify provenance or authorize training. Old v4 seeds
  1273001/1273101–103 and v5 1291001/1291101–103 stay consumed.
- **RLT:** AW weakens at gap127; AY fresh training collapsed; BA carry and BB
  inference-only interventions succeeded on synthetic recall. Training stability
  is unresolved and real-data AK–AQ favored Transformer. Protected replication
  seeds 58232/58233 and 250M/5B scaling remain unapproved. AY weights are missing;
  do not reconstruct them by rerunning. BA checkpoint is recoverable from Git:
  `experiments/rlt/cpu/checkpoints/rlt-carry-intervention-20261010-ba/carry.pt`,
  120,530 bytes, SHA256 `74fe2bb1cfb74242aaba3d0eb8ac5df13c1a421a7d50054a0d7dbf503fb4149b`.
- **CHM-v4:** historical `CHM_V3_100M_DAEC_STAGE_C_STOP` and consumed seed2013161
  remain. #1430 uses extra TRAIN address supervision and soft first-hop mixture;
  it is not a fair learned Transformer comparison or hard two-hop proof. Future
  science requires frozen matched supervision/compute, independent-entity
  uncertainty and new scope. With 32 no-match test entities, zero FP still gives
  ~8.94% one-sided95% upper bound, so that panel cannot establish <5% FPR.

The repository's [breakthrough protocol](breakthrough_protocol.md) remains
unchanged. Infrastructure readiness, synthetic mechanics or a selected favorable
seed cannot establish a breakthrough. Do not run an open-ended search until
one wins; advance through reviewed, prospectively frozen gates and preserve all
negative results.

## Modal dependency and administrative transfer inventory

No existing Modal volume appears inside Colab. Transfers require source access,
owner-approved cloud storage and an explicit **read-only storage export**; they
do not permit a Modal compute function, rerun, resume or rescoring.

| Track | Historical launcher / source | Required immutable storage |
|---|---|---|
| CONV7 predecessor | `modal_cortex_reduced_attention_v2_attention8_pair1_successor_100m_2b_v1.py`; H100!, Python3.11/Torch2.10/NumPy2; package sources and Actions callback. | `tam-research-data:/data/tam100m-2b-curated-v1`; historical root `/cortex-s-v0/100m-2b/reduced-attention-v2-attention8-pair1-successor-v1`. Trainer has no model save path; reported periodic checkpoints are metrics, not recoverable weights. |
| PGW historical core | `modal_pgw_core_mechanism_1201_app.py`; H100!, 3h; frozen synthetic/LM runner and Actions callbacks. | `tam-research-data:/data/fineweb-edu-gpt2`; `/pgw-v2/local-predictor-panel-v1`, ATTEMPT/RESULT/FAILURE and per-arm latest.pt/summary/metrics. Static Stage-2B5 needs none of these. |
| CPW historical v5 | Branch `modal_cpw_v5_afm_1290.py`; H100!, 3h/8CPU/32GiB; deterministic synthetic data, issue callbacks #1297/#1294/#1290. | `tam-research-data:/cpw-v5/associative-fast-memory-v1` and `/cpw-v4/delayed-associative-recall-v1`. Corrected audit needs no historical weights/datasets. |
| RLT historical scaling | Branch `experiments/rlt/modal_*.py`; A100/A100-80GB, HF FineWeb-Edu/GPT2 dependencies, data hashes and RLT_*_RESULT_B64 callbacks. | `tam-rlt-scale15m-checkpoints-account2:/checkpoints/<job_id>`. Synthetic AW–BB needs no Modal corpus; BA weights are in Git, AY missing. |
| CHM historical Stage-C | `modal_chm_v3_100m_daec_stage_c_1323_v1.py`; explicitly L4, 9,600s, Python3.11/Torch2.10/NumPy2/tiktoken; package sources, volume commits, workflow callback. | `tam-research-data:/data/tam100m-2b-curated-v1`; `/chm-v3/100m-daec-stage-c/issue-1323/seed-2013161-v1`, checkpoints/{local,raw_eiem,daec_eiem}/step-2048.{pt,json}, reservation/gate/failure/results. Consumed run cannot be ported to T4 as a retry. |

Transfer procedure:

1. Inventory source account/volume/path and existing files without executing
   models or altering source evidence. Include attempt receipts, configurations,
   full metrics/logs, tokenizer/meta files and optimizer/RNG state where present.
2. Owner or existing storage administrator exports immutable bytes through
   available storage access. No Modal GPU function is started. If source access
   was removed, stop and document the missing source access; do not regenerate.
3. Create byte-count/SHA256 manifests and retain original issue/run/commit IDs.
   Curated corpus requires train.bin 4,000,000,000 bytes and val.bin 10,000,000 bytes;
   hashes: train `93e9cb0b7076a4ddd855fc696f657ea62592b8a03a05220be99c402f9043265b`,
   val `ae0bc5adf36d0aa8e55e5e3903401d3f114b93037f43221944b4e88c5d1a5760`,
   meta `14bbbcf0ab0b8cba374074ef8ccb80a04ecead06c74780f35a1becd5aef1b8f3`.
4. Upload to agreed private cloud storage, independently retrieve/hash again,
   then stage only the exact approved files into Colab for a future reviewed job.
   These >4GB datasets exceed the small-job exporter; review separate large-file
   transport before allocating training. Keep old scientific namespaces intact.

## Current implementation and validation limits

This PR adds one standalone CHM engineering job and generic small-job cloud
export/preflight/cleanup. It does not import unmerged CPW/RLT models, rewrite
scientific launchers, disable historical records or silently modify hardware,
precision, data, seeds, token budgets, optimizer settings or acceptance criteria.

CPU validation covers actual CHM fixtures, PGW corpus/preflight, CONV7 causality/
readiness/parity, and simulated CLI launch/export/timeout/cleanup. Live Google
OAuth, Drive writes, current balance, WSL worker and Colab allocation remain
unverified here. No Modal or Colab job was executed. The cloud folder proposal
must be approved/configured, and the new script must be merged into main before
an issue may receive `colab-approved`.

Validation in this task (no Colab allocation or scientific training):

| Source | Checks | Outcome |
|---|---|---|
| Changed bridge + standalone job + main CHM-v4 | `test_colab_pro_issue_worker`, `test_colab_chm_v4_job`, four `test_chm_v4_*` files; exclude the single optimizer smoke | 150 passed, 1 deselected |
| Main PGW-v3 | Stage-2B5 corpus and Stage-2B4 preflight suites | 42 passed |
| Main CONV7 | Conv7 architecture, proposal, LR prefix, readiness, sampler parity, dispatch dry-run | 46 passed |
| CPW `7b0579cf08304779f1df8267f032e944a27caf6a` | Isolated `/tmp` git-archive; gradient1335/readout1330/safety1326/science1321/task1307 suites; exclude optimizer parametrizations | 65 passed, 5 deselected |
| RLT `7cff4302caed515b19f683ba1a0f9be10a065659` | Isolated `/tmp` git-archive; bounded model, two scan suites, sliding window, checkpoint retention guard | 112 passed |

Total: **415 passed, 6 deliberately deselected optimizer checks**. Branch-export
checks do not establish integration of CPW/RLT onto main. Syntax checks and
whitespace checks passed. The real bootstrap helper also fetched/verified
`107342cb15fa53e81948869eac450f2c69d1419a` and installed its pinned requirements
in this CPU environment; standalone runtime/cloud transport remains simulated.
