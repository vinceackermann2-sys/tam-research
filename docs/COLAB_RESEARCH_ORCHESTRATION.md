# Five-track continuation and Colab migration — 2026-10-13

Implementation / CPU validation only. No Colab runtime, GPU, 256-step microfit,
held-out score, scientific seed, replication or scale-up is authorized by this PR.
Owner clarified names: **CONV7, PGW, CPW, RLT, CHM-v4**. The earlier informal
SUBCONAI/UNK labels are not used to merge architecture histories.
Audited main: `107342cb15fa53e81948869eac450f2c69d1419a` (PR #1442).
Read COLAB_PRO_CHATGPT_BRIDGE.md, COLAB_COMPUTE.md, CONTINUATION_RUNBOOK.md,
and breakthrough_protocol.md first. The historical continuation runbook's #67
and old Modal credit figures are archival, not current launch authority.

## Separate continuation points

| User project | Authoritative source at audit | Last state / next prerequisite | Colab eligibility |
|---|---|---|---|
| CPW | Canonical `aa8bdafa47e27a91ce64ba3771f1515603f815f9`; latest #1335 audit `7b0579cf08304779f1df8267f032e944a27caf6a`, PR #1338 | Corrected task, five-arm random-init audit completed. Dedicated diagnostic PASS, original full CI FAILED; next inspect that failure without replaying the audit | Source confined to branch; original failure and main integration remain gates. No paid authority |
| PGW | main, Stage-2B5 PR #1432 merge `ceb2008b24441eb650af7ebb2f4ca96b7fc456ec` | #1437 Stage-2B6 fairness/authorization design: active parameter parity, auxiliary objective parity, seeds/budgets and collision audit | CPU design/preflight; no scientific optimizer authority |
| CONV7 | Main includes CPU prototype/parity; latest source-lock PR #1433 `6e0173fb719af30f78df9f8197f01dd03bf3b4b3` is unmerged | Untrained causal mixer, 101,803,536 parameters; prior #1300 SCREEN_FAIL / seed60232 consumed | No T4 substitution for H100 comparison, no assigned seed |
| CHM AI | main, PR #1435 merge `143ffc39fa131da9336f5bac0fb94528400dc9e6` | V4 neural source/copy/abstention model. Next #1436 TRAIN-only capacity microfit; V3 #1323 STOP / seed 2013161 remain consumed | First portable small CPU workload. First perform no-training bootstrap/artifact job |
| RLT AI | `exp/rlt-ax-bounded-transformer`, `7cff4302caed515b19f683ba1a0f9be10a065659` | BA saved checkpoint, BB inference causal-state ablation completed. AW/AY/BA/BB consumed. Next preregistration for training stability and richer tasks | Preserve new branch over older `exp/rlt-colab`; migrate reviewed sources to main first, never replay BA/BB |

CPW and CHM are distinct. No result is reassigned between tracks. Branch-local
code stays untouched by this migration. PR #1435 original head CI #38081373070
PASS; PGW original merged-main CI #38081489818 PASS, run_attempt=1, directly
read back. PGW direct `candidate_corpus_manifest()` readback reproduces all four sealed
#1437 roots and 5,760 unique rows; structural CPU tests pass. No optimizer ran.
CHM original merge has no push CI returned by the live Actions API; this
remains an explicit missing gate that must be resolved in review, not rerun or
assumed passed. The new integration source and its eventual merged-main CI
are separate evidence. CHM merged-main first CI must still be separately established before
microfit approval; unrelated failing historical workflow files are not science.

## Orchestration policy

One eligible attempt at a time. Audit the live branch and immutable callbacks,
record the next prerequisite, propose the smallest informative workload, review
and merge code, verify the original CI gates, then request exact job authority.
Collect all results (including negative/invalid/infra failures), adjudicate only
against precommitted criteria, and update this ledger through a new PR. No
self-approving queue, adaptive rerun, seed search, automatic replication or
unlimited 'until breakthrough' training loop. A synthetic capacity pass only
opens a separately reviewed generalization proposal; breakthrough requires the
frozen multi-seed, multi-scale, equal-compute, second-corpus and mechanism gates.
No Modal execution or fallback. This PR does not create an unattended scheduler.

## First concrete Colab job: no-training CPU source/artifact preflight

- Script: `colab_jobs/chm_v4_readiness.py`.
- Literal unique job ID: `chm-v4-colab-source-artifact-preflight-20261013-v1`.
- Commit: the exact reviewed integration commit, **after it is an ancestor of
  main** and original source/head and merged-main CPU CI succeed. The PR body
  supplies the candidate commit; do not queue an unmerged script.
- One new owner-authored issue, approval label `colab-approved` only after
  explicit job approval. Never issue #1443 (consumed Tesla T4 CUDA probe).
- CPU; default worker `--max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4`.
- Zero optimizer steps, zero dataset episodes, no model import, no scientific
  seed, no CUDA or dependency installation. Public exact-SHA git checkout,
  worker-supplied SHA-256 submitted script check, three frozen CHM source blob/syntax checks.
- Files: configuration.json, metrics.json, provenance.json, run.log,
  SHA-256 member manifest and bundle.zip. Each is checksum verified on the worker
  before runtime stop. Missing/corrupt retrieval is FAILED, never SUCCESS.
- No retry or resume. Allocation/execution/collection share the 180s deadline;
  bounded cleanup/checks can take additional time. This is not a compute-unit cap.
- Storage: owner declined retained local artifacts. An approved private cloud
  destination is required. For this source-only job the GitHub readiness mode
  retains strictly allowlisted configuration/metrics/provenance/log contents and
  original hashes, with independent issue-comment readback before stop; Drive
  is optional. Full training bundles use separately agreed private cloud storage.
  The source ZIP/raw transport log are not retained by GitHub mode. The worker
  uses only temporary transfer files. Authentication,
  mandatory receipts and upstream CLI diagnostics remain local. No credential
  is sent to the runtime/repo. Existing #1445 is pending storage review; do not
  create another issue or reuse its literal job ID to bypass a consumed attempt.
- Before allocation: fresh `colab usage`, `colab sessions`, positive balance,
  no active assignments, local worker exclusive lock, remote single-use tag.
- After each attempt: successful download/member hashes/provenance, explicit
  stop, fresh usage and no remaining assignment. Unverified teardown blocks
  subsequent worker startup. Preserve the receipt, never delete it to retry.

## Prepared follow-on: #1436 microfit (not executed or approved)

`colab_jobs/chm_v4_microfit_1436.py` bootstraps exactly the approved commit, checks
its own bytes, installs `torch==2.10.0+cpu`, and runs the additive CPU experiment
module without changing any frozen model/benchmark/helper. Existing probe torch
2.11.0+cu130 is not used: the package/CPU CI contract is torch2.10.0, so this is
an explicitly disclosed CPU software environment, not comparable GPU telemetry.

Frozen engineering process RNG1436001; the unchanged model constructor retains
initialization RNG1377005. Neither is a claimed fresh scientific replication seed.
32 TRAIN episodes: family-major direct/overwrite/two_hop/no_match, entity0,
rotations0..7. Cycle all32 eight times: exactly256 single-example AdamW updates,
LR.005, betas(.9,.999), eps1e-8, wd0, no schedule/clipping, one CPU thread, FP32.
Exactly 18,560 input tokens from the frozen 32-item stream (a 2,320-token
cycle repeated eight times); reject a token-count mismatch. Record tokens, parameters, time, every loss, per-case first-source,
payload, copied answer, false-present/false-abstain diagnostics. One independent
negative entity despite eight identical rotations. Gate: 24/24 positive TRAIN
answers and abstention on that negative entity. No optional baseline is run;
extra TRAIN gold address supervision and soft first-hop remain confounds.

Terminal checkpoint contains model, optimizer, RNG, step, configuration and token
count; full metrics/losses/log/provenance and actual Python/dependency versions join the verified bundle. No restart,
restore, or holdout generation. Model/supervision/acceptance criteria are intact.
256-step CPU run, dependency installation and download **have not** been timed on
Colab within 180s; source preflight comes first. If it cannot fit, block rather
than truncate training or enlarge limits. Longer jobs need a reviewed deadline
implementation and separately approved limits; hard maximum remains 300s.

## Modal dependency trace and explicit transfer plan

CHM #1436 uses synthetic TRAIN data and a new tiny model; it has **no Modal
launcher, secret, volume, existing checkpoint or external corpus dependency**.
Public callback is the existing issue-worker filtered summary. Its only runtime
dependency is the pinned CPU torch build; transitives are recorded in cloud job
artifacts. Historical 100M CHM parity is unresolved and is not bypassed.

For future CPW/conv7/PGW/RLT language workloads, inventory the exact frozen launcher
and transitive imports on their source commit: Python/torch/CUDA/Triton versions,
GPU class/VRAM/precision, data builder and tokenizer revisions, exact train/val
bin files and SHA-256 hashes, optimizer/sampler RNGs, compiler namespace,
checkpoint format and remote callback contract. Main's historical scaling uses
Modal `tam-research-data` with `/vol/data/fineweb-edu-gpt2`, isolated
`/vol/scaling-runs/...`, `/vol/compile-cache/...`; these are **not copied by CLI**.
Track-specific paths must be read from that track's pinned launcher, not guessed.
RLT BA's checkpoint is already archived in its Git tree at SHA-256
`74fe2bb1cfb74242aaba3d0eb8ac5df13c1a421a7d50054a0d7dbf503fb4149b`.

If source data is only on Modal: owner exports exact files read-only to a
separately agreed private cloud store, creates manifest with sizes,
SHA-256s, source attempt/checkpoint IDs and license provenance, verifies every
export on the local machine, then reviews a Colab upload/bootstrap path that
verifies the same hashes before optimizer start. No Modal job is launched by
this PR; no attempt identity is resumed. An inference-only checkpoint transfer
still needs its own explicit scope. If old volume is inaccessible, block; a
regenerated corpus is not presumed identical. Do not migrate H100 compiler
caches to T4. Larger checkpoints require an approved storage path beyond 32 MiB.

## Local worker readiness is not yet verified from this chat

This environment cannot reach the owner's WSL localhost or local Google OAuth.
108.89 units is **historical**, not a current balance. #1443 confirmed Tesla T4,
CUDA and teardown; worker exited afterward. Do not claim an executing queue.
After review/merge, owner checks in WSL:

```sh
git pull --ff-only
python3 scripts/colab_pro_issue_worker.py --status
python3 scripts/colab_pro_issue_worker.py --poll --max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4 --artifact-github-readiness
```

Run `--status` in a second WSL terminal while worker polls. It probes the held
exclusive lock and reports actual worker settings plus freshly queried balance
and assignment count. Only this nonsecret JSON may be shared; OAuth, session
URLs/endpoints and upstream CLI diagnostics stay local; research artifacts/logs
use private cloud storage. No job is labeled for execution here.

The official Colab CLI 0.7.4 was also installed here in an isolated tool venv;
its actual status has `worker_running=false`, `oauth_cache_present=false`,
`account_verified=false` and no current balance. It cannot inherit WSL OAuth.
GitHub authentication alone cannot authorize the owner's Google account.
No unattended worker or login flow was started.

Separate successors #1447 (CPW stale test contract), #1448 (CONV7 immutable
manifest validation) and #1449 (RLT dimension types) passed their original
first-attempt CPU CI at exact heads recorded in the continuation ledger.
PGW and RLT prospective designs are linked there; no seeds/jobs are assigned.

## Validation scope

CPU syntax, CLI argument limits, standalone stdlib preflight without torch,
source pins, TRAIN-only data order/token budget, archive digest/provenance checks,
remote duplicate reservation rejection, local lock readiness, successful retrieval,
corrupt-artifact failure and timeout cleanup are tested. Lifecycle tests mock the
Colab CLI; they are not a new live Colab attempt. Relevant existing CHM/PGW tests
pass. See [COLAB_RESEARCH_CONTINUATION.md](COLAB_RESEARCH_CONTINUATION.md) for this
follow-up's current validation counts and live issue readbacks.
