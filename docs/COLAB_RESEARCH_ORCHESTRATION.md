# Five-track continuation and Colab migration — 2026-10-13

Implementation / CPU validation only. No Colab runtime, GPU, 256-step microfit,
held-out score, scientific seed, replication or scale-up is authorized by this PR.
User mapping: **SUBCONAI = PGW; UNK AI = attention-8 causal conv7**.
Audited main: `107342cb15fa53e81948869eac450f2c69d1419a` (PR #1442).
Read COLAB_PRO_CHATGPT_BRIDGE.md, COLAB_COMPUTE.md, CONTINUATION_RUNBOOK.md,
and breakthrough_protocol.md first. The historical continuation runbook's #67
and old Modal credit figures are archival, not current launch authority.

## Separate continuation points

| User project | Authoritative source at audit | Last state / next prerequisite | Colab eligibility |
|---|---|---|---|
| CPW | `experiment/cpw-v0-brain-predictive-workspace-20261005`, `aa8bdafa47e27a91ce64ba3771f1515603f815f9` | Corrected-binding #1319; five-arm harness #1321, safety #1326, provisional readout #1330; #1335 source-to-query gradient diagnostic is next, not yet on canonical head | Source confined to branch; review/import into main before bridge submission. H100/BF16 protocol cannot become T4 |
| SUBCONAI / PGW | main, Stage-2B5 PR #1432 merge `ceb2008b24441eb650af7ebb2f4ca96b7fc456ec` | #1437 Stage-2B6 fairness/authorization design: active parameter parity, auxiliary objective parity, seeds/budgets and collision audit | CPU design/preflight; no scientific optimizer authority |
| UNK AI / conv7 | `exp/cortex-attention8-causal-conv7-cpu-prototype-v1`, `b90beeed8a0a2d41f2a1f5e77d1f47d7dc3870ff` | Untrained causal mixer, 101,803,536 parameters; prior attention-8 successor #1300 / seed 60232 consumed. Needs reviewed prototype and preregistered same-hardware comparison | Not main; no T4 substitution for H100 comparison, no assigned seed |
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
- Storage proposed for approval: the **existing local WSL worker state directory**,
  `~/.local/state/colab-pro-chatgpt/issue-N-bundle.zip`, beside its durable receipt
  and full CLI log. No Drive OAuth/mount or credentials in the runtime/repo.
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
dependency is the pinned CPU torch build; all transitives are recorded by pip in
the local CLI log. Historical 100M CHM parity is unresolved and is not bypassed.

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
separately agreed durable store/local directory, creates manifest with sizes,
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
python3 scripts/colab_pro_issue_worker.py --poll --max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4
```

Run `--status` in a second WSL terminal while worker polls. It probes the held
exclusive lock and reports actual worker settings plus freshly queried balance
and assignment count. Only this nonsecret JSON may be shared; OAuth, session
URLs/endpoints and raw logs stay local. No job is labeled for execution here.

## Validation scope

CPU syntax, CLI argument limits, standalone stdlib preflight without torch,
source pins, TRAIN-only data order/token budget, archive digest/provenance checks,
remote duplicate reservation rejection, local lock readiness, successful retrieval,
corrupt-artifact failure and timeout cleanup are tested. Lifecycle tests mock the
Colab CLI; they are not a new live Colab attempt. Relevant existing CHM/PGW tests
pass (79 tests). The new bridge contract suite has 24 passing tests.
