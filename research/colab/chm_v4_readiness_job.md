# Pending Colab CPU source/artifact preflight — existing issue #1445

Infrastructure verification only. **Not approved or executed.** Continue the
existing owner issue; no duplicate job identity. #1443 remains consumed.

| Field | Exact scope |
|---|---|
| Commit | `ea062eca77ad58669a85cabf6a6528ba0623bfad`, reviewed #1444 source, only after merged into main |
| Script | `colab_jobs/chm_v4_readiness.py`, original bytes retained |
| Job ID | `chm-v4-colab-source-artifact-preflight-20261013-v1`; atomic global tag, no retries |
| Hardware | CPU, Python>=3.11, standard library; no Torch install/import or GPU |
| Workload | Exact-commit checkout, submitted-script SHA256 check, three frozen CHM source blob/syntax checks |
| Excluded scope | Zero episodes, updates, model forwards, heldouts, scientific seeds or resume |
| Limits | `--max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4`; unchanged |
| Storage | Proposed GitHub readiness-report mode on owner issue #1445; public nonsecret allowlist only, ≤128KiB input bundle / ≤24KiB report; no retained local artifact or Drive requirement |
| Acceptance | Source passes; all four readiness member contents plus original hashes preserved, independent issue-comment GET/digest verified before stop; zero assignments and fresh post-attempt usage |
| Failure | FAILED and consumed; retain available verified cloud evidence and attempt teardown |
| Meaning | Infrastructure only; no learned capability or breakthrough claim |

Exact pending JSON:

```json
{"commit":"ea062eca77ad58669a85cabf6a6528ba0623bfad","script":"colab_jobs/chm_v4_readiness.py","gpu":"CPU"}
```

Requires original #1444 gates, reviewed storage follow-up #1446 and merged-main CI,
source ancestry, explicit public-readiness retention approval, authorized live worker,
fresh positive balance/zero assignments, then explicit one-job approval.
Historical balance is not fresh evidence. No live Colab/storage attempt was
verified here. Allocation/execution/export share180s with up to45s reserved for
retrieval; bounded cleanup/account checks can take extra time. This is not a
compute-unit cap.

After review/merge, on the owner's authorized WSL or trusted Linux host:

```sh
python3 scripts/colab_pro_issue_worker.py --status
colab --auth=oauth2 sessions
python3 scripts/colab_pro_issue_worker.py --poll --max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4 --artifact-github-readiness
```

The owner issue includes a proposed standalone
`<!-- colab-public-readiness-artifacts:v1 -->` marker. It is required for this
backend but grants no allocation without the separately approved label and gates.
GitHub preflight is read-only; actual one-time report POST and independent GET
must succeed during the approved attempt before teardown. All JSON values,
encoding and the canonical log are allowlisted; raw CLI logs/checkpoints/source
ZIP never enter the public issue. Original member bytes/manifest can be
reconstructed and rehashed from the report. A failure retains only a fixed
incomplete provenance receipt and cannot claim complete retrieval.

The current cloud host has official CLI 0.7.4 but no Colab OAuth cache and no
worker. Actual `--status` returns `account=null`, `account_verified=false`,
`worker_running=false`; current balance is unknown. GitHub auth cannot supply
Google login. Never share credential values through chat. Leave #1445 without
`colab-approved` until all prerequisites and this single attempt are approved.

The separate 256-update #1436 TRAIN microfit remains unapproved. Its original
merged-main source CI and ability to finish setup/training/export within180s
remain unresolved. Do not truncate budgets or reuse consumed attempts.
