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
| Storage | Owner-approved private cloud folder; proposed `tam-drive:TAM Research Artifacts`, not yet configured; ≤32MiB bundle |
| Acceptance | Source passes; config/metrics/provenance/run log and transport log checksum-verified by cloud readback before stop; zero assignments and fresh post-attempt usage |
| Failure | FAILED and consumed; retain available verified cloud evidence and attempt teardown |
| Meaning | Infrastructure only; no learned capability or breakthrough claim |

Exact pending JSON:

```json
{"commit":"ea062eca77ad58669a85cabf6a6528ba0623bfad","script":"colab_jobs/chm_v4_readiness.py","gpu":"CPU"}
```

Requires original #1444 gates, reviewed cloud follow-up #1446 and merged-main CI,
source ancestry, approved cloud write/retrieval connection, live WSL worker,
fresh positive balance/zero assignments, then explicit one-job approval.
Historical balance is not fresh evidence. No live Colab/cloud connection was
verified here. Allocation/execution/export share180s with up to45s reserved for
retrieval; bounded cleanup/account checks can take extra time. This is not a
compute-unit cap.

Verify on the owner's WSL machine:

```sh
python3 scripts/colab_pro_issue_worker.py --status
colab --auth=oauth2 sessions
rclone lsf 'tam-drive:TAM Research Artifacts' --max-depth 1
```

Also verify a small upload/readback in the chosen folder locally; listing or
Colab OAuth alone does not prove write access. Leave #1445 without
`colab-approved` until all prerequisites and this single attempt are approved.

The separate 256-update #1436 TRAIN microfit remains unapproved. Its original
merged-main source CI and ability to finish setup/training/export within180s
remain unresolved. Do not truncate budgets or reuse consumed attempts.
