# Pending first Colab research-infrastructure job

Classification: **CHM-v4 #1431 TRAIN-only engineering verification**.
Status: prepared for review; **not queued, approved or executed**. It does not
authorize training or reuse issue #1443.

| Field | Exact scope |
|---|---|
| Script | `colab_jobs/chm_v4_train_audit.py`, reviewed and merged into main |
| Source | Full merged-main commit SHA, verified by worker ancestry and remote bootstrap |
| Hardware | CPU only, Python≥3.11, Torch2.10.0+cpu, NumPy2.5.4; no GPU/precision substitution |
| Workload | Existing `train_only_factorized_audit()` unchanged; 2,048 TRAIN fixtures |
| Excluded scope | Zero optimizer updates, model forwards, heldouts, scientific seeds, token training budget or checkpoint resume |
| Limits | `--max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4`; one attempt, no retry |
| Export time | 135s provisioning/execution, reserve45s for cloud retrieval; separate bounded cleanup checks |
| Storage | Owner-approved private cloud remote; proposed `tam-drive:TAM Research Artifacts`; ≤64MiB bundle, verified cloud readback before stop |
| Acceptance | Unchanged audit assertions; final status ok; checksummed config/result/environment/job log; full cloud transport log; confirmed teardown; fresh post-attempt usage with zero assignments |
| Failure | FAILED with preserved verified cloud evidence when available; receipt consumed even on timeout/export/cleanup failure |
| Scientific meaning | Engineering mechanics and infrastructure only; no breakthrough or learned-performance claim |

Prerequisites: reviewed merge, configured/verified cloud write connection, live
WSL worker at existing limits, current positive compute balance and zero active
assignments. Last displayed108.89 units is not a fresh balance check. CPU timing
was measured locally; Colab bootstrap/provisioning and cloud duration remain
unmeasured. A timeout consumes this issue; another attempt requires a separate
diagnosis, issue and authorization.

After merge, generate the concrete issue body from main without allocating
compute (run from the checkout):

```sh
git fetch --quiet origin main
python3 - <<'PY'
import json, subprocess
sha = subprocess.check_output(['git','rev-parse','FETCH_HEAD'], text=True).strip()
subprocess.run(['git','cat-file','-e',sha+':colab_jobs/chm_v4_train_audit.py'], check=True)
print('<!-- colab-pro-job:v1 -->')
print('```json')
print(json.dumps({'commit':sha,'script':'colab_jobs/chm_v4_train_audit.py','gpu':'CPU'}))
print('```')
PY
```

Use that exact marker/JSON as the issue's first lines. Add this complete scope
and limits after the JSON for human review. The issue must be owner-authored;
leave it **without** `colab-approved` until explicit approval for this one job.
No unmerged implementation SHA or moving `main` string may be substituted.

Before authorization, verify in WSL:

```sh
python3 scripts/colab_pro_issue_worker.py --status
colab --auth=oauth2 sessions
rclone lsf 'tam-drive:TAM Research Artifacts' --max-depth 1
```

Confirm current storage read/write access locally, not merely an existing OAuth
login. After the attempt, check the receipt and cloud objects, verify the reported
hashes and inspect current `colab usage`/`sessions`; do not rerun a consumed issue.
