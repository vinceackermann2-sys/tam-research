# Google Cloud GPU compute

This is a separate backend for agent-selected GPU jobs. Modal workflows and credentials are unchanged.

## Current activation state

The keyless GitHub connection is installed and verified by [the successful connection check](https://github.com/vinceackermann2-sys/tam-research/actions/runs/38256328433). The cloud identities, closed-ingress VPC, private results bucket and permanent budget baseline are created. GPU execution remains disabled pending GPU quota. On 2026-10-12 the selected project (`project-c40afe76-9188-4739-b66`) had global GPU quota 0; us-central1 L4, T4 and A100 quotas were also 0. No GPU was provisioned during setup.

The single-GPU global quota request `tam-single-gpu-global` was submitted and declined by Google: preferred value 1, granted value 0. The quota API reports `NOT_ENOUGH_USAGE_HISTORY` for the global quota and relevant GPU types. Google must approve quota before this backend can run; never spend credits on unrelated resources to bypass this restriction. Once quota is approved, recheck the available hardware and remaining credits before enabling `GCP_GPU_ENABLED` and running a short GPU smoke check.

Billing was verified as a paid account on 2026-10-12, with 100% of the welcome credit available: SEK 2,985.81. Google lists its expiry as November 21, 2026. The launcher stops accepting jobs ahead of that date, at November 20, 2026 00:00 UTC. A project budget named `TAM compute lifetime 260 USD credit allowance` tracks gross usage, excludes all credits, does not reset monthly, and alerts at 50%, 80% and 100%. Its amount is SEK 2,587, rounded down from 260/300 of Google's recorded original credit value. This is an alert budget; Compute Engine has no enforced spend-cap status.

Google's non-billable free trial does not permit GPU VMs. The account owner must upgrade billing if still on that trial, then obtain GPU quota. The remaining unexpired welcome credit is retained after upgrade. Do not upgrade billing automatically.

## Requesting a job from chat

An agent with GitHub issue-write access can open an issue in this repository with the exact title `[gcp-run]` and a JSON body:

```json
{
  "gpu": "L4",
  "minutes": 10,
  "source_sha": "FULL_CURRENT_MAIN_COMMIT_SHA",
  "script": "scripts/gcp_smoke.py",
  "args": [],
  "zone": "us-central1-a"
}
```

The agent chooses L4, T4, A100 or H100 according to the experiment, memory requirements and available quota. It must select the smallest sufficient device and duration. Each machine has one GPU. H100 is 80 GB; A100 here is 40 GB. These are Spot machines: interrupted runs are incomplete, and there are no automatic retries, paid reservations or hardware substitutions. Zones are limited to us-central1-a/b/c/f; hardware availability still varies by zone.

Only owner-created issues trigger jobs. For future ChatGPT sessions, the GitHub connection must be able to create an issue as the repository owner. Repository access alone does not make GPU tools appear in every chat. The workflow also supports manual dispatch on main.

Manual dispatch with `check_connection=true` checks keyless login and reads the budget baseline while paid execution stays disabled. It always skips VM creation. The regular source/config validation and guard tests still run.

Use a committed Python experiment entry point, not a Modal wrapper or `.remote()` invocation. Existing Modal volumes and checkpoints do not move to Google Cloud automatically. Save outputs under `TAM_RESULTS_DIR`; maximum returned output is 100 MiB. The workflow posts a result link on the issue and uploads a GitHub artifact. GCS results expire after seven days.

## Safety limits

- Default 30 minutes; each request specifies 5–240 minutes, including boot and dependency installation.
- Google Compute Engine independently deletes the VM at its configured maximum runtime, even if GitHub disappears. Boot disk auto-deletes with the VM.
- GitHub serializes requests; a pre-existing VM blocks another launch. Completion/failure attempts immediate deletion. The worker also powers off when finished.
- Current-main full SHA required before authentication or provisioning; no shell command is taken from issue text.
- Dedicated VPC with no permitted ingress, blocked project SSH keys and no GitHub credentials on the VM.
- Keyless Workload Identity Federation is restricted to this repository, main and this exact workflow. Worker identity only reads its result bucket and creates result objects. Launcher cannot edit IAM or delete reservation records.

## $260 allowance and billing limitation

The launcher uses a **non-resetting $260 USD lifetime reservation allowance**, not a monthly allowance. $20 is withheld for safety, and an initial baseline must be recorded before enabling jobs. It reserves the full requested duration plus five minutes and $2 overhead per job. Conservative hourly allowances are $2 for L4/T4, $8 for A100 and $20 for H100. Reservations are kept even when provisioning fails; do not automatically refund them. Storage ledger failures and budget exhaustion block launches. These rates are allowances, not quoted Google prices.

This guard limits this launcher's reservations. It **cannot guarantee that the Google invoice or total credit consumption never exceeds $260**. Compute Engine has no native spend-cap budget; billing alerts and automated billing shutdown have reporting delays. Other services/projects, price changes, arbitrary experiment network traffic, and retained storage can accrue charges outside these reservations. Google billing budgets must be based on gross usage (exclude promotional credits), otherwise $300 credits can hide consumption.

The user's $260 limit applies to agent-launched GPU compute. This is enforced as an upfront reservation ceiling for those jobs, with a safety reserve inside that amount. It is not an account-wide invoice cap. Enabling the backend requires verified GPU quota, verified credit expiry and remaining credit, reviewed price allowances, and a billing alert budget. An agent must never lift the budget, reset the ledger, increase the ceiling, or turn on paid execution to bypass a rejection.

Official references: [VM runtime limits](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime), [GitHub federation](https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines), [billing budgets](https://docs.cloud.google.com/billing/docs/how-to/budgets), [eligible native spend caps](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps), [free-trial restrictions](https://docs.cloud.google.com/free/docs/free-cloud-features).

## One-time cloud setup

Review and run `scripts/setup_gcp_compute.sh` in Cloud Shell after approving its IAM grants. It creates identities, a private result bucket, a VPC and a GitHub OIDC trust; it creates no VM. Populate the five `GCP_*` GitHub Actions secrets printed by the setup script. They are identifiers, not a private key. Also set `GCP_CREDIT_EXPIRES_AT` to the verified promotional credit expiry as a UTC timestamp, e.g. `2026-12-01T00:00:00Z`; never assume this example is the actual expiry. Missing/expired credit dates block compute. Keep `GCP_GPU_ENABLED=false` until the activation conditions above are satisfied.

The initial budget ledger must be written by the administrator as `budget/reservations/initial-safety-baseline.json`, with `run_id` `initial-safety-baseline` and positive integer `cents` covering existing gross usage plus at least 1 cent. Ledger records never expire. Credit expiration should also disable execution; do not run beyond the verified promotional credit expiry.
