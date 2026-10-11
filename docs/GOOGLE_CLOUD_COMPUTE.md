# Google Cloud compute from chat

This connection follows the existing Modal pattern: an owner-created GitHub issue
launches compute and receives the outcome as a comment. Google Cloud Batch provisions
a temporary Compute Engine VM, runs code from the exact triggering main commit, and
removes the VM afterwards. The Modal workflows and research protocols are unchanged.

## One-time connection

Project: `project-c40afe76-9188-4739-b66`. Region: `us-central1`.

Review and run `bash scripts/setup_gcp_compute.sh` in the project owner's Cloud Shell.
It enables the required APIs, creates two dedicated service accounts, and configures
GitHub OIDC federation for this repository's numeric ID, owner's numeric ID, main
branch, and only `.github/workflows/google-cloud-compute.yml`. Pull requests and
other workflows cannot authenticate. No service-account private key is created.

The launcher can create/manage Batch jobs and read project logs. It can act as only
the dedicated runtime service account. The runtime can report Batch status and write
logs; it has no Editor/Owner permissions or access to Modal credentials.

Save the five `GCP_*` values printed by the script as GitHub Actions repository
**variables**, then merge the compute workflow to main. Run the default workflow
once; success requires `SUCCEEDED` cloud status and `TAM_GCP_COMPUTE_OK` in task logs.

## Request compute

From a chat with GitHub write access, create an issue owned by `vinceackermann2-sys`
whose title starts with `[gcp-run]` and whose entire body is JSON:

```json
{"script":"scripts/gcp_compute_smoke.py","args":[],"timeout_seconds":600}
```

Or dispatch **Google Cloud TAM compute** on main with the same JSON in `config`.
Ask the chat assistant to run or inspect this workflow/issue. A direct Google Cloud
ChatGPT connector is not required; the GitHub tools or CLI submit requests and read
the results, as with the Modal issue-triggered workflows.

To run another CPU Python script already present on main, set `script` and `args`.
Set `install_dependencies` to `true` if it needs CPU PyTorch, the TAM package,
pytest, datasets, or transformers. Installation time counts toward the task limit.
Code requiring `modal` remote functions must be adapted separately; Modal launchers
are not portable Compute Engine scripts.

## Boundaries

- One CPU `e2-standard-4` VM, one task, zero automatic retries; no GPU allocation.
- Task limit 60–1800 seconds (default 600); watcher limit 45 minutes including scheduling.
- Concurrent requests queue; failed/cancelled watchers stop their unfinished job.
- Compute VM and temporary files are ephemeral. Logs/configuration/status are saved
  as GitHub workflow artifacts for 14 days and the requesting issue receives logs.
- Local output files/checkpoints are not copied off the VM. Add explicitly scoped
  persistent storage before any workload that needs durable outputs or resumption.
- Existing Modal checkpoints and datasets remain on the Modal volume.
- Only infrastructure smoke verification is part of connection setup; no scientific
  seed, training run, or held-out evaluation is authorized by this setup.
- This account currently uses the Google Cloud free trial. GPUs require paid-account
  activation and appropriate quota, followed by a separately scoped GPU job configuration.

Cloud resources consume trial credits or incur billed costs if billing is upgraded.
The runtime limits restrict resource use; they are not a monetary billing cap.

References: [GitHub keyless authentication](https://github.com/google-github-actions/auth),
[Batch setup](https://docs.cloud.google.com/batch/docs/get-started),
[Batch VM lifecycle](https://docs.cloud.google.com/batch/docs/create-run-job),
[Free trial restrictions](https://docs.cloud.google.com/free/docs/free-cloud-features).
