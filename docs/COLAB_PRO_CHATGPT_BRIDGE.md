# Colab Pro compute controlled from ChatGPT web chat

The repository's earlier docs/COLAB_COMPUTE.md connects Google's Colab MCP
to a **local** coding agent. It does not give this ChatGPT web session
direct access to Windows localhost. This bridge instead uses GitHub issues
as a permissioned queue. I can create an issue and, with your explicit
approval, add the job approval label. Your **local** worker then uses the
official Google Colab CLI to run a pinned script on your personal Colab Pro
account and returns a sanitized issue comment.

Google's official Colab CLI: https://github.com/googlecolab/google-colab-cli
As of October 2026 it officially supports Linux/macOS only, **not Windows**.
Use WSL2 Ubuntu on your Windows PC (or a trusted Linux/macOS host). This
bridge is not connected until you complete OAuth and run the worker.

## One-time setup (Windows)

1. Open PowerShell **as administrator** and install Ubuntu on WSL2:

       wsl --install -d Ubuntu

   Restart if requested. Launch Ubuntu and choose a Linux username.

2. Inside the Ubuntu terminal, install the dependencies:

       sudo apt-get update
       sudo apt-get install -y git gh curl python3
       curl -LsSf https://astral.sh/uv/install.sh | sh
       source "$HOME/.local/bin/env"
       uv python install 3.12
       uv tool install --python 3.12 google-colab-cli

   Verify that both colab and gh work:

       colab version
       gh --version

3. Log into your GitHub and **your own Colab Pro Google account**:

       gh auth login
       colab --auth=oauth2 sessions

   Follow the printed Google authorization URL and paste the browser's
   code into WSL. Never send OAuth tokens to ChatGPT or GitHub issues.
   The credentials remain on this computer.

4. After reviewing and merging the PR adding this bridge, clone the repo
   and start the worker in Ubuntu:

       git clone https://github.com/vinceackermann2-sys/tam-research.git
       cd tam-research
       python3 -m py_compile scripts/colab_pro_issue_worker.py scripts/colab_bridge_gpu_probe.py
       python3 scripts/colab_pro_issue_worker.py --poll --max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4

   Leave the WSL terminal and computer running. Stop with Ctrl+C.
   The worker polls GitHub every 45 seconds. It requires NO personal
   GitHub token in the repository, Google Cloud billing, or Colab Enterprise.

## How to issue a job from this ChatGPT conversation

Ask ChatGPT to create a pending Colab Pro GPU probe issue, using the
current **merged** main commit. It will have this exact body format:

    <!-- colab-pro-job:v1 -->
    ```json
    {"commit":"REPLACE_WITH_ACTUAL_40_CHARACTER_MERGED_COMMIT_SHA","script":"scripts/colab_bridge_gpu_probe.py","gpu":"T4"}
    ```

The first line must be the marker, followed by a JSON code fence.
The job will NOT execute until the label **colab-approved** is applied.
When you authorize the GPU test, ChatGPT's GitHub connection can
add the label. The worker then runs one task on your Colab Pro quota,
writes a nonsecret public summary back to the issue. Legacy probe logs and
durable no-retry receipts are stored on your PC at:

       ~/.local/state/colab-pro-chatgpt/

The worker accepts only jobs authored by the repository owner, only
commits already merged into origin/main, and only the self-contained
probe or scripts within colab_jobs/*.py. Jobs cannot choose arbitrary
local commands. No script is run without the approval label. An
atomic local receipt is reserved before GPU use; retrying the same
issue is intentionally disabled.

## Costs, security and scientific rules

- **Default worker limits:** one attempted issue per worker invocation, CPU/T4
  only, with an outer wall-clock execution timeout of 180 seconds including
  provisioning. Hard-coded maximum accepted by the CLI wrapper is 300 seconds.
  The Linux/macOS/WSL worker holds an exclusive lock to prevent simultaneous
  workers sharing the same state directory. No automatic retry.
- **Compute-unit cap caveat:** the three-minute watchdog is **not** a hard
  Google billing/compute-unit limit. A timeout or machine shutdown might
  fail to release a session immediately. Review `colab --auth=oauth2 usage`
  and `colab --auth=oauth2 sessions` before and after each job. Colab Pro
  itself controls quota and accelerator availability. Do not assume a
  particular cost in units based only on requested hardware.
- This PR does not itself start a GPU or spend any compute units.
- A T4 probe consumes some Colab Pro quota, subject to availability.
- Colab Pro may impose compute-unit, session, and accelerator limits.
- GitHub issues in tam-research are public. Do not include secrets,
  private data, or raw private logs in the job or public metrics.
- Scripts opt into small, filtered public telemetry with a line
  beginning COLAB_PUBLIC_RESULT= followed by a JSON object.
- The bridge uses the official CLI's ephemeral colab run command and
  additionally attempts colab stop in a finally block. If an attempt
  fails or times out, use colab --auth=oauth2 sessions to check and
  clean up residual runtimes; do not rerun a consumed attempt.
  Research scripts use `run --keep` to allow verified cloud retrieval before
  the worker's `finally` stops the runtime. `--keep` never grants ongoing use.
- For research, retain pinned baselines, preregistration, unique
  job ID/seed, review and CI gates, and explicit GPU authorization.
  No automatic experimental retry, replication or scaling is granted.
- Colab CLI runs the submitted standalone Python script. Existing
  Modal-based scripts do not automatically work on Colab.
  New research jobs should be added as reviewed, standalone
  colab_jobs/*.py scripts and merged to main before queueing.
- First-time GitHub issue labeling may require creating the
  colab-approved label. GitHub labels are repository-admin-controlled.
- This is an **opt-in local worker bridge**, not an installed Colab
  plugin in this chat. If your Windows machine or WSL worker is off,
  the queued job waits.

## Research jobs after the consumed #1443 probe

Issue #1443 tested a T4 and teardown successfully and is permanently consumed.
The worker excludes it even if started with a new receipt directory. Its last
reported 108.89 units is historical; the worker queries current usage before
any allocation and after every attempt. Unknown balance, zero balance or an
existing active assignment blocks allocation. Google credentials stay on WSL.

The five-track continuation record and Modal transfer inventory are in
[COLAB_RESEARCH_CONTINUATION.md](COLAB_RESEARCH_CONTINUATION.md). The first
prepared job is [the existing CPU source/artifact preflight](../research/colab/chm_v4_readiness_job.md),
not new training or a GPU probe.

### Private cloud artifacts

Colab's disk is temporary. Research jobs therefore require a private cloud
folder configured locally through `rclone`; Google Drive is the recommended
destination. No credentials belong in issues, scripts, or this chat. Review the
storage destination and locally authorize it before any approved job starts:

```sh
sudo apt-get install -y rclone
rclone config
# Create a Google Drive remote named tam-drive using local browser OAuth.
# Choose your own account and permissions; do not publish the configuration.
rclone mkdir 'tam-drive:TAM Research Artifacts'
rclone lsf 'tam-drive:TAM Research Artifacts' --max-depth 1
```

These are storage setup commands, not Colab allocation. The named folder is a
proposal until the owner chooses and verifies it. Existing Colab OAuth does not
prove that a Drive upload connection is available. A different approved cloud
remote may be supplied; do not configure a local-filesystem remote.

After this PR is reviewed and merged, leave this worker running in one WSL
terminal, preserving its existing state directory and limits:

```sh
python3 scripts/colab_pro_issue_worker.py --poll \
  --max-seconds 180 --max-attempts 1 --allowed-gpus CPU,T4 \
  --artifact-remote 'tam-drive:TAM Research Artifacts'
```

In a second WSL terminal, inspect real worker liveness and **fresh** account
usage without allocating compute:

```sh
python3 scripts/colab_pro_issue_worker.py --status
colab --auth=oauth2 sessions
```

`--status` probes the held exclusive lock to report a running worker and shows
its last recorded settings. It also queries current usage. A worker that consumed
its one attempt exits and releases that lock. A status command on a different machine
does not verify your Windows worker. Keep this output private; public job
comments contain only balance-check/teardown booleans and opted-in telemetry.

### Bootstrap, export, and failure behavior

Every new `colab_jobs/*.py` script receives the exact reviewed full commit SHA,
issue identity, script path/hash and requested accelerator as arguments. The
readiness script fetches that exact commit, verifies the submitted script bytes
and three frozen source blobs, and uses the standard library only. The separate,
unapproved microfit adapter installs CPU Torch2.10.0 with normal TLS verification;
its 256-step/18,560-token scientific configuration remains unchanged from #1444.
CLI source alone carries neither repository files nor Modal datasets/checkpoints.

Research scripts create `/content/tam-results/bundle.zip` containing
`configuration.json`, `metrics.json`, `provenance.json`, `run.log`, any required
terminal checkpoint, and a SHA256 member
manifest. The worker downloads and validates the bundle, uploads it to the
cloud, downloads it again and verifies every hash **before** stopping the
runtime. The full transport log is independently uploaded/read back first;
an incomplete bundle also triggers retrieval of any current log/configuration/
provenance/checkpoint files. Failed exports cannot produce SUCCESS. The runtime
packages ordinary failures as evidence and raises afterward; partial recovery is
explicitly marked incomplete. No result,
checkpoint or downloaded bundle is retained on the worker: temporary transfer
files are removed. Mandatory attempt receipts and authentication metadata remain
local. The official CLI also maintains local connection diagnostics/history;
never upload those files because they can contain runtime connection secrets.

The unchanged 180-second allocation/execution/export deadline reserves 45
seconds for retrieval (135 seconds for provisioning/execution). Cleanup is
attempted separately with the existing stop/session-check timeouts, then usage
is queried. This is **not a hard compute-unit cap**. Unknown teardown or unknown
post-attempt usage blocks the next worker using its existing consumed receipt.
Resolve the actual runtime state and confirm zero active assignments/current
usage before recording verified cleanup in that receipt; never delete issue
receipts or global job tags to enable retries.

Small bundles keep #1444's 32 MiB compressed and uncompressed limit. Larger
checkpoint workflows need a separately reviewed exporter/storage plan, not a
quietly increased limit. Storage/network failure can prevent complete recovery;
the worker reports FAILED, retains any verified cloud evidence and still attempts
teardown rather than keeping paid compute alive indefinitely. No job retry,
replication, scaling, precision change or Modal fallback is automatic.

CLI lifecycle contracts were checked against official `googlecolab/google-colab-cli`
commit `d4f855d885cf39f47af5bcae9cd7f7963c18ef09`. The worker checks required
`--keep`/`--timeout` support locally before allocation. CPU mocks validate these
paths; they do not establish live Google OAuth, cloud write access or quota.

## Zero-GPU validation

       python3 -m py_compile scripts/colab_pro_issue_worker.py scripts/colab_bridge_gpu_probe.py
       python3 scripts/colab_pro_issue_worker.py --help

An actual end-to-end GPU test requires your Google OAuth sign-in and
explicit approval. No token should be supplied through this chat.

## Research job durability and continuation

See [COLAB_RESEARCH_ORCHESTRATION.md](COLAB_RESEARCH_ORCHESTRATION.md) for the
five separate project states and pending job scopes. New `colab_jobs` scripts
must declare a reviewed literal `COLAB_JOB_ID`. The worker reserves an atomic,
never-updated GitHub tag `colab-attempts/<job-id>` before provisioning, in addition
to the local issue receipt and remote issue callback. Existing identity means
no launch even if a new issue or machine tries to repeat it. GitHub authentication
needs Contents write for this receipt, as well as issue comment access.

Research runs use CLI `run --keep --timeout ...`: allocation, execution and
collection share the worker deadline. The CLI default code timeout is 30s, so the
wrapper now explicitly passes the remaining deadline. The reviewed standalone
script receives the exact issue commit, issue number, script path and requested
GPU; it must bootstrap the exact public repository SHA itself. No credential is
sent to Colab. Current CPU scripts reject other GPUs before repository setup.

The worker downloads `/content/tam-results/bundle.zip`, verifies its declared
SHA-256/size, every manifest member and issue/commit/script provenance, then
uploads and independently reads back the cloud copy **before** calling stop.
Local transfer files are temporary; it never extracts untrusted archives.
Bundles are bounded to 32 MiB. Research success additionally requires confirmed
teardown and fresh post-attempt usage. Missing artifacts or teardown are failures,
not architecture evidence. Cleanup is attempted even on timeout; if retrieval
cannot complete, record artifact loss and release the runtime to bound exposure.

Run `python3 scripts/colab_pro_issue_worker.py --status` on the local WSL machine
for a read-only lock-based worker check and fresh account balance. A stale
heartbeat alone is not proof that the worker is running. No automatic worker
restart or attempt retry is introduced. Default limits and maximum 300s unchanged.
