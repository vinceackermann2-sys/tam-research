# RLT AWS EC2 lane

This lane replaces the failed Colab transport with an isolated AWS EC2 execution path on branch `exp/rlt-aws`. It does not write to `main` and does not depend on GitHub write permissions during GPU execution.

## Recommended first instance

Use a single-GPU `g6.xlarge` in `eu-north-1` (Stockholm). It provides one NVIDIA L4 with 24 GB GPU memory, which is ample for the bounded first RLT run. Use a current AWS Deep Learning OSS Nvidia Driver AMI with PyTorch on Ubuntu 24.04.

Suggested root volume: 100 GB gp3.

## Fresh AWS experiment namespace

Smoke:

- job id: `rlt-publicspec-smoke-aws-20260913-a`
- seed: `20260916`

Bounded FineWeb training, only after smoke PASS:

- job id: `rlt-tiny-64k-aws-20260913-a`
- seed: `20260916`
- profile: `tiny`
- token budget: 65,536
- sequence length: 64
- micro batch: 2
- gradient accumulation: 4
- training shard: 1,000,000 tokens
- validation shard: 100,000 tokens

The earlier Colab-scoped `...-d` / `...-c` jobs are tombstoned as unstarted on this branch so they cannot accidentally run.

## Run

After connecting to the EC2 instance, clone `exp/rlt-aws` and run `aws/rlt_ec2_run.sh`.

The bootstrap script:

1. verifies `nvidia-smi`;
2. clones only `exp/rlt-aws`;
3. installs the repo editable into the DLAMI Python environment;
4. runs `experiments/rlt/aws_run_once.py`;
5. writes create-once local claims/results under `/opt/tam-rlt-output`;
6. keeps dataset and checkpoints outside the repository under `/opt/tam-rlt-data` and `/opt/tam-rlt-runs`.

No GitHub token is required because the repository is public and the AWS runner does not write back during execution.

## Result files

On success:

- `/opt/tam-rlt-output/rlt-publicspec-smoke-aws-20260913-a.json`
- `/opt/tam-rlt-output/rlt-tiny-64k-aws-20260913-a.json`

The runner refuses to execute if a local claim or result with the same job id already exists.
