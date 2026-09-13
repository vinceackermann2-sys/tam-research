#!/usr/bin/env bash
set -euo pipefail

BRANCH="${RLT_BRANCH:-exp/rlt-aws}"
REPO_URL="https://github.com/vinceackermann2-sys/tam-research.git"
REPO_DIR="${RLT_REPO_DIR:-/opt/tam-rlt-aws}"
OUTPUT_ROOT="${RLT_OUTPUT_ROOT:-/opt/tam-rlt-output}"
DATA_ROOT="${RLT_DATA_ROOT:-/opt/tam-rlt-data}"
RUN_ROOT="${RLT_RUN_ROOT:-/opt/tam-rlt-runs}"
PYTHON_BIN="${RLT_PYTHON:-/opt/pytorch/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Launch an AWS GPU instance with an NVIDIA-capable DLAMI." >&2
  exit 2
fi

nvidia-smi

if [[ -e "$REPO_DIR" ]]; then
  echo "$REPO_DIR already exists; refusing to replace an existing experiment checkout." >&2
  exit 3
fi

sudo mkdir -p "$REPO_DIR" "$OUTPUT_ROOT" "$DATA_ROOT" "$RUN_ROOT"
sudo chown -R "$(id -u):$(id -g)" "$REPO_DIR" "$OUTPUT_ROOT" "$DATA_ROOT" "$RUN_ROOT"
rmdir "$REPO_DIR"

git clone --depth 1 --single-branch --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
"$PYTHON_BIN" -m pip install -q -e "$REPO_DIR"

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -u experiments/rlt/aws_run_once.py \
  --output-root "$OUTPUT_ROOT" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT"
