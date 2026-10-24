#!/usr/bin/env bash
# Generic GPU worker; resource settings are supplied by scheduler.py
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python; install the Python environment in this repository." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "GPU job has no nvidia-smi" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ "$gpu_name" != *"$EXTRACTOR_EXPECT_GPU"* ]]; then
  echo "Wrong GPU: expected $EXTRACTOR_EXPECT_GPU, got $gpu_name" >&2
  exit 2
fi
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -m ulm_extractor.worker "$@"
