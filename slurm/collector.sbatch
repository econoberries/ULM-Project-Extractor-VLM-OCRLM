#!/usr/bin/env bash
# CPU-only summary collector: dependency afterany, so it also runs after failures.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python" >&2
  exit 2
fi
exec .venv/bin/python -m ulm_extractor.worker "$@"
