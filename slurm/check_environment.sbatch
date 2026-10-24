#!/usr/bin/env bash
#SBATCH --job-name=ulm-env-check
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=gpu-check-%j.out

# Submit from the repository root:
#   sbatch slurm/check_environment.sbatch torch
#   sbatch slurm/check_environment.sbatch paddle
#   sbatch slurm/check_environment.sbatch ocr /path/to/image.jpg
# Optional ONE-document model test:
#   sbatch --mem=64G --time=00:45:00 slurm/check_environment.sbatch model /path/to/EN_part qwen35_vlm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python missing. Create the Python 3.12 venv first." >&2
  exit 2
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/check_environment.sbatch {torch|paddle|ocr IMAGE|model EN_PART [EXPERIMENT_ID]}" >&2
  exit 2
fi

echo "SLURM job: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "Repository: $(pwd)"
echo "Virtualenv Python: $(.venv/bin/python --version)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found on the allocated GPU node." >&2
  exit 2
fi
nvidia-smi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A100; then
  echo "ERROR: Requested A100, but no A100 GPU was reported." >&2
  exit 2
fi

case "$1" in
  torch)
    exec .venv/bin/python scripts/check_torch_cuda.py
    ;;
  paddle)
    exec .venv/bin/python scripts/check_paddle_cuda.py
    ;;
  ocr)
    if [[ $# -ne 2 ]]; then
      echo "Usage: sbatch slurm/check_environment.sbatch ocr /absolute/path/to/image.jpg" >&2
      exit 2
    fi
    exec .venv/bin/python scripts/check_paddleocr.py --image "$2"
    ;;
  model)
    if [[ $# -lt 2 || $# -gt 3 ]]; then
      echo "Usage: sbatch --mem=64G --time=00:45:00 slurm/check_environment.sbatch model /path/to/EN_part [EXPERIMENT_ID]" >&2
      exit 2
    fi
    if [[ $# -eq 3 ]]; then
      exec .venv/bin/python scripts/check_model_inference.py --input "$2" --experiment-id "$3"
    fi
    exec .venv/bin/python scripts/check_model_inference.py --input "$2"
    ;;
  *)
    echo "ERROR: Unknown test '$1'. Choose torch, paddle, ocr or model." >&2
    exit 2
    ;;
esac
