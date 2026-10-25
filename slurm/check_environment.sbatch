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

# From repository root:
#   sbatch slurm/check_environment.sbatch torch    # .venv (PyTorch)
#   sbatch slurm/check_environment.sbatch paddle   # .venv-ocr (PaddlePaddle)
#   sbatch --gres=gpu:H200:1 --export=ALL,EXTRACTOR_EXPECT_GPU=H200 slurm/check_environment.sbatch paddle
# The #SBATCH GPU request defaults to A100; --gres on the command line overrides it.
#   sbatch slurm/check_environment.sbatch ocr /path/to/image.jpg
# Optional:
#   sbatch --mem=64G --time=00:45:00 slurm/check_environment.sbatch model /path/to/EN_part qwen35_vlm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/check_environment.sbatch {torch|paddle|ocr IMAGE|la-paddle|la-ocr IMAGE|model EN_PART [EXPERIMENT_ID]}" >&2
  exit 2
fi

case "$1" in
  paddle|ocr) python_bin=".venv-ocr/bin/python" ;;
  la-paddle|la-ocr) python_bin=".venv-la-ocr/bin/python" ;;
  torch|model) python_bin=".venv/bin/python" ;;
  *) echo "Unknown GPU test '$1'." >&2; exit 2 ;;
esac
if [[ ! -x "$python_bin" ]]; then
  echo "Missing $python_bin. Install the matching environment before testing." >&2
  exit 2
fi

echo "SLURM job: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "Repository: $(pwd)"
echo "Python environment: $python_bin ($("$python_bin" --version))"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found on the allocated GPU node." >&2
  exit 2
fi
nvidia-smi
expected_gpu="${EXTRACTOR_EXPECT_GPU:-A100}"
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiF "$expected_gpu"; then
  echo "Expected GPU containing '$expected_gpu', but the allocation reports:" >&2
  nvidia-smi --query-gpu=name --format=csv,noheader >&2
  exit 2
fi
echo "GPU family check: $expected_gpu"

case "$1" in
  torch)
    exec "$python_bin" scripts/check_torch_cuda.py
    ;;
  paddle|la-paddle)
    exec "$python_bin" scripts/check_paddle_cuda.py
    ;;
  ocr)
    if [[ $# -ne 2 ]]; then
      echo "Usage: sbatch slurm/check_environment.sbatch ocr /absolute/path/to/image.jpg" >&2
      exit 2
    fi
    exec "$python_bin" scripts/check_paddleocr.py --image "$2"
    ;;
  la-ocr)
    if [[ $# -ne 2 ]]; then
      echo "Usage: sbatch slurm/check_environment.sbatch la-ocr /absolute/path/to/image.jpg" >&2
      exit 2
    fi
    exec "$python_bin" scripts/check_layoutocr.py --image "$2"
    ;;
  model)
    if [[ $# -lt 2 || $# -gt 3 ]]; then
      echo "Usage: sbatch --mem=64G --time=00:45:00 slurm/check_environment.sbatch model /path/to/EN_part [EXPERIMENT_ID]" >&2
      exit 2
    fi
    if [[ $# -eq 3 ]]; then
      exec "$python_bin" scripts/check_model_inference.py --input "$2" --experiment-id "$3"
    fi
    exec "$python_bin" scripts/check_model_inference.py --input "$2"
    ;;
esac
