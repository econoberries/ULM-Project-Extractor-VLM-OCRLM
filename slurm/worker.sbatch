#!/usr/bin/env bash
# SLURM GPU worker. PaddleOCR and PyTorch use isolated Python environments:
#   ocr -> .venv-ocr (flat PaddleOCR)
#   ocr-layout -> .venv-la-ocr (coordinate-preserving PaddleOCR)
#   ppstructure -> .venv-ppstructure (document layout and Markdown)
#   run -> .venv (PyTorch VLM / OCR+LM / LA-OCR+LM / PP-Structure+LM)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Missing worker action: expected ocr, ocr-layout, ppstructure or run." >&2
  exit 2
fi

case "$1" in
  ocr) python_bin=".venv-ocr/bin/python" ;;
  ocr-layout) python_bin=".venv-la-ocr/bin/python" ;;
  ppstructure) python_bin=".venv-ppstructure/bin/python" ;;
  run) python_bin=".venv/bin/python" ;;
  *)
    echo "Unsupported GPU worker action '$1'; expected ocr, ocr-layout, ppstructure or run." >&2
    exit 2
    ;;
esac

if [[ ! -x "$python_bin" ]]; then
  echo "Missing $python_bin. Follow README's separate-environment installation." >&2
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
echo "SLURM worker action: $1; Python: $python_bin"
exec "$python_bin" -m ulm_extractor.worker "$@"
