#!/usr/bin/env bash
# Submit the original FOUR English KIE architectures N times on Helix.
# Usage:
#   EN_PART=/absolute/path/to/OCRBench_v2/EN_part bash scripts/run_repeated_baselines.sh 3 --dry-run
#   EN_PART=/absolute/path/to/OCRBench_v2/EN_part bash scripts/run_repeated_baselines.sh 3
#
# Every repetition creates a separate immutable 400-document batch:
#   runs/repeated_baselines/<series-id>/repeat-001/{batches, <run-id>, ...}
# This deliberately excludes LA-OCR JSON and PP-StructureV3.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_repeated_baselines.sh REPEATS [--dry-run]

  REPEATS           Number of FULL independent benchmark submissions (1..20).
  --dry-run         Preview all repetitions; submit no jobs, write no files.

Environment variables (optional):
  EN_PART            OCRBench_v2/EN_part path
  GPU_TYPE           A100 (default), H200, or available cluster GPU family
  GPU_PARTITION      gpu-single (default)
  COLLECT_PARTITION  cpu-single (default)
  GPU_MEM            64G (default)
  GPU_CPUS           4 (default)
  GPU_TIME           02:00:00 (default)
  EVALUATOR_PATH     Upstream OCRBench eval.py (otherwise CLI default)
  REPEAT_OUTPUT_BASE Repository runs/repeated_baselines (default)

Each repetition submits: 2 direct VLM jobs + 1 shared flat-PaddleOCR
GPU job + 2 OCR->LM GPU jobs + 1 CPU collector job. The GPU jobs are
queued through Slurm; this script does not wait for their completion.
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi
repeats="$1"
shift
dry_run=false
if [[ $# -eq 1 ]]; then
  if [[ "$1" != "--dry-run" ]]; then
    usage >&2
    exit 2
  fi
  dry_run=true
fi
if [[ ! "$repeats" =~ ^[1-9][0-9]*$ ]] || (( 10#$repeats > 20 )); then
  echo "ERROR: REPEATS must be an integer from 1 through 20." >&2
  exit 2
fi
repeats=$((10#$repeats))

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python="$root/.venv/bin/python"
extractor="$root/.venv/bin/extractor"
models="$root/model-baselines.yml"
for required in "$python" "$extractor" "$models"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: Missing $required (follow the README installation)." >&2
    exit 2
  fi
done
if [[ ! -x "$extractor" ]]; then
  echo "ERROR: Not executable: $extractor" >&2
  exit 2
fi

input="${EN_PART:-$HOME/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part}"
if [[ ! -d "$input" ]]; then
  echo "ERROR: Missing EN_part directory: $input" >&2
  exit 2
fi
input="$("$python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$input")"
gpu_type="${GPU_TYPE:-A100}"
gpu_partition="${GPU_PARTITION:-gpu-single}"
collector_partition="${COLLECT_PARTITION:-cpu-single}"
gpu_mem="${GPU_MEM:-64G}"
gpu_cpus="${GPU_CPUS:-4}"
gpu_time="${GPU_TIME:-02:00:00}"

# New, separate result series every time; no historical batch is overwritten.
random_suffix="$("$python" -c 'import uuid; print(uuid.uuid4().hex[:8])')"
series_id="baselines-$(date -u +%Y%m%dT%H%M%SZ)-$random_suffix"
base="${REPEAT_OUTPUT_BASE:-$root/runs/repeated_baselines}"
base="$("$python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$base")"
series_dir="$base/$series_id"
commit="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"

printf 'Series: %s\n' "$series_id"
printf 'Input: %s\n' "$input"
printf 'GPU request: %s / %s\n' "$gpu_partition" "$gpu_type"
printf 'Repetitions: %s, models per repetition: 4, documents per model: 400\n' "$repeats"
printf 'Output series: %s\n' "$series_dir"

if [[ "$dry_run" == false ]]; then
  mkdir -p "$series_dir"
  manifest="$series_dir/series_manifest.tsv"
  printf 'repeat_index\tbatch_id\toutput_root\tcollector_job_id\tgit_commit\n' > "$manifest"
fi

for (( i=1; i<=repeats; i++ )); do
  label="$(printf '%03d' "$i")"
  output_root="$series_dir/repeat-$label"
  printf '\n===== Repetition %d/%d =====\n' "$i" "$repeats"

  args=(
    benchmark
    --input "$input"
    --models "$models"
    --output-root "$output_root"
    --partition "$gpu_partition"
    --collector-partition "$collector_partition"
    --gres "gpu:${gpu_type}:1"
    --gpu-family "$gpu_type"
    --mem "$gpu_mem"
    --cpus "$gpu_cpus"
    --time "$gpu_time"
  )
  if [[ -n "${EVALUATOR_PATH:-}" ]]; then
    args+=(--evaluator "$EVALUATOR_PATH")
  fi

  if [[ "$dry_run" == true ]]; then
    "$extractor" "${args[@]}" --dry-run
    continue
  fi

  mkdir -p "$output_root"
  # pipefail ensures a failed scheduler submission stops the loop.
  "$extractor" "${args[@]}" 2>&1 | tee "$output_root/submission.log"

  # This repeat's output root is unique, so exactly one batch.json must exist.
  shopt -s nullglob
  matches=("$output_root"/batches/*/batch.json)
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "ERROR: Expected one batch.json in $output_root; found ${#matches[@]}." >&2
    exit 1
  fi
  batch_path="${matches[0]}"
  batch_id="$(basename "$(dirname "$batch_path")")"
  jobs_path="$(dirname "$batch_path")/jobs.json"
  if [[ ! -f "$jobs_path" ]]; then
    echo "ERROR: Collector job manifest missing: $jobs_path" >&2
    exit 1
  fi
  collector_id="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["collector"])' "$jobs_path")"
  printf '%d\t%s\t%s\t%s\t%s\n' "$i" "$batch_id" "$output_root" "$collector_id" "$commit" >> "$manifest"
  printf 'Submitted repetition %d: batch %s, collector SLURM job %s\n' "$i" "$batch_id" "$collector_id"
done

if [[ "$dry_run" == true ]]; then
  echo "DRY RUN ONLY. No Slurm jobs submitted or output files created."
else
  printf '\nAll %d repetitions queued. Manifest: %s\n' "$repeats" "$manifest"
  printf 'The batch-specific benchmark_summary.csv files appear after their collectors finish.\n'
  printf 'Aggregate afterward with:\n  %s scripts/aggregate_repeated_baselines.py %s\n' "$python" "$series_dir"
fi
