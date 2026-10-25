# ULM Document Extraction Benchmark

Compare **direct vision-language models (VLMs)** with **PaddleOCR → language model (OCR + LM)** pipelines for key information extraction (KIE) from document images. The benchmark uses the **400 English KIE examples from OCRBench v2**, scores predictions with the official evaluator, and records accuracy, latency, and computational-resource measurements.

The repository targets **bwForCluster Helix** (SLURM, NVIDIA A100/H200, Python 3.12, CUDA 13.0). It can serve as a starting point for other clusters, but its CUDA wheels and SLURM settings are Helix-specific. Select a GPU family explicitly when launching tests or benchmarks; see [GPU selection](#4-choose-a-gpu-a100-h200-or-h100-where-available).

## Experiments

Configure experiments in [`model.yml`](model.yml):

| Experiment ID | Pipeline | Model |
| --- | --- | --- |
| `qwen35_vlm` | Image → VLM | Qwen3.5-9B |
| `minicpm_vlm` | Image → VLM | MiniCPM-V-4.6 |
| `qwen35_ocr` | Image → PaddleOCR → LM | Qwen3.5-9B |
| `qwen25_ocr` | Image → PaddleOCR → LM | Qwen2.5-14B-Instruct-1M |

Both OCR + LM experiments reuse the same OCR extraction cache. Each model has its own GPU job, and a CPU collector computes the official scores and writes a comparison CSV. The original OCRBench questions are used without extra output-formatting instructions; the default generation limit is 384 tokens.

## Quick start: install on Helix

**Prerequisites:** a working Helix account with SLURM/GPU access, Git, `/usr/bin/python3.12`, and the OCRBench v2 dataset. **Run installation commands on a login node; run GPU tests and inference through `sbatch`.**

### 1. Clone the repository

For a new checkout:

```bash
cd ~
git clone https://github.com/econoberries/ULM-Project-Extractor-VLM-OCRLM.git
cd ULM-Project-Extractor-VLM-OCRLM
```

For an existing checkout, use `cd ~/ULM-Project-Extractor-VLM-OCRLM` followed by `git pull --ff-only origin main`. Do not recreate working virtual environments or delete `runs/` when updating the code.

### 2. Install two isolated Python environments

**Do not put PaddlePaddle GPU and PyTorch in the same environment.** On Helix, their CUDA 13 packages require incompatible cuDNN/NCCL versions.

| Environment | Purpose | Packages |
| --- | --- | --- |
| `.venv` | Direct VLMs, OCR + LM model workers, CLI and scoring | PyTorch 2.14.1+cu130, Torchvision, Transformers, evaluator |
| `.venv-ocr` | Shared OCR worker | PaddlePaddle GPU 3.3.0, PaddleOCR 3.7.0, PaddleX 3.7.2 |

Create/install **PyTorch/LM**:

```bash
test -x .venv/bin/python || /usr/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/helix-pytorch-cu130.txt
.venv/bin/python -m pip install -e '.[vlm]'
.venv/bin/python -m pip install -r requirements/ocrbench-v2-kie-eval.txt
```

Create/install **PaddleOCR** separately:

```bash
test -x .venv-ocr/bin/python || /usr/bin/python3.12 -m venv .venv-ocr
.venv-ocr/bin/python -m pip install -e '.[ocr]'
.venv-ocr/bin/python -m pip install -r requirements/helix-paddle-cu130.txt
```

Use these environment-specific `python -m pip` commands rather than the system-wide `pip` or an activated environment. First-time installations may download large CUDA/model packages.

### 3. Prepare OCRBench v2

Get the [OCRBench v2 dataset and evaluator](https://github.com/Yuliang-Liu/MultimodalOCR). The CLI expects:

- `EN_part/`: the English document images, including the KIE subsets;
- `OCRBench_v2.json`: annotations in the parent directory (or pass `--annotations PATH`);
- `eval_scripts/eval.py`: official evaluator, normally in `~/MultimodalOCR/OCRBench_v2/eval_scripts/`.

Clone the official evaluator repository if you do not already have it:

```bash
test -f "$HOME/MultimodalOCR/OCRBench_v2/eval_scripts/eval.py" || \
  git clone https://github.com/Yuliang-Liu/MultimodalOCR.git "$HOME/MultimodalOCR"
```

The benchmark repository does **not** download the OCRBench data for you. Point `EN_PART` at your actual image directory. For the existing Helix project, the images have been stored in the older data checkout:

```bash
export EN_PART="$HOME/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part"
test -d "$EN_PART" || echo "Set EN_PART to the actual OCRBench_v2/EN_part path"
```

The included `requirements/ocrbench-v2-kie-eval.txt` covers **English KIE scoring only**; it intentionally omits the `Polygon3` extension used for other OCRBench tasks. Keep the upstream evaluation code unchanged. For reproducibility, record the upstream evaluator commit (the project's reference commit is `4e363b3f499811b43d792ef0bb33932ef05eca27`).

### 4. Choose a GPU: A100, H200, or H100 where available

**Helix currently documents A100 (40/80 GB) and H200 (141 GB), not H100.** See the [Helix hardware inventory](https://wiki.bwhpc.de/e/Helix/Hardware) and [Slurm GPU request guide](https://wiki.bwhpc.de/e/Helix/Slurm). A cluster with an H100 GRES can use the same selection mechanism, but do **not** request H100 on Helix unless its live Slurm configuration lists it.

| GPU family | Slurm GRES for one GPU | Helix status |
| --- | --- | --- |
| A100 | `gpu:A100:1` | Listed (40 GB and 80 GB variants) |
| H200 | `gpu:H200:1` | Listed (141 GB) |
| H100 | `gpu:H100:1` | Not listed; only use where supported |

Check the live Slurm GPU types and idle availability **on the cluster**:

```bash
sinfo -N -o '%N %G' | grep -E 'A100|H100|H200'
sinfo_t_idle
```

Choose one family for the **entire new benchmark batch**:

```bash
export GPU_TYPE=A100  # Helix: A100 or H200; H100 only if your cluster lists it
```

The GPU smoke tests in step 5 use `--gres="gpu:${GPU_TYPE}:1"` to request that type and `--export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}"` to check the allocated GPU against it. `slurm/check_environment.sbatch` defaults to A100 if neither is provided. Its printed `nvidia-smi` output records the **actual GPU name and memory**.

When creating a new batch, the `benchmark` commands below take **both** `--gres "gpu:${GPU_TYPE}:1"` (Slurm's physical GPU request) and `--gpu-family "$GPU_TYPE"` (the worker's GPU-family verification and batch metadata). Helix's `gpu-single` partition supports the listed A100 and H200 requests.

**Existing batches:** GPU family is recorded in `batch.json`. The `resume` CLI uses that stored family, not a new `--gpu-family` flag. Keep this project's existing batch on A100 when resuming with `--ocr-only`; do not switch its `--gres` to H200/H100. For a different family, submit a **new batch** after testing the GPU and model compatibility.

**Benchmark interpretation:** the A100 request does not specify 40 GB vs 80 GB. Record the actual device and VRAM from every job (`nvidia-smi` and run metadata) and keep hardware consistent when comparing latencies or memory use.

### 5. Verify installation and GPU access

First check package versions and environment separation on the login node:

```bash
.venv/bin/python -m pip check
.venv-ocr/bin/python -m pip check
.venv/bin/python scripts/check_dependencies.py --stage all
.venv/bin/python -m unittest discover -s tests -v
```

**Important:** Helix login nodes may not expose `libcuda.so.1`. The dependency checker then prints `DEFERRED import paddle`: that is expected and is **not proof GPU inference works**. Choose a GPU family (`A100` by default, or `H200` on Helix) and verify both runtimes on an allocated GPU:

```bash
export GPU_TYPE=A100   # Change to H200 to request a Helix H200
sbatch --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch torch
sbatch --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch paddle
```

Each submission prints a Slurm job ID. Inspect jobs and logs (replace the example ID):

```bash
squeue -u "$USER"
sacct -j 12345678 --format=JobID,State,ExitCode,Elapsed
cat gpu-check-12345678.out
```

Look for `PyTorch CUDA: PASSED` and `PaddlePaddle CUDA: PASSED` in their respective logs. The smoke tests perform real GPU calculations.

Then test a single image with the same PaddleOCR constructor used by the benchmark:

```bash
IMAGE="$(find "$EN_PART" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)"
test -n "$IMAGE" && sbatch --gres="gpu:${GPU_TYPE}:1" \
  --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch ocr "$IMAGE"
```

The output log must end with `PaddleOCR single-image inference: PASSED`. The first run may download PaddleOCR model weights. For an optional one-document Qwen smoke test:

```bash
sbatch --mem=64G --time=00:45:00 \
  --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch model "$EN_PART" qwen35_vlm
```

## Run a new benchmark

Only start a **new** batch when you intend to run the selected experiments again. This creates new batch/run IDs and schedules GPU jobs.

Use the `GPU_TYPE` and `EN_PART` variables from the installation steps. Preview the 400-document input and Slurm plan **without scheduling jobs**:

```bash
.venv/bin/extractor benchmark \
  --input "$EN_PART" \
  --models model.yml \
  --partition gpu-single \
  --gres "gpu:${GPU_TYPE}:1" \
  --gpu-family "$GPU_TYPE" \
  --dry-run
```

After checking the plan and completing the smoke tests, submit the **new** batch:

```bash
.venv/bin/extractor benchmark \
  --input "$EN_PART" \
  --models model.yml \
  --partition gpu-single \
  --gres "gpu:${GPU_TYPE}:1" \
  --gpu-family "$GPU_TYPE"
```

By default, GPU jobs use four CPUs, 64 GiB system RAM, and two hours; the collector uses `cpu-single`. Without explicit GPU flags, the CLI defaults to A100; the commands above use your chosen GPU instead. Customize resources through the CLI options if needed (`.venv/bin/extractor benchmark --help`). The official evaluator defaults to `../MultimodalOCR/OCRBench_v2/eval_scripts/eval.py` relative to this repository; `--evaluator PATH` overrides it.

The schedule is: **shared PaddleOCR (if needed) → OCR + LM workers**, alongside direct VLM workers, followed by a CPU collector. Dependencies use `afterany` so downstream logs and failure summaries are still generated if a preceding job fails. An OCR + LM job may therefore fail when OCR did not create the required cache; inspect the OCR job first.

## Monitor runs and read results

```bash
squeue -u "$USER"
ls -lt runs/batches/*/logs/
cat runs/benchmark_summary.csv
```

The batch-specific `runs/batches/<batch-id>/benchmark_summary.csv` is authoritative; the top-level CSV is the latest collected summary.

```text
runs/
  <run-id>/
    run.json                  # status, configuration, errors
    metadata.json             # GPU, package versions, load time
    predictions.json          # raw outputs and per-document timings
    predictions_scored.json   # official scores (when available)
    metrics.json              # aggregated run metrics
    scoring.log               # evaluation traceback, if scoring fails
  batches/<batch-id>/
    batch.json                # original immutable batch definition
    documents.json
    jobs.json                 # initial Slurm jobs
    resumes/                  # subsequent resume job records
    logs/                     # batch and resume job logs
    ocr/paddleocr/            # shared OCR text and timing cache
    benchmark_summary.csv
  benchmark_summary.csv
```

Each summary row identifies one model run, its status, completed document count, official KIE F1 (including subset scores), latency, model-load time, output tokens, and GPU-memory measurements where recorded. **A failed run has no F1 score; it is not an F1 of zero.** OCR + LM latency incorporates OCR time as well as LM inference time, even when OCR text is reused from cache. Record the actual A100 memory configuration (40 GB vs 80 GB) when comparing computational efficiency.

## Resume an interrupted batch

**Do not call `benchmark` again merely to repair or finish an existing run.** `resume` preserves the original batch and run IDs, existing predictions and scored VLM outputs; it only schedules unfinished work.

**Before resuming**, check `squeue -u "$USER"` and the relevant Slurm job states with `sacct`. Never submit a second worker for a run that is still queued or running. This project's original batch was created for **A100**: keep that GPU family for resumes, even if you later benchmark H200 in a new batch.

For this project's existing batch (ID `5dd8a6af-de7f-4d4c-b496-44068933a072`), the OCR-only recovery command is:

```bash
BATCH="$PWD/runs/batches/5dd8a6af-de7f-4d4c-b496-44068933a072/batch.json"

# Preview: does not submit jobs or modify files.
.venv/bin/extractor resume --batch "$BATCH" --ocr-only --dry-run

# Submit ONLY if no recovery for this batch is already active.
.venv/bin/extractor resume --batch "$BATCH" --ocr-only
```

`--ocr-only` **never resubmits direct VLM jobs**. It can reuse completed OCR cache documents and saves the retry job IDs in `runs/batches/<batch-id>/resumes/`. The CPU collector reuses existing scored predictions.

If inference finished but scoring failed, inspect `runs/<run-id>/scoring.log` and repair the evaluator dependencies before submitting a **CPU-only** collector; there is no need to rerun GPU inference:

```bash
sbatch --partition=cpu-single --cpus-per-task=2 --mem=4G \
  --time=00:30:00 --job-name=ulm-rescore \
  --output=rescore-%j.out \
  slurm/collector.sbatch collect --batch "$BATCH"
```

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Home directory inaccessible or `squeue` fails with DNS configuration errors | Cluster/storage/Slurm issue; check Helix notices/support. Do **not** recreate your home directory. |
| `torch` reports package version `2.14.1` instead of `2.14.1+cu130` | Use the updated checker, which verifies `torch.__version__`. Check `torch.version.cuda` before reinstalling. |
| `torch` and PaddlePaddle conflict on cuDNN/NCCL | Keep separate `.venv` and `.venv-ocr`. If Paddle was accidentally installed in `.venv`, uninstall `paddlepaddle-gpu paddleocr paddlex` **there**, then install `requirements/helix-torch-repair-cu130.txt` and recheck `pip check`. |
| `libcuda.so.1` missing when importing Paddle on a login node | Expected on nodes without NVIDIA driver libraries; run the GPU smoke test via `sbatch`. If it also fails on an allocated A100, inspect the log. |
| `ModuleNotFoundError: torchvision` | Install `requirements/helix-pytorch-cu130.txt` and `-e '.[vlm]'` in `.venv`. |
| `No module named paddleocr` / missing Paddle backend | Install `-e '.[ocr]'` and `requirements/helix-paddle-cu130.txt` in `.venv-ocr`, not `.venv`. |
| OCR model download stalls | Inspect the OCR Slurm log and the model cache. `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` skips the host connectivity check, but **does not replace missing model weights**. |
| OCR + LM cannot find `ocr/paddleocr/index.json` | Inspect the shared PaddleOCR job; its failure commonly prevents the cache from being created. |
| Scoring fails after predictions were saved | Inspect `runs/<run-id>/scoring.log`. The English KIE evaluator requirements omit `Polygon3`; rerun the CPU collector rather than GPU inference. |

## References

- [OCRBench v2: official dataset and evaluator](https://github.com/Yuliang-Liu/MultimodalOCR)
- [OCRBench v2 paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)
- [PyTorch CUDA 13.0 wheels](https://download.pytorch.org/whl/cu130/)
- [PaddlePaddle GPU installation](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/linux-pip_en.md)
