# ULM Document Extraction Benchmark

Compare **direct VLMs**, **flat-text PaddleOCR → LM**, **coordinate-JSON PaddleOCR → LM**, and **PP-StructureV3 Markdown → LM** for key information extraction (KIE) from document images. The benchmark uses the **400 English KIE examples from OCRBench v2**, scores predictions with the official evaluator, and records accuracy, latency, and computational-resource measurements.

The repository targets **bwForCluster Helix** (SLURM, NVIDIA A100/H200, Python 3.12, CUDA 13.0). It can serve as a starting point for other clusters, but its CUDA wheels and SLURM settings are Helix-specific. Select a GPU family explicitly when launching tests or benchmarks; see [GPU selection](#4-choose-a-gpu-a100-h200-or-h100-where-available).

## Experiments

Configure experiments in [`model.yml`](model.yml):

| Experiment ID | Pipeline | Model |
| --- | --- | --- |
| `qwen35_vlm` | Image → VLM | Qwen3.5-9B |
| `minicpm_vlm` | Image → VLM | MiniCPM-V-4.6 |
| `qwen35_ocr` | Image → PaddleOCR → LM | Qwen3.5-9B |
| `qwen25_ocr` | Image → flat PaddleOCR → LM | Qwen2.5-14B-Instruct-1M |
| `qwen35_laocr` | Image → coordinate JSON PaddleOCR → LM | Qwen3.5-9B |
| `qwen35_ppstructure` | Image → PP-StructureV3 Markdown → LM | Qwen3.5-9B |

The two **flat OCR + LM** runs share `paddleocr` OCR cache. **Coordinate JSON** uses `paddleocr_layout`, and **PP-StructureV3 Markdown** uses `ppstructurev3`. Each extractor has an isolated Paddle environment; **all language models still run in `.venv`**. All six experiments use the same 400 OCRBench v2 English KIE documents, the unmodified questions, the same official scorer, and the default 384-token generation limit.

**What “layout-aware” means here:** we use the **same pretrained PP-OCRv6 text detector/recognizer** as flat OCR. Instead of discarding coordinates, we serialize every `rec_texts` entry together with its `rec_boxes` [x_min, y_min, x_max, y_max] and `rec_scores` into **JSON text**. We include the source image dimensions and send that JSON string followed by the unchanged question to Qwen3.5. This is **coordinate-aware OCR**, not PP-StructureV3 table recognition or a newly trained model.

Example LM input (illustrative):

```json
{"format":"paddleocr-layout-v1","coordinate_system":"absolute_pixels","box_format":"xyxy","image_size":{"width":1000,"height":800},"lines":[{"text":"Protein","bbox":[50,200,140,222],"confidence":0.99},{"text":"8g","bbox":[650,200,700,222],"confidence":0.98}]}
```

The benchmark question follows the JSON as plain text. This lets Qwen infer proximity, but does **not** reconstruct tables.

**The fourth pipeline uses PP-StructureV3 plus OCR completeness recovery, under the *same* `qwen35_ppstructure` model ID.** PP provides layout, OCR, reading order, tables, and initial Markdown. Inside the existing `.venv-ppstructure` parser job, we also run the **same English PaddleOCR constructor used in the flat OCR baseline** on the original image. PP's internal OCR and OCR on PP-generated image crops are additional sources. The code then:

- Preserves PP's existing headings and HTML/Markdown tables.
- Replaces local image references with text recognized in the corresponding layout region or by OCR over the actual Markdown image crop.
- Recovers text missing *entirely* from PP Markdown, even when there is no image placeholder. If spatial placement is ambiguous, the text goes into an **"Additional OCR-recognized text (reading order)"** section rather than being silently discarded or placed in an invented table cell.
- Saves PP's original Markdown, raw parser JSON, original-image OCR transcript, and a per-document recovery audit alongside the **resolved `.md`**. Only the resolved Markdown and unchanged OCRBench question are passed as text to Qwen3.5; the raw JSON and image data are never passed as visual input.

This guarantees that lines output by the two recognizers are retained in the generated text (subject to conservative duplicate detection), **not** that OCR recognizes every character correctly or that ambiguous table cells can always be positioned accurately. The extra recognizer may raise OCR latency and GPU memory requirements. This is a repaired *implementation of the fourth pipeline*, not a new pipeline or model configuration.

```text
image → PP-StructureV3 + English PaddleOCR (.venv-ppstructure)
      → reconcile recognized text with PP regions and Markdown
      → markdown/<document-key>.md
      → Qwen3.5-9B (.venv) → predictions → official KIE F1
```

PP-StructureV3 can employ its **own OCR recognition models**, and the repaired mode also runs flat PaddleOCR, so it is a comparison of complete architectures, not a pure layout-only ablation. The original **59.01%** PP Markdown-only score remains a historical baseline; **do not attribute that score to the recovered-Markdown implementation until it has been rerun**. The CPU collector records parsing+OCR and LM times; other baselines and their cache formats are unchanged.

**Cache safety:** the repaired PP Markdown has a new format signature (`ppstructurev3-markdown-recovered-v2`). The old Markdown-only `ppstructurev3` cache cannot be silently reused. **Submit a new batch with the same `model-ppstructure.yml`** to evaluate the correction. Do not edit or delete the historical `runs/` results; the revised run will have new batch/run UUIDs.


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

### 2. Install four isolated Python environments

**Never install PaddlePaddle GPU into the PyTorch `.venv`.** On Helix, the frameworks require incompatible CUDA 13 cuDNN/NCCL dependency versions.

| Environment | Purpose | Packages |
| --- | --- | --- |
| `.venv` | **All** VLM and OCR→LM language-model jobs, CLI and scoring | PyTorch 2.14.1+cu130, Torchvision, Transformers |
| `.venv-ocr` | Existing shared **flat-text** OCR worker | PaddlePaddle GPU 3.3.0, PaddleOCR 3.7.0, PaddleX 3.7.2 |
| `.venv-la-ocr` | Existing **coordinate-JSON** OCR worker | Same pretrained PaddlePaddle/PaddleOCR versions, installed independently |
| `.venv-ppstructure` | **PP-StructureV3 + internal English OCR recovery**, produces complete Markdown | PaddlePaddle GPU 3.3.0, PaddleOCR 3.7.0 with `doc-parser` extra, PaddleX 3.7.2 |

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

Create/install **layout-aware OCR** into a third environment (does not modify existing environments):

```bash
test -x .venv-la-ocr/bin/python || /usr/bin/python3.12 -m venv .venv-la-ocr
.venv-la-ocr/bin/python -m pip install -e '.[ocr]'
.venv-la-ocr/bin/python -m pip install -r requirements/helix-paddle-cu130.txt
```

Create/install **PP-StructureV3** into a *fourth*, independent Paddle environment. No training or new model weights are needed beyond the pretrained downloads:

```bash
test -x .venv-ppstructure/bin/python || /usr/bin/python3.12 -m venv .venv-ppstructure
.venv-ppstructure/bin/python -m pip install -e '.[ppstructure]'
.venv-ppstructure/bin/python -m pip install -r requirements/helix-paddle-cu130.txt
```

The `[ppstructure]` extra is defined in `pyproject.toml`: it installs `paddleocr[doc-parser]==3.7.0` and `paddlex==3.7.2`. Installing the cu130 requirements **last** ensures the GPU PaddlePaddle wheel is used. Use each environment's own `python -m pip`, not the system `pip`, and never mix Paddle GPU with `.venv`. Initial parser weights and CUDA package downloads can be substantial. Each extraction engine runs in its own SLURM GPU job.

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
.venv-la-ocr/bin/python -m pip check
.venv-ppstructure/bin/python -m pip check
.venv/bin/python scripts/check_dependencies.py --stage all
.venv/bin/python -m unittest discover -s tests -v
```

**Important:** Helix login nodes may not expose `libcuda.so.1`. The dependency checker then prints `DEFERRED import paddle`: that is expected and is **not proof GPU inference works**. Choose a GPU family (`A100` by default, or `H200` on Helix) and verify both runtimes on an allocated GPU:

```bash
echo "Testing GPU family: $GPU_TYPE"  # Selected in step 4
sbatch --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch torch
sbatch --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch paddle
sbatch --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch la-paddle
sbatch --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch pp-paddle
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

Also verify a **layout-aware JSON extraction** using the third environment:

```bash
test -n "$IMAGE" && sbatch --gres="gpu:${GPU_TYPE}:1" \
  --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch la-ocr "$IMAGE"
```

Run a **real PP-StructureV3 one-image Markdown smoke test** on an allocated GPU:

```bash
test -n "$IMAGE" && sbatch --mem=64G --time=00:45:00 \
  --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch ppstructure "$IMAGE"
```

Check `gpu-check-<JOBID>.out` for `PP-StructureV3 Markdown single-image inference: PASSED`, a **resolved** Markdown preview, and OCR recovery statistics. This tests both PP-StructureV3 and the full-image English OCR fallback on an allocated GPU. The flat OCR and coordinate-JSON logs should respectively end with `PaddleOCR single-image inference: PASSED` and `PaddleOCR layout single-image inference: PASSED`. For an optional one-document Qwen smoke test:

```bash
sbatch --mem=64G --time=00:45:00 \
  --gres="gpu:${GPU_TYPE}:1" --export="ALL,EXTRACTOR_EXPECT_GPU=${GPU_TYPE}" \
  slurm/check_environment.sbatch model "$EN_PART" qwen35_vlm
```

## Run a new benchmark

**Recommended: run just the new full 400-document PP-StructureV3 → Qwen3.5 experiment.** This does not resubmit the previously completed VLM, flat OCR or coordinate-JSON experiments. Use [`model-ppstructure.yml`](model-ppstructure.yml). `model-laocr.yml` is still available to run only the earlier coordinate-JSON experiment. The full [`model.yml`](model.yml) includes **all six** configurations and should be used only for a fresh matched-hardware comparison.

Use the `EN_PART` and `GPU_TYPE` variables from the installation steps. Preview the *full* 400-document schedule (does not submit anything):

```bash
.venv/bin/extractor benchmark \
  --input "$EN_PART" \
  --models model-ppstructure.yml \
  --partition gpu-single \
  --gres "gpu:${GPU_TYPE}:1" \
  --gpu-family "$GPU_TYPE" \
  --time 04:00:00 \
  --dry-run
```

After checking the preview and confirming the PP-StructureV3 GPU smoke test, submit the *full* run by omitting `--dry-run`:

```bash
.venv/bin/extractor benchmark \
  --input "$EN_PART" \
  --models model-ppstructure.yml \
  --partition gpu-single \
  --gres "gpu:${GPU_TYPE}:1" \
  --gpu-family "$GPU_TYPE" \
  --time 04:00:00
```

`--time 04:00:00` provides extra scheduling headroom for parser-model initialization and full-document parsing, rather than restricting the experiment to a subset. Adjust it if cluster wall-time limits require; the CLI default is 2 hours. GPU workers use 4 CPUs and 64 GiB RAM by default; the final collector runs on `cpu-single`. Override with the CLI resource flags if necessary.

The new schedule is **PP-StructureV3 parser job (`.venv-ppstructure`) → Qwen3.5 worker (`.venv`) → CPU scoring**. When running `model.yml`, flat and coordinate OCR engines also get their own jobs, each with an independent cache. Dependencies use `afterany` so logs and failure summaries are produced even if parsing fails: always inspect the parser job before retrying inference.

The scorer defaults to `../MultimodalOCR/OCRBench_v2/eval_scripts/eval.py` relative to this repository; pass `--evaluator PATH` if your upstream checkout differs.

## Repeat the original four baseline models (no LA OCR or PP-Structure)

Use [`model-baselines.yml`](model-baselines.yml) and
[`scripts/run_repeated_baselines.sh`](scripts/run_repeated_baselines.sh).
This selection is **only** the four original experiments:
`qwen35_vlm`, `minicpm_vlm`, `qwen35_ocr`, and `qwen25_ocr`.
It retains the same pinned checkpoints, unmodified OCRBench questions,
`max_new_tokens: 384`, `do_sample: false`, and the full 400-document dataset.

The **single positional parameter** is the number of complete repetitions.
**Recommended starting point: three**. For example, on the Helix login node:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM
git pull --ff-only origin main
export EN_PART="$HOME/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part"
export GPU_TYPE=A100

# Validate the complete three-repeat Slurm plan without submitting anything:
bash scripts/run_repeated_baselines.sh 3 --dry-run

# Submit three full independent batches, each containing four models:
bash scripts/run_repeated_baselines.sh 3
```

To repeat five times instead, change **only** the positional argument:
`bash scripts/run_repeated_baselines.sh 5`. The script accepts 1–20
repetitions to protect against accidental very large submissions. Optional
environment variables: `GPU_PARTITION`, `COLLECT_PARTITION`, `GPU_MEM`,
`GPU_CPUS`, `GPU_TIME`, `EVALUATOR_PATH`, and `REPEAT_OUTPUT_BASE`.
For example, `GPU_TIME=04:00:00` raises the GPU-job time limit. Run this
script from the login node, not inside an `sbatch` job. It **queues** jobs
and does **not** wait for their completion; check `squeue -u "$USER"` and
the relevant Slurm logs.

Each complete repetition schedules **five GPU jobs**
(two VLM jobs, one shared flat PaddleOCR extraction, and two OCR→LM jobs)
plus one **CPU collector**, using the existing Slurm dependency design.
The script runs the OCR extraction independently per repetition to measure
end-to-end execution, while sharing flat OCR between both OCR→LM models
*within* each repetition. All repetitions are queued when the command runs;
the Slurm scheduler manages their execution subject to your allocation.
For GPU-memory comparisons, check the actual allocated A100 variant (40 GB
or 80 GB) rather than assuming the `A100` label fixes VRAM.

To prevent confusing repeated measurements with older PP/LA experiments,
the script creates one **new series directory** per invocation:

```text
runs/repeated_baselines/
  baselines-<UTC-timestamp>-<unique-suffix>/
    series_manifest.tsv          # repeat_index, batch_id, paths, job IDs, git SHA
    repeat-001/
      submission.log
      batches/<batch-id>/benchmark_summary.csv
      <run-id>/predictions_scored.json
    repeat-002/
      ...
    repeat-003/
      ...
```

Each repetition has new batch/run UUIDs. **No existing `runs/` artifacts
are deleted or overwritten.** The script prints the exact series path.
After the CPU collectors finish, aggregate only that series:

```bash
.venv/bin/python scripts/aggregate_repeated_baselines.py \
  runs/repeated_baselines/<series-id>
```

This writes `repeated_runs.csv` (all individual runs and statuses) and
`repeat_summary.csv` (mean and sample SD for accuracy and latency) inside
that series folder. Pending or failed runs are **not** counted as zero.
It rejects different dataset manifest hashes, and it never includes
other historical experiments or PP/LA OCR. Monitor statuses in
`repeated_runs.csv` and rerun the aggregator after the collectors finish.

**Statistical interpretation:** all runs use the *same* 400 images, pinned
weights, `do_sample: false`, and default deterministic generation options.
Three repetitions are a reasonable initial check of runtime and software
reproducibility, but are **not 1,200 independent test documents per model**.
Report accuracy on 400 matched images, and estimate accuracy uncertainty
using a **paired document-level bootstrap** or similarly appropriate
within-document method. If accuracy differs negligibly across runs, further
repetition adds little statistical information; increase to five mainly to
stabilize time/resource measurements. Queuing and competing GPU activity
may still contribute to latency variation.

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
    ocr/paddleocr/            # original flat OCR text/timing cache
    ocr/paddleocr_layout/     # coordinate-aware JSON text/timing cache
    ocr/ppstructurev3/        # same parser engine, repaired Markdown v2
      index.json              # cache version, settings, timing, audit paths
      markdown/<key>.md       # final resolved text supplied to Qwen
      original_markdown/      # unmodified PP rendering, for audit
      raw_json/               # complete PP OCR and parsing results by page
      fallback_ocr/           # English flat-OCR recognized text for each image
      images/                 # PP's embedded crop images (where available)
      audit/                  # counts of recovered and unplaced OCR lines
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

`--ocr-only` **never resubmits direct VLM jobs**. For batches containing LA OCR or PP-StructureV3, it resumes unfinished `la_ocr_lm` or `ppstructure_lm` runs with their separate per-engine caches. It can reuse completed OCR cache documents and saves the retry job IDs in `runs/batches/<batch-id>/resumes/`. The CPU collector reuses existing scored predictions.

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
| `torch` and PaddlePaddle conflict on cuDNN/NCCL | Keep `.venv` separate from **all three Paddle environments**. If Paddle was accidentally installed in `.venv`, uninstall `paddlepaddle-gpu paddleocr paddlex` **there**, then install `requirements/helix-torch-repair-cu130.txt` and recheck `pip check`. |
| `libcuda.so.1` missing when importing Paddle on a login node | Expected on nodes without NVIDIA driver libraries; run the GPU smoke test via `sbatch`. If it also fails on an allocated A100, inspect the log. |
| `ModuleNotFoundError: torchvision` | Install `requirements/helix-pytorch-cu130.txt` and `-e '.[vlm]'` in `.venv`. |
| `No module named paddleocr` / missing Paddle backend | Install `-e '.[ocr]'` and `requirements/helix-paddle-cu130.txt` in `.venv-ocr`, not `.venv`. |
| OCR model download stalls | Inspect the OCR Slurm log and the model cache. `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` skips the host connectivity check, but **does not replace missing model weights**. |
| OCR + LM cannot find `ocr/paddleocr/index.json` | Inspect the shared flat OCR job; its failure commonly prevents the cache from being created. |
| LA OCR + LM cannot find `ocr/paddleocr_layout/index.json` | Verify `.venv-la-ocr`, inspect the `ulm-la-ocr` Slurm job, and run the `la-ocr` one-image smoke test. |
| PP-StructureV3 imports fail or doc parsing features are missing | Check `.venv-ppstructure/bin/python -m pip check`, ensure `-e '.[ppstructure]'` installed `paddleocr[doc-parser]`, and run the `ppstructure` GPU smoke test. |
| PP-StructureV3 Markdown cache is missing/incomplete | Inspect the `ulm-ppstructure` job log and `runs/batches/<batch-id>/ocr/ppstructurev3/index.json`; incomplete documents are resumable. |
| PP-StructureV3 produces empty Markdown | The parser job now permits initially empty PP Markdown, recovers text from full-image English OCR/PP internal OCR and raises an error only if the final Markdown is empty. Inspect `raw_json/`, `fallback_ocr/`, and `audit/`. |
| Earlier PP cache rejected after code update | Expected: the old Markdown-only cache has a different format. Submit a new `benchmark --models model-ppstructure.yml` batch, preserving previous results. |
| Scoring fails after predictions were saved | Inspect `runs/<run-id>/scoring.log`. The English KIE evaluator requirements omit `Polygon3`; rerun the CPU collector rather than GPU inference. |

## References

- [OCRBench v2: official dataset and evaluator](https://github.com/Yuliang-Liu/MultimodalOCR)
- [OCRBench v2 paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)
- [PyTorch CUDA 13.0 wheels](https://download.pytorch.org/whl/cu130/)
- [PaddlePaddle GPU installation](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/linux-pip_en.md)
- [PaddleOCR 3.x output fields: rec_texts, rec_scores and rec_boxes](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md)
- [PP-StructureV3 official usage and Markdown API](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)
- [PaddleOCR optional `doc-parser` installation](https://www.paddleocr.ai/latest/en/version3.x/installation.html)
