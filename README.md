# ULM Project Extractor with VLM and OCR + LM

Extracting information from document images has become an increasingly common task, especially with the development of language models (LMs). Two approaches are to use a **Vision-Language Model (VLM)** to interpret document images directly, or to first extract their text using **Optical Character Recognition (OCR)** and then process that text with an LM. Both aim to extract information accurately, but they may differ in processing time and computational requirements.

In this project, we compare **VLM and OCR + LM** approaches for document information extraction. We want to understand their trade-offs in **extraction accuracy, inference time, and computational resources**—and whether an OCR + LM pipeline can achieve results comparable to a direct VLM.

We develop a benchmarking tool for **Key Information Extraction (KIE)** using the **400 English KIE samples** in [OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html) (Fu et al., 2025). The dataset and official evaluation code are available in the authors' [MultimodalOCR repository](https://github.com/Yuliang-Liu/MultimodalOCR).

We compare **Qwen3.5-9B** and **MiniCPM-V-4.6** as end-to-end VLMs with **PaddleOCR + Qwen3.5-9B** and **PaddleOCR + Qwen2.5-14B-Instruct-1M**. Qwen3.5 appears in both approaches so we can examine the role of the input pipeline; MiniCPM provides another VLM architecture, and Qwen2.5 provides a text-focused LM comparison.

The tool runs on an **HPC cluster with SLURM**, where users specify the model experiments in a single `model.yml` file. It submits the jobs, records run-specific measurements, and gathers the results in a CSV with **one run per row**.

## Requirements

1. **Python 3.12** and `pip` in a project-local `.venv` (no `uv` required).
2. **SLURM and an NVIDIA GPU**, with the current scripts set up for **Helix A100**.
3. **OCRBench v2 English KIE images** in `OCRBench_v2/EN_part/` and the corresponding `OCRBench_v2.json` annotations in its parent folder.
4. **`model.yml`**, listing the models to benchmark.
5. **Model and evaluator dependencies**: PyTorch, Torchvision, Transformers, PaddleOCR with GPU-enabled PaddlePaddle, and the official OCRBench v2 evaluation code.

The benchmark command takes **two required inputs**: `--input .../EN_part` and `--models model.yml`. The annotation file is discovered automatically, or can be supplied using optional `--annotations`. The original OCRBench question is used **without adding output-formatting instructions**.

## Model configurations

Edit [`model.yml`](model.yml) to select experiments. The four initial configurations are:

| Experiment ID | Pipeline | Model |
| --- | --- | --- |
| `qwen35_vlm` | Direct VLM | Qwen3.5-9B |
| `minicpm_vlm` | Direct VLM | MiniCPM-V-4.6 |
| `qwen35_ocr` | PaddleOCR + LM | Qwen3.5-9B |
| `qwen25_ocr` | PaddleOCR + LM | Qwen2.5-14B-Instruct-1M |

Each entry specifies an experiment `id`, `pipeline`, `adapter`, and `checkpoint`; OCR + LM entries also specify `ocr: paddleocr`. Known model checkpoints have pinned revisions. The default output limit is `max_new_tokens: 384`, which can be overridden per model. We retain raw model predictions and record token-limit hits.

## Workflow

```text
OCRBench EN_part/ + model.yml
              |
      Validate 400 English KIE records
              |
      Create batch UUID + run UUIDs
              |
       +------+-----------------------+
       |                              |
  Shared PaddleOCR job          Direct VLM jobs
       |                              |
  Cached OCR text               Qwen / MiniCPM
       |
  OCR + LM jobs
  (Qwen3.5 / Qwen2.5)
       |                              |
       +------------------------------+
                      |
        CPU collector (afterany)
                      |
          Official KIE scoring
                      |
          benchmark_summary.csv
```

Each model runs in its own SLURM GPU job. The two OCR + LM models share a single PaddleOCR extraction for the same batch. The CPU collector runs after the jobs finish, even if some fail, so unsuccessful runs remain visible in the summary.

## Helix installation: separate PyTorch and PaddleOCR environments

Our Helix A100 setup uses **Python 3.12** and CUDA 13.0 wheels. **Keep PyTorch and PaddlePaddle GPU in different virtual environments.** Installing `paddlepaddle-gpu==3.3.0` into the existing PyTorch `.venv` replaced NVIDIA cuDNN and NCCL packages used by `torch==2.14.1+cu130`, causing a dependency conflict.

| Environment | Install | Used for |
| --- | --- | --- |
| `.venv/` | PyTorch 2.14.1+cu130, Torchvision 0.29.1+cu130, Transformers, official KIE evaluator | VLM jobs, OCR + LM jobs, scoring/collector and command line |
| `.venv-ocr/` | PaddlePaddle GPU 3.3.0, PaddleOCR 3.7.0, PaddleX 3.7.2 | Shared GPU PaddleOCR job and Paddle-specific tests |

The shared OCR stage writes text and timing data to disk. The LM stage reads those files in a different SLURM job: **it does not import PaddlePaddle**. The scripts automatically select the correct Python interpreter depending on job type.

Install packages on the login node; run GPU operations through SLURM. Do not install dependencies in the older `~/ULM-VLM-OCR-IE/.venv` environment. Helix's bare `python3` might be an older version; use `/usr/bin/python3.12` to create environments.

### 1. Get the latest code

If you have already cloned the project:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM
git pull --ff-only origin main
```

For a new checkout:

```bash
cd ~
git clone https://github.com/econoberries/ULM-Project-Extractor-VLM-OCRLM.git
cd ULM-Project-Extractor-VLM-OCRLM
```

### 2. Repair the existing PyTorch environment after the Paddle installation

This step applies if you already installed PaddlePaddle GPU into `.venv` and saw these pip warnings:

```text
torch 2.14.1 requires nvidia-cudnn-cu13==9.24.0.43
torch 2.14.1 requires nvidia-nccl-cu13==2.30.7
```

**Do not delete `runs/`, recreate the benchmark, or reinstall all model weights.** The correction below changes only the Python packages in `.venv`:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM

# Remove Paddle packages from the PyTorch environment.
.venv/bin/python -m pip uninstall -y paddlepaddle-gpu paddleocr paddlex

# Restore the two versions required by the already-installed PyTorch.
.venv/bin/python -m pip install -r requirements/helix-torch-repair-cu130.txt

# Ensure the extractor and VLM dependencies are still installed.
.venv/bin/python -m pip install -e '.[vlm]'

# Check that the entire PyTorch environment is consistent.
.venv/bin/python -m pip check
.venv/bin/python scripts/check_dependencies.py --stage vlm
```

If `pip check` reports any other conflicts, **stop and examine its full output** rather than repeatedly installing frameworks on top of each other. If `.venv` does not exist yet, create it with `/usr/bin/python3.12 -m venv .venv` and install `requirements/helix-pytorch-cu130.txt` followed by `-e '.[vlm]'`; the repair-only file is unnecessary in a fresh environment.

### 3. Create the separate PaddleOCR environment

Do not install PaddlePaddle GPU in `.venv`. Instead:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM

# Create only if it doesn't already exist.
test -x .venv-ocr/bin/python || /usr/bin/python3.12 -m venv .venv-ocr

# PaddleOCR + PaddleX; installs the extractor code but no Torch extra.
.venv-ocr/bin/python -m pip install -e '.[ocr]'

# Download/install CUDA 13.0 PaddlePaddle GPU INTO .venv-ocr only.
.venv-ocr/bin/python -m pip install -r requirements/helix-paddle-cu130.txt

.venv-ocr/bin/python -m pip check
.venv-ocr/bin/python scripts/check_dependencies.py --stage ocr
```

First-time Paddle GPU installation downloads a large wheel and several CUDA runtime packages. That's expected. Its CUDA packages now belong only to `.venv-ocr`; they cannot replace the working PyTorch packages in `.venv`.

Check the backend import without requiring an allocated GPU:

```bash
.venv-ocr/bin/python - <<'PY'
import paddle
print("PaddlePaddle:", paddle.__version__)
print("Compiled with CUDA:", paddle.is_compiled_with_cuda())
PY
```

We expect `Compiled with CUDA: True`. Actual A100 operations must still pass the SLURM tests.

### 4. Install official OCRBench v2 KIE scoring requirements in the PyTorch environment

The official evaluator executes on the CPU collector, which uses `.venv`, **not** `.venv-ocr`. Our English KIE-only requirements avoid the unused `Polygon3` C extension (which requires `Python.h` on Helix), while preserving the official KIE scorer.

```bash
# Only clone the official repository if you do not already have it:
test -f "$HOME/MultimodalOCR/OCRBench_v2/eval_scripts/eval.py" || \
  git clone https://github.com/Yuliang-Liu/MultimodalOCR.git "$HOME/MultimodalOCR"

.venv/bin/python -m pip install -r requirements/ocrbench-v2-kie-eval.txt

.venv/bin/python -m pip check
.venv/bin/python scripts/check_dependencies.py --stage all
.venv/bin/python -m unittest discover -s tests -v
```

`--stage all` checks both virtual environments separately. It verifies the Python packages and known CUDA library versions; it does **not** run GPU inference. For reproducibility, use the same official evaluator commit across runs (reference: `4e363b3f499811b43d792ef0bb33932ef05eca27`).

## GPU verification with SLURM

We use **`sbatch`**, just like our earlier Helix experiments—**no interactive `salloc` session is needed**. The test job in [`slurm/check_environment.sbatch`](slurm/check_environment.sbatch) requests one A100, four CPUs, 16 GiB system RAM and up to 20 minutes by default.

Run each test from the repository root. They perform only small checks, **not a 400-document benchmark**.

### Test 1: PyTorch CUDA

```bash
sbatch slurm/check_environment.sbatch torch
```

Expected result in the SLURM log: `CUDA available: True`, the A100 name, and `PyTorch CUDA: PASSED`. This includes a real GPU matrix multiplication.

### Test 2: PaddlePaddle CUDA

```bash
sbatch slurm/check_environment.sbatch paddle
```

Expected: `CUDA compiled: True`, a visible A100, and `PaddlePaddle CUDA: PASSED`. The frameworks are tested in **separate Python processes**.

### Test 3: PaddleOCR on one image

```bash
EN_PART="$HOME/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part"
IMAGE="$(find "$EN_PART" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)"
echo "$IMAGE"

if [[ -n "$IMAGE" && -f "$IMAGE" ]]; then
  sbatch slurm/check_environment.sbatch ocr "$IMAGE"
else
  echo "OCRBench image not found; adjust EN_PART."
fi
```

Expected: `PaddleOCR single-image inference: PASSED` and a sample of extracted text. PaddleOCR may download its recognition weights on the first execution.

### Test 4: Optional Qwen adapter inference on one document

After PyTorch succeeds, try a single real Qwen3.5 image/question (temporary 64-token cap, no benchmark scoring):

```bash
sbatch --mem=64G --time=00:45:00 \
  slurm/check_environment.sbatch model "$EN_PART" qwen35_vlm
```

Expected: the GPU name, model output token count, prediction sample, and `Single-document model inference: PASSED`. This is heavier than the framework checks; model-loading compatibility is **not** established just because Torchvision imports.

### Reading SLURM status and logs

Each submission returns a job number. Replace `12345678` with the actual number:

```bash
squeue -j 12345678
sacct -j 12345678 --format=JobID,State,ExitCode,Elapsed
cat gpu-check-12345678.out
```

A missing `.out` file immediately after `sbatch` often means the job is still pending. A successful test has a `PASSED` message and typically `COMPLETED 0:0`. These GPU tests cannot run on GitHub Actions and still need validation on your Helix allocation.

## Run the benchmark

Validate the 400 records and SLURM commands **without submitting jobs**:

```bash
.venv/bin/extractor benchmark \
  --input ~/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part \
  --models model.yml \
  --dry-run
```

After the environment tests and a one-document model inference succeed, submit the complete experiments:

```bash
.venv/bin/extractor benchmark \
  --input ~/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part \
  --models model.yml
```

Default scheduling uses `gpu-single`, `gpu:A100:1`, four CPUs, 64 GiB host RAM and two hours per GPU job; the collector uses `cpu-single`. The official evaluator defaults to a checkout in `../MultimodalOCR/OCRBench_v2/eval_scripts/eval.py` relative to the project. Override with `--evaluator /path/to/eval.py` if needed.

**Do not rerun the entire benchmark to fix the OCR environment.** Our latest batch already has **400/400 officially scored direct VLM predictions each** (Qwen3.5-9B F1 **0.8707**; MiniCPM-V-4.6 F1 **0.5593**). Only the shared OCR job and the two OCR + LM runs failed. After verifying PaddlePaddle GPU and the single-image OCR test, use **`extractor resume --ocr-only`** below.

### Resume the existing batch — no direct VLM inference

**Use this instead of `extractor benchmark` when repairing a failed batch.** A fresh `benchmark` command creates new run UUIDs and schedules all configured models; `resume` reads the **original immutable** `batch.json`, keeps its run IDs, checks which predictions already exist, and submits only the incomplete jobs. `--ocr-only` adds an explicit guarantee: **no direct VLM GPU jobs are submitted**, regardless of their recorded state.

Our existing batch is `5dd8a6af-de7f-4d4c-b496-44068933a072`. First repair `.venv`, install Paddle GPU **only into `.venv-ocr`**, and make sure the PyTorch, PaddlePaddle and single-image OCR SLURM checks have **PASSED**.

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM
git pull --ff-only origin main
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/check_dependencies.py --stage all

BATCH="$PWD/runs/batches/5dd8a6af-de7f-4d4c-b496-44068933a072/batch.json"

# Verify the plan. This submits nothing and writes no files.
.venv/bin/extractor resume --batch "$BATCH" --ocr-only --dry-run

# Only after reviewing the jobs displayed above:
.venv/bin/extractor resume --batch "$BATCH" --ocr-only
```

For this batch, the plan should show: **SKIP `qwen35_vlm` and `minicpm_vlm`**, submit one shared PaddleOCR GPU job, then **`qwen35_ocr` and `qwen25_ocr`** GPU jobs depending on OCR, and finally a CPU-only collector. If all 400 OCR texts were previously cached, `resume` can reuse the cache and omit the OCR GPU job. The OCR worker resumes any incomplete cache documents; model workers also reuse valid saved document predictions.

The original `batch.json`, `jobs.json`, existing VLM `predictions.json`, and saved VLM scores are **not overwritten**. Each resubmission's job IDs are recorded separately under `runs/batches/<batch-id>/resumes/`. SLURM logs remain under that batch's `logs/` directory. The collector reuses already scored official VLM outputs and updates both the batch-specific and top-level comparison CSV.

Check the completed run and summary:

```bash
cat runs/batches/5dd8a6af-de7f-4d4c-b496-44068933a072/jobs.json
ls runs/batches/5dd8a6af-de7f-4d4c-b496-44068933a072/resumes/
squeue -u "$USER"
cat runs/batches/5dd8a6af-de7f-4d4c-b496-44068933a072/benchmark_summary.csv
```

**Safety:** `resume` refuses to schedule a selected model still marked `submitted` or `running`. Let the old SLURM job finish and inspect `sacct` before attempting a retry; never submit two workers simultaneously for the same run ID. A batch with scored VLMs is safe to resume with `--ocr-only`.

### Retry official scoring without rerunning GPU inference

If a model has **400 completed predictions** but `status=scoring_failed`, the `predictions.json` file is already usable. Our collector can retry scoring previously failed results, keep the same UUIDs, and preserve the complete evaluator traceback in each run's `scoring.log`.

For the batch `5dd8a6af-de7f-4d4c-b496-44068933a072`:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM
.venv/bin/python -m pip install -r requirements/ocrbench-v2-kie-eval.txt
.venv/bin/python -m pip check

sbatch --partition=cpu-single --cpus-per-task=2 --mem=4G \
  --time=00:30:00 --job-name=ulm-rescore \
  --output=rescore-%j.out \
  slurm/collector.sbatch collect \
  --batch "$PWD/runs/batches/5dd8a6af-de7f-4d4c-b496-44068933a072/batch.json"
```

This submits a **CPU-only scoring job**. It uses the original OCRBench v2 evaluator without any changes to the scoring functions. It does not load models, use an A100, regenerate predictions, or replace the previous run UUIDs. View the batch-scoped `benchmark_summary.csv` and the SLURM log afterward. If scoring still fails, inspect `runs/<run-uuid>/scoring.log` for the full official evaluator traceback.

## Outputs

```text
runs/
  <run-uuid>/
    run.json                 # configuration, UUID, status/errors
    metadata.json            # GPU, library versions, load time
    predictions.json         # raw predictions and per-document timings
    predictions_scored.json  # scores, if evaluation succeeded
    metrics.json             # aggregated metrics and status
  batches/<batch-uuid>/
    batch.json
    documents.json
    jobs.json                # SLURM job IDs
    logs/                    # model, OCR and collector logs
    ocr/paddleocr/           # shared OCR texts and timings
    benchmark_summary.csv    # authoritative summary for this batch
  benchmark_summary.csv      # latest collected batch (convenience copy)
```

Each row of `benchmark_summary.csv` identifies a **run UUID** and includes its model configuration, GPU, status, completed documents, overall and per-dataset official KIE F1 (only for fully scored runs), latency statistics, resource measurements, output token counts, and token-limit hits. Failed runs remain visible and must **not** be interpreted as having F1 = 0.

Inference timing includes model input preparation, generation and decoding; OCR + LM pipeline timing also includes measured OCR cost, even when OCR text is cached. Results are hardware-aware, **not hardware-independent**—we should compare timings on equivalent GPU configurations.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `uv: command not found` | Use `/usr/bin/python3.12 -m venv` and `.venv/bin/python -m pip`; `uv` is optional. |
| `.venv/bin/python: No such file or directory` | Create the virtual environment from the repository root; verify Python and pip before installing. |
| `Torchvision library ... was not found` | Install `requirements/helix-pytorch-cu130.txt`, then the `[vlm]` extra. Match the Torchvision **CUDA 13.0** wheel to the installed CUDA 13.0 Torch build. |
| `No module named paddleocr` | Install `-e '.[ocr]'` in **`.venv-ocr`**, then install `requirements/helix-paddle-cu130.txt` in that same OCR environment. |
| `Engine 'paddle_static' ... dependency 'paddlepaddle' is not installed` | PaddleOCR and PaddleX exist, but `paddle` is missing. Install `paddlepaddle-gpu` in **`.venv-ocr` only**, verify `import paddle`, and run PaddlePaddle/OCR smoke tests. |
| `torch 2.14.1 requires nvidia-cudnn-cu13 ...` | PaddlePaddle was installed in the Torch environment and downgraded shared CUDA packages. Remove the Paddle packages from `.venv`, restore Torch CUDA library versions using `requirements/helix-torch-repair-cu130.txt`, and create `.venv-ocr` instead. |
| Need to resume only OCR + LM | Use `extractor resume --batch /path/to/original/batch.json --ocr-only`, preceded by `--dry-run`. Do not submit a fresh `extractor benchmark`. |
| `scoring_failed` after 400 completed predictions | The GPU inference succeeded but the official evaluator failed. Inspect `runs/<run-uuid>/scoring.log`; install `requirements/ocrbench-v2-kie-eval.txt`. **Do not rerun model inference.** Resubmit the CPU collector for the existing batch, as shown above. |
| `Polygon3` build error: `Python.h: No such file or directory` | For **English KIE only**, install `requirements/ocrbench-v2-kie-eval.txt` instead of upstream's full requirements file. It omits text-spotting-only Polygon3; the official scoring code is unchanged. |
| OCR + LM reports missing `ocr/paddleocr/index.json` | Inspect the shared OCR job log first; the missing cache is usually a downstream symptom of an OCR failure. |
| GPU check shows `False` or CUDA errors | Inspect `nvidia-smi`, the SLURM node allocation, CUDA wheel compatibility, and the full job log; do not blindly reinstall. |
| `gpu-check-<id>.out` missing | Check `squeue` and `scontrol show job <id>`; SLURM may not have started the job yet. |

If a full benchmark fails, inspect its `runs/batches/<batch-uuid>/logs/` directory and the individual `runs/<run-uuid>/run.json` files before resubmitting.

## Tests and references

Run the no-GPU tests on the login node:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

GitHub Actions also runs offline tests. CUDA tests require an allocated Helix GPU.

- [OCRBench v2 dataset and official evaluation code](https://github.com/Yuliang-Liu/MultimodalOCR)
- [PyTorch CUDA 13.0 Torchvision wheel index](https://download.pytorch.org/whl/cu130/torchvision/)
- [PaddlePaddle Linux GPU installation guide](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/linux-pip_en.md)
