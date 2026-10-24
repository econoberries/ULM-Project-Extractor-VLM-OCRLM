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

## Helix installation: Python 3.12 and CUDA 13.0

This section reflects the **new extractor environment on Helix**, where Python 3.12 is available at `/usr/bin/python3.12` and the installed PyTorch reported **`2.14.1+cu130` (CUDA 13.0)**. The previous project's CUDA 12.6 environment is separate: do not copy its wheels into the new `.venv`.

**Install packages on the login node; run GPU calculations through SLURM jobs.** The current CUDA 13.0 setup is a concrete Helix installation recipe, not a claim that the entire four-model benchmark has already passed GPU validation.

### 1. Clone or update the repository

For a fresh checkout:

```bash
cd ~
git clone https://github.com/econoberries/ULM-Project-Extractor-VLM-OCRLM.git
cd ULM-Project-Extractor-VLM-OCRLM
```

If you already cloned the repository, just update it:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM
git pull --ff-only origin main
```

### 2. Create the virtual environment

Only create `.venv` if it does not already exist:

```bash
test -x .venv/bin/python || /usr/bin/python3.12 -m venv .venv

.venv/bin/python --version
.venv/bin/python -m pip --version
.venv/bin/python -m pip install -e .
```

Use `.venv/bin/python -m pip` for all installations. Do not rely on the default `python3`, and do not install packages into `~/ULM-VLM-OCR-IE/.venv`.

### 3. Install PyTorch and the matching Torchvision build

Our first real benchmark attempt revealed that **both direct VLM jobs failed because `torchvision` was not installed**, even though `torch` and `transformers` were present.

The repository includes [`requirements/helix-pytorch-cu130.txt`](requirements/helix-pytorch-cu130.txt), which specifies the matched pair: **PyTorch `2.14.1+cu130`** and **Torchvision `0.29.1+cu130`** from the [official PyTorch CUDA 13.0 wheels](https://download.pytorch.org/whl/cu130/torchvision/).

```bash
.venv/bin/python -m pip install -r requirements/helix-pytorch-cu130.txt
```

If PyTorch `2.14.1+cu130` is already installed, pip should reuse it rather than replace it. Avoid installing a CUDA 12.6 Torchvision wheel against this CUDA 13.0 Torch build.

### 4. Install PaddlePaddle GPU and PaddleOCR

Our first OCR job failed with `ModuleNotFoundError: No module named 'paddleocr'`. Because the shared OCR cache was never created, **both OCR + LM runs also failed**.

PaddleOCR is the document-recognition package; **PaddlePaddle GPU** is its inference backend. They are different packages, and both must be installed.

For the current CUDA 13.0 profile, [`requirements/helix-paddle-cu130.txt`](requirements/helix-paddle-cu130.txt) uses the official **PaddlePaddle GPU 3.3.0 CUDA 13.0 wheel**, as described in the [PaddlePaddle Linux pip installation guide](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/pip/linux-pip_en.md).

```bash
.venv/bin/python -m pip install -r requirements/helix-paddle-cu130.txt
```

Then install both the extractor's VLM and OCR extras:

```bash
.venv/bin/python -m pip install -e '.[vlm,ocr]'
```

The `[vlm]` extra includes **`torchvision`**, which the previous setup omitted. The `[ocr]` extra includes **`paddleocr`**, which the previous setup did not install. The CUDA-specific PaddlePaddle wheel remains separate because it must match the cluster GPU driver and environment.

**Do not install the CPU-only `paddlepaddle` alongside `paddlepaddle-gpu`.** If pip reports conflicts, examine them before proceeding. PyTorch's CUDA build version alone does not prove PaddlePaddle will run on the A100; the SLURM tests below do that.

### 5. Check installed packages before allocating GPUs

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/check_dependencies.py --stage all
.venv/bin/python -m unittest discover -s tests -v
```

The dependency checker verifies the required **installed distributions and Helix CUDA 13.0 versions**. It does not need a GPU, download weights, or submit a job. In particular, it checks for the `torchvision`, `paddlepaddle-gpu`, and `paddleocr` packages that caused the initial failures.

If your full evaluator is not yet available, clone its repository:

```bash
# Only when ~/MultimodalOCR does not already exist:
git clone https://github.com/Yuliang-Liu/MultimodalOCR.git ~/MultimodalOCR

.venv/bin/python -m pip install -r requirements/ocrbench-v2-kie-eval.txt
.venv/bin/python -m pip check
```

**Why a separate KIE requirements file?** The upstream `OCRBench_v2/requirements.txt` includes `Polygon3`, which failed to compile on Helix's Python 3.12 because the compiler could not find `Python.h`. Polygon is loaded dynamically only for the **text-spotting** metric, not English KIE. Our [`requirements/ocrbench-v2-kie-eval.txt`](requirements/ocrbench-v2-kie-eval.txt) keeps the other dependencies required to import and run the **unchanged official `eval.py` for KIE**, avoiding the unnecessary C-extension build. If you evaluate other OCRBench tasks, this selective environment may be insufficient.

Use the same official evaluator commit for every run (previous project reference: `4e363b3f499811b43d792ef0bb33932ef05eca27`). Check dependency compatibility again after installing its requirements.

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

**Do not rerun the entire benchmark merely to test an installation fix.** The first four-model attempt produced no scored documents: the two VLMs were missing Torchvision, and the shared OCR job was missing PaddleOCR. Validate the GPU and model smoke tests before spending time on another full run.

### Retry official scoring without rerunning GPU inference

If a model has **400 completed predictions** but `status=scoring_failed`, the `predictions.json` file is already usable. Our collector can retry scoring previously failed results, keep the same UUIDs, and preserve the complete evaluator traceback in each run's `scoring.log`.

For the batch `5dd8a6af-de7f-4d4c-b496-44068933a072`:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM
.venv/bin/python -m pip install -r requirements/ocrbench-v2-kie-eval.txt
.venv/bin/python -m pip check

sbatch --partition=cpu-single --cpus-per-task=2 --mem=4G \\
  --time=00:30:00 --job-name=ulm-rescore \\
  --output=rescore-%j.out \\
  slurm/collector.sbatch collect \\
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
| `No module named paddleocr` | Install `-e '.[ocr]'`; also install and verify **GPU PaddlePaddle** separately. |
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
