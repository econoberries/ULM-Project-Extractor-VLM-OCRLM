# ULM Project Extractor — VLM and OCR→LM

A **SLURM-first, configuration-driven benchmark runner** for 400 English key-information-extraction records from OCRBench v2.

The benchmark has **two required inputs**: the path to the benchmark's `EN_part/` directory and a `model.yml` file. It reads `OCRBench_v2.json` from the parent directory, or accepts an optional `--annotations` override. The model receives the **original benchmark question unchanged**.

## Workflow

```text
EN_part/ + model.yml
        |
  Validate 400 OCRBench English KIE records
        |
  Assign batch UUID and one immutable run UUID per configuration
        |
        +--- shared PaddleOCR GPU job (only when OCR-LM models are present)
        |           |
        |      cache 400 OCR texts
        |           |
        +--- direct VLM GPU jobs (independent)
        +--- OCR->LM GPU jobs (wait until OCR finishes; check cache before loading)
                         |
            CPU collector (SLURM afterany)
                         |
        official evaluator + one models-as-rows CSV
```

The orchestration creates one GPU SLURM job per configured model and a single shared PaddleOCR job for the OCR→LM pipelines. All inference jobs use the same source records in the original order. The collector runs on the `cpu-single` partition after the submitted jobs finish, including failures, so unsuccessful experiments appear as such rather than silently disappearing. For OCR-LM, if OCR fails, dependent jobs exit without loading the LM.

## Model definitions

Edit [model.yml](model.yml). Supported adapter/pipeline combinations:

| Adapter | Pipeline | Model |
| --- | --- | --- |
| `qwen35` | `vlm` | Qwen3.5-9B |
| `minicpm` | `vlm` | MiniCPM-V-4.6 |
| `qwen35` | `ocr_lm` | PaddleOCR + Qwen3.5-9B |
| `qwen25` | `ocr_lm` | PaddleOCR + Qwen2.5-14B-Instruct-1M |

All entries require `id`, `pipeline`, `adapter`, and `checkpoint`; OCR→LM additionally requires `ocr: paddleocr`. The known checkpoint IDs are resolved to pinned revisions from our earlier implementations. Other checkpoints require an explicitly specified immutable revision. A single global `max_new_tokens` (default **384**) can be overridden per experiment using `inference: {max_new_tokens: 512}`.

The extractor adds **no output-formatting prompt**. It preserves raw generation strings and counts token-limit hits, without automatic truncation retries. Current primary profile uses greedy, batch-one inference.

## Installing on Helix (without `uv`)

The Helix login node may not have `uv` installed. **This project does not require `uv`**: use Python 3.12's built-in `venv` module and `pip` instead.

Run these commands **inside your Helix SSH session**. For an initial checkout:

```bash
cd ~
git clone https://github.com/econoberries/ULM-Project-Extractor-VLM-OCRLM.git
cd ULM-Project-Extractor-VLM-OCRLM
```

If the new repository is already cloned, **do not clone again**; instead run:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM
git pull --ff-only origin main
```

### Step 1 — Confirm Python 3.12

In the Helix setup we verified both of these commands:

```bash
~/ULM-VLM-OCR-IE/.venv/bin/python --version  # Python 3.12.12 from the previous project
command -v python3.12                        # /usr/bin/python3.12
```

**Use `/usr/bin/python3.12` for the new environment.** Do not rely on bare `python3` (which previously resolved to Python 3.6) or install into the old project's `.venv`.

### Step 2 — Create and verify a fresh environment

From `~/ULM-Project-Extractor-VLM-OCRLM`:

```bash
/usr/bin/python3.12 -m venv .venv
.venv/bin/python --version
.venv/bin/python -m pip --version
```

The Python version should be 3.12, and the pip command should display its installed version. If either check fails, **fix the environment before proceeding**. This environment is local to the new repository; it does not change `~/ULM-VLM-OCR-IE/.venv`.

### Step 3 — Install the lightweight extractor and run offline tests

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/extractor --help
```

The base installation provides the benchmark config loader, dataset validator, SLURM orchestrator, and CPU collector. These checks do not require a GPU or submit SLURM jobs.

### Step 4 — Install GPU dependencies before actual inference

For the VLM/LM adapters, install the extra dependencies into the **same new environment**:

```bash
.venv/bin/python -m pip install -e '.[vlm]'
```

**Verify CUDA PyTorch compatibility on a GPU compute node before submitting full jobs.** The historical Qwen3.5 experiment used Python 3.12.12, PyTorch 2.14.1+cu126 and Transformers 5.19.0, but this **new environment and its model adapters have not yet been GPU-validated**. A default pip installation is not proof that the right CUDA wheel was selected.

For OCR stages, separately install and verify a cluster-compatible `paddlepaddle-gpu` and `paddleocr`. Their CUDA compatibility must be checked on Helix; a verified GPU PaddlePaddle build is not currently pinned by the base package.

The official evaluator needs its checkout and dependencies. If `~/MultimodalOCR` already exists, **reuse it** rather than cloning again:

```bash
# Only if the official evaluator repository is not already present:
git clone https://github.com/Yuliang-Liu/MultimodalOCR.git ~/MultimodalOCR

.venv/bin/python -m pip install -r ~/MultimodalOCR/OCRBench_v2/requirements.txt
```

For reproducible comparisons, use the same evaluator commit across runs (earlier reference: `4e363b3f499811b43d792ef0bb33932ef05eca27`). Check dependency compatibility after installing the evaluator's requirements.

### Step 5 — Verify GPU frameworks with small SLURM jobs

The quick tests are **committed to this repository**. You do **not** need `salloc`, `uv`, or an interactive GPU shell. Submit the tests with `sbatch` as in the previous Qwen project. SLURM allocates the A100 automatically, runs one short test on a compute node, and records the results in a log file.

Start from the new repository on the Helix login node:

```bash
cd ~/ULM-Project-Extractor-VLM-OCRLM
git pull --ff-only origin main
.venv/bin/python -m unittest discover -s tests -v
```

The test job script is `slurm/check_environment.sbatch` and its Python programs are in `scripts/`. It requests **one A100 GPU**, **4 CPUs**, **16 GiB host RAM**, and **20 minutes** by default. Jobs run the test in the new `.venv`; they do not alter the original Qwen environment or submit any full benchmark.

**5a. PyTorch + Transformers CUDA test**

```bash
sbatch slurm/check_environment.sbatch torch
```

The output should show `CUDA available: True`, an A100 GPU, and `PyTorch CUDA: PASSED`. This executes an actual 512×512 GPU matrix multiplication in PyTorch.

**5b. PaddlePaddle CUDA test (separate Python process)**

```bash
sbatch slurm/check_environment.sbatch paddle
```

The output should show `CUDA compiled: True`, a visible GPU, and `PaddlePaddle CUDA: PASSED`. This performs a GPU matrix multiplication using PaddlePaddle, not PyTorch. If `paddle` is missing or compiled CPU-only, install a **Helix-compatible** `paddlepaddle-gpu` distribution into `.venv` after checking the node's CUDA driver and wheel compatibility. Do not mix CPU-only and GPU PaddlePaddle installations.

**5c. Real PaddleOCR inference on one OCRBench image**

After the PaddlePaddle test succeeds, select one actual image. Adjust `EN_PART` if the dataset is stored elsewhere:

```bash
EN_PART="$HOME/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part"

IMAGE="$(find "$EN_PART" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)"
echo "$IMAGE"
if [[ -n "$IMAGE" && -f "$IMAGE" ]]; then
  sbatch slurm/check_environment.sbatch ocr "$IMAGE"
else
  echo "No OCRBench image found; check EN_PART."
fi
```

This creates **one GPU job and processes only that image**, using the exact PaddleOCR initialization flags from `src/ulm_extractor/ocr.py`. Expected output includes a number of recognized text lines and `PaddleOCR single-image inference: PASSED`. First-time PaddleOCR weight downloads can take additional time; downloads are not benchmark measurements. If model downloads fail due to compute-node network restrictions, configure the model cache separately.

**5d. Optional direct VLM smoke test on ONE document**

This step is useful after PyTorch succeeds and model weights are accessible. It actually loads the Qwen3.5 adapter and runs inference on the *first* English KIE image with the **unaltered OCRBench question**. It does not evaluate or process 400 documents, does not change `model.yml`, and uses a **64-token temporary cap** only for the smoke test.

```bash
sbatch --mem=64G --time=00:45:00 \
  slurm/check_environment.sbatch model "$EN_PART" qwen35_vlm
```

Expected output: model ID, GPU, number of tokens generated, a sample prediction, and `Single-document model inference: PASSED`. This test may need model downloads and uses significantly more GPU memory/time than the three framework tests, so it is **optional** until you are ready to verify actual model inference. If your config uses a different ID, replace `qwen35_vlm`.

**How to read the test results**

Each `sbatch` command prints a job ID such as `Submitted batch job 12345678`. Check its status:

```bash
squeue -u "$USER"
```

After the job exits, inspect the corresponding log **using the actual job number**:

```bash
cat gpu-check-12345678.out
```

Use `tail -n 60 gpu-check-12345678.out` for a shorter error report. You can also check the exit state with:

```bash
sacct -j 12345678 --format=JobID,State,ExitCode
```

Do not interpret merely seeing `nvidia-smi` as success: look for each script's `PASSED` line. If a job fails, **share the log before changing library versions**.

The offline unit tests and GitHub Actions only validate the Python logic; the four GPU smoke tests require an actual Helix A100 job and have not been executed by GitHub Actions.

### Troubleshooting the initial Helix setup

| Symptom | Explanation and next action |
| --- | --- |
| `-bash: uv: command not found` | `uv` is not available; the documented installation uses `/usr/bin/python3.12 -m venv` and `python -m pip` instead. |
| `.venv/bin/python: No such file or directory` | The environment has not been created successfully. Run Step 2 from the repository root, then verify its Python and pip. |
| `python3 --version` shows Python 3.6 | Do not use that interpreter for this project; explicitly use `/usr/bin/python3.12` when creating the environment. |
| `No module named pip` inside `.venv` | Stop before installing packages; verify that the system's Python 3.12 provides `venv`/`ensurepip` or use an available cluster-supported Python 3.12 module. Avoid installing packages globally. |

## Run a dry-run first (no SLURM jobs)

If the original OCRBench v2 dataset is still in your previous repository, you may use it directly instead of copying images:

```bash
.venv/bin/extractor benchmark \
  --input ~/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part \
  --models model.yml \
  --dry-run
```

This validates all 400 image paths and prints the GPU job and dependency graph. It **does not call `sbatch` or create output files**.

The default evaluator path is `../MultimodalOCR/OCRBench_v2/eval_scripts/eval.py` relative to this repository.

## Submit the full benchmark (GPU hours consumed)

Only after confirming the environment and testing a small GPU inference separately:

```bash
.venv/bin/extractor benchmark \
  --input ~/ULM-VLM-OCR-IE/data/OCRBench_v2/EN_part \
  --models model.yml
```

Defaults are tailored to **Helix A100**: `--partition gpu-single --gres gpu:A100:1 --gpu-family A100` (4 CPUs, 64G RAM, 2h per GPU job). The collector uses `cpu-single`, 1 CPU and 4G RAM. You may override scheduling flags for other clusters. The worker checks the actual GPU name before importing/loading models.

To use an annotation JSON outside the usual parent folder, add `--annotations /path/to/OCRBench_v2.json`. To use a differently located official scorer, add `--evaluator /path/to/eval.py`.

**This implementation has offline orchestration tests but has not yet been validated by a complete SLURM/GPU run on Helix.** The workflow is implemented, not a claim that A100 and H100 model dependencies are already verified.

## Outputs

```text
runs/
  <run-uuid>/
    run.json                  # status, run UUID, resolved model config
    metadata.json             # GPU, model load time, versions, dataset hashes
    predictions.json          # per-document raw predictions + timings
    predictions_scored.json   # official scored predictions (if fully completed)
    metrics.json              # single model row / failure status
  batches/<batch-uuid>/
    batch.json                # task execution contract
    documents.json            # 400 selected OCRBench annotation records
    jobs.json                 # actual SLURM job IDs
    logs/                     # job logs
    ocr/paddleocr/            # shared OCR text and timings
    benchmark_summary.csv     # authoritative one-row-per-run comparison
  benchmark_summary.csv       # convenience snapshot of last collected batch
```

The comparison CSV contains the UUID `run_id`, descriptive `experiment_id`, checkpoint and revision, pipeline, actual GPU name, completion status, official overall and per-dataset KIE F1, mean/median/P95/total per-document latency, OCR and LM time, peak PyTorch allocated memory (not total process VRAM), model load time, output-token statistics, and token-limit hits. F1 is only published for 400 valid completed predictions.

**Timing is hardware-aware, not hardware-independent:** GPU synchronization brackets preprocessing, generation and decoding; OCR extraction is measured separately and cached, while OCR→LM end-to-end latency includes OCR cost even when reused. Hardware metadata is recorded, and A100/H100 latency should not be treated as directly interchangeable.

Data, weights, raw predictions and caches are excluded from Git. Historical completed experiments remain in the earlier repositories and are **not** automatically labeled as results from the new protocol.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests do not require a GPU, Slurm installation, benchmark download or model weights. GitHub Actions runs them on pushes and pull requests.
