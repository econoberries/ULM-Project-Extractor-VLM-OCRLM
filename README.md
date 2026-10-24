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

## Installing on Helix

Run these **inside your Helix SSH session**, from your home directory:

```bash
git clone https://github.com/econoberries/ULM-Project-Extractor-VLM-OCRLM.git
cd ULM-Project-Extractor-VLM-OCRLM
```

**Do not use Helix's system `python3` if it resolves to Python 3.6.** This project requires Python 3.12 for our intended GPU environment. With `uv` available, initialize a project-local interpreter:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m unittest discover -s tests -v
```

Install and verify the appropriate **CUDA PyTorch, Transformers, Pillow and Accelerate** versions for the GPU model adapters. The previous Qwen3.5 experiment used Python 3.12.12, PyTorch 2.14.1+cu126 and Transformers 5.19.0; these versions are good starting reference points, **not proof that this new combined environment has been tested**. The `[vlm]` extra declares model dependencies:

```bash
uv pip install --python .venv/bin/python -e '.[vlm]'
```

For OCR stages, install a cluster-compatible `paddlepaddle-gpu` and `paddleocr`, validate PaddlePaddle sees CUDA, and record their versions. PaddlePaddle's CUDA build is installation-specific and is **not pinned by the lightweight package**. The official evaluator also needs its own dependencies:

```bash
git clone https://github.com/Yuliang-Liu/MultimodalOCR.git ~/MultimodalOCR
uv pip install --python .venv/bin/python -r ~/MultimodalOCR/OCRBench_v2/requirements.txt
```

If you already have `~/MultimodalOCR`, reuse the checkout instead of cloning it. For reproducible comparisons, pin the evaluator commit (earlier project reference: `4e363b3f499811b43d792ef0bb33932ef05eca27`). In particular, verify dependency compatibility before submitting GPU work.

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
