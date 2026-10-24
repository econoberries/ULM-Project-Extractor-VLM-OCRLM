# ULM Project Extractor — VLM and OCR→LM

Configuration-driven benchmarking of **400 English KIE documents** from OCRBench v2.

The two required inputs are an `EN_part/` image directory and `model.yml`. The benchmark finds `OCRBench_v2.json` beside `EN_part/` (or accepts `--annotations`), then submits isolated model jobs via SLURM. Predictions retain each original OCRBench question **without adding an output-format instruction**.

## Scope
- Direct vision-language models: Qwen3.5-9B and MiniCPM-V-4.6
- OCR→LM: shared PaddleOCR extraction + Qwen3.5-9B / Qwen2.5-14B-Instruct-1M
- Per-run UUIDs, official KIE F1, timing/resource metrics, and models-as-rows summary CSV
- A100 Helix scheduling with GPU dependencies and CPU result collection

See `model.yml` and the command-line help after installation. Model weights, OCRBench data, cached OCR outputs and raw predictions do not belong in Git.
