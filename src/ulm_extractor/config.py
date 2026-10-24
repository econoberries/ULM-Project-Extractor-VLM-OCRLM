"""Resolve explicit model configurations; no model auto-detection."""
from pathlib import Path
import re
import yaml

# Revisions from the previously audited implementation, not "latest" aliases.
PINNED = {
    "qwen35": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "minicpm": ("openbmb/MiniCPM-V-4.6", "36f34a661a4bd35d0dc2294cb044d2584646c7d3"),
    "qwen25": ("Qwen/Qwen2.5-14B-Instruct-1M", "620fad32de7bdd2293b3d99b39eba2fe63e97438"),
}
PIPELINES = {"qwen35": {"vlm", "ocr_lm"}, "minicpm": {"vlm"}, "qwen25": {"ocr_lm"}}

def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    if not isinstance(conf, dict) or conf.get("version") != 1:
        raise ValueError("model.yml must have version: 1")
    settings = conf.get("inference", {})
    if not isinstance(settings, dict):
        raise ValueError("inference must be a mapping")
    if settings.get("prompt_mode", "benchmark_native") != "benchmark_native":
        raise ValueError("Version 1 only supports unmodified OCRBench questions")
    if settings.get("batch_size", 1) != 1 or settings.get("do_sample", False) is not False:
        raise ValueError("Version 1 requires batch_size=1 and do_sample=false")
    cap = settings.get("max_new_tokens", 384)
    if type(cap) is not int or cap < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    models = conf.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list")
    seen = set()
    resolved = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Each model must be a mapping")
        exp_id, adapter, pipe, checkpoint = (item.get(k) for k in ("id", "adapter", "pipeline", "checkpoint"))
        if not isinstance(exp_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,62}", exp_id):
            raise ValueError("Each model needs a safe, unique id (letters, numbers, _ or -)")
        if exp_id in seen:
            raise ValueError(f"Duplicate model id: {exp_id}")
        seen.add(exp_id)
        if adapter not in PINNED or pipe not in PIPELINES[adapter]:
            raise ValueError(f"Unsupported adapter/pipeline for {exp_id}: {adapter}/{pipe}")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError(f"Missing checkpoint for {exp_id}")
        standard_checkpoint, standard_revision = PINNED[adapter]
        revision = item.get("revision")
        if revision is None:
            if checkpoint != standard_checkpoint:
                raise ValueError(f"{exp_id}: custom checkpoint requires an explicit immutable revision")
            revision = standard_revision
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{exp_id}: revision must be an immutable 40-character commit SHA")
        ocr = item.get("ocr")
        if pipe == "ocr_lm" and ocr != "paddleocr":
            raise ValueError(f"{exp_id}: OCR->LM requires ocr: paddleocr")
        if pipe == "vlm" and ocr is not None:
            raise ValueError(f"{exp_id}: direct VLM must not declare an OCR engine")
        overrides = item.get("inference") or {}
        if not isinstance(overrides, dict) or set(overrides) - {"max_new_tokens"}:
            raise ValueError(f"{exp_id}: only max_new_tokens overrides are supported")
        n = overrides.get("max_new_tokens", cap)
        if type(n) is not int or n < 1:
            raise ValueError(f"{exp_id}: max_new_tokens must be positive")
        resolved.append({
            "id": exp_id, "pipeline": pipe, "adapter": adapter,
            "checkpoint": checkpoint, "revision": revision,
            "ocr": ocr, "max_new_tokens": n, "batch_size": 1,
            "do_sample": False, "prompt_mode": "benchmark_native",
        })
    return resolved
