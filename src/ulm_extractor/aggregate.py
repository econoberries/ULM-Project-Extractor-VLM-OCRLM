"""Evaluate completed predictions with the official scorer and collect one CSV row per run."""
from pathlib import Path
import math
import statistics
import subprocess
import sys

import pandas as pd

from .storage import read_json, write_json

DATASETS = ("FUNSD", "POIE", "Receipt", "SROIE")

def p95(values):
    values = sorted(values)
    if not values:
        return None
    position = 0.95 * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)

def average(values):
    return statistics.mean(values) if values else None

def run_summary(batch, spec):
    run_dir = Path(batch["output_root"]) / spec["run_id"]
    status_path = run_dir / "run.json"
    state = read_json(status_path) if status_path.exists() else {}
    path = run_dir / "predictions.json"
    predicted = read_json(path) if path.exists() else []
    successful = [r for r in predicted if isinstance(r.get("predict"), str) and
                  not r.get("error") and isinstance(r.get("inference_seconds"), (int, float))]
    row = {
        "run_id": spec["run_id"], "batch_id": batch["batch_id"],
        "experiment_id": spec["id"], "model": spec["checkpoint"],
        "model_revision": spec["revision"],
        "pipeline": spec["pipeline"], "ocr_engine": spec["ocr"],
        "gpu_name": None, "status": state.get("status", "missing"),
        "completed_documents": len(successful), "total_documents": 400,
        "official_kie_f1": None, "funsd_f1": None, "poie_f1": None,
        "receipt_f1": None, "sroie_f1": None,
        "mean_latency_s": average([r["inference_seconds"] for r in successful]),
        "median_latency_s": (statistics.median([r["inference_seconds"] for r in successful])
                             if successful else None),
        "p95_latency_s": p95([r["inference_seconds"] for r in successful]),
        "total_latency_s": sum(r["inference_seconds"] for r in successful),
        "mean_ocr_time_s": average([r["ocr_seconds"] for r in successful
                                    if isinstance(r.get("ocr_seconds"), (int, float))]),
        "mean_lm_time_s": average([r["lm_seconds"] for r in successful
                                   if isinstance(r.get("lm_seconds"), (int, float))]),
        "peak_torch_memory_gib": (max((max(r.get("peak_allocated_gpu_bytes") or [0])
                                      for r in successful), default=0) / 1024**3
                                  if successful else None),
        "model_load_time_s": None,
        "mean_output_tokens": average([r["output_tokens"] for r in successful
                                       if isinstance(r.get("output_tokens"), int)]),
        "token_limit_hits": sum(bool(r.get("hit_token_limit")) for r in successful),
        "max_new_tokens": spec["max_new_tokens"],
    }
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        row["gpu_name"] = ", ".join(meta.get("gpu_names", [])) or None
        row["model_load_time_s"] = meta.get("model_load_time_s")
    # A completed or previously failed scoring job can be reevaluated with the
    # same saved predictions, without rerunning GPU inference or changing UUIDs.
    if state.get("status") in {"completed", "scoring_failed", "scored"} and len(successful) == 400 and len(predicted) == 400:
        try:
            scored_path = run_dir / "predictions_scored.json"
            command = [
                sys.executable, batch["evaluator"], "--input_path", str(path),
                "--output_path", str(scored_path)
            ]
            try:
                scorer = subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                # Preserve the root cause from the official evaluator, not just
                # the subprocess exit code, which hides missing dependencies.
                log = run_dir / "scoring.log"
                log.write_text(
                    "Official OCRBench evaluator failed\n"
                    + "Command: " + " ".join(command) + "\n"
                    + "Exit status: " + str(exc.returncode) + "\n"
                    + "STDOUT:\n" + str(exc.stdout or "") + "\n"
                    + "STDERR:\n" + str(exc.stderr or "") + "\n",
                    encoding="utf-8",
                )
                detail = (exc.stderr or exc.stdout or "").strip().splitlines()
                last_line = detail[-1] if detail else "No evaluator traceback was captured"
                raise RuntimeError(
                    f"Official OCRBench evaluator exited {exc.returncode}: {last_line}. "
                    f"Full traceback: {log}"
                ) from exc
            scored = read_json(scored_path)
            if len(scored) != 400 or any(not isinstance(r.get("score"), (int, float)) for r in scored):
                raise ValueError("Official evaluator did not provide 400 numeric scores")
            row["official_kie_f1"] = average([r["score"] for r in scored])
            for name, field in [("FUNSD", "funsd_f1"), ("POIE", "poie_f1"),
                                ("Receipt", "receipt_f1"), ("SROIE", "sroie_f1")]:
                row[field] = average([r["score"] for r in scored if
                                      str(r.get("dataset_name", "")).casefold() == name.casefold()])
            row["status"] = "scored"
            state["status"] = "scored"
            write_json(status_path, state)
        except Exception as exc:
            row["status"] = "scoring_failed"
            row["scoring_error"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "scoring_failed"
            state["error"] = row["scoring_error"]
            write_json(status_path, state)
    write_json(run_dir / "metrics.json", row)
    return row

def collect(batch):
    rows = [run_summary(batch, spec) for spec in batch["runs"]]
    frame = pd.DataFrame(rows)
    directory = Path(batch["output_root"]) / "batches" / batch["batch_id"]
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / "benchmark_summary.csv"
    frame.to_csv(filename, index=False)
    # Convenience snapshot; the batch-scoped file is the authoritative output.
    frame.to_csv(Path(batch["output_root"]) / "benchmark_summary.csv", index=False)
    print(frame[["run_id", "experiment_id", "status", "official_kie_f1",
                 "mean_latency_s", "gpu_name"]].to_string(index=False))
    print(f"Saved {filename}")
    return filename
