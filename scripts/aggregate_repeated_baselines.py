"""Aggregate a repeated-baseline series without mixing other historical batches.

Usage:
    .venv/bin/python scripts/aggregate_repeated_baselines.py \
        runs/repeated_baselines/<series-id>

Only fully scored 400-document runs contribute to means/SD. Pending or failed
jobs remain in the run-level CSV and are never treated as zero F1.
"""
import argparse
import csv
from pathlib import Path
import statistics
import sys

# The original four experiments; never include LA-OCR or PP-StructureV3.
BASELINES = ("qwen35_vlm", "minicpm_vlm", "qwen35_ocr", "qwen25_ocr")
METRICS = ("official_kie_f1", "mean_latency_s", "mean_ocr_time_s",
           "mean_lm_time_s")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def measure(values):
    return (statistics.mean(values) if values else "",
            statistics.stdev(values) if len(values) >= 2 else "")


def aggregate(series):
    series = Path(series).expanduser().resolve()
    manifest = series / "series_manifest.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing submission manifest: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as f:
        submitted = list(csv.DictReader(f, delimiter="\t"))
    if not submitted:
        raise ValueError("Submission manifest contains no repetitions")

    recorded = []
    expected_document_hash = None
    seen_repeats = set()
    for entry in submitted:
        rep_index = int(entry["repeat_index"])
        if rep_index in seen_repeats:
            raise ValueError(f"Duplicate repetition index: {rep_index}")
        seen_repeats.add(rep_index)
        batch_id = entry["batch_id"]
        root = Path(entry["output_root"]).expanduser().resolve()
        batchdir = root / "batches" / batch_id
        batch_config = batchdir / "batch.json"
        if not batch_config.is_file():
            raise FileNotFoundError(f"Missing batch metadata: {batch_config}")
        import json
        batch = json.loads(batch_config.read_text(encoding="utf-8"))
        if batch["batch_id"] != batch_id:
            raise ValueError(f"Batch identity mismatch: {batch_config}")
        hash_ = batch["manifest_sha256"]
        if expected_document_hash is None:
            expected_document_hash = hash_
        if hash_ != expected_document_hash:
            raise ValueError("Repetitions use different 400-document manifests")

        path = batchdir / "benchmark_summary.csv"
        results = read_csv(path) if path.is_file() else []
        found = set()
        for result in results:
            exp = result.get("experiment_id")
            if exp not in BASELINES:
                raise ValueError(f"Unexpected experiment {exp!r} in {path}")
            if exp in found:
                raise ValueError(f"Duplicate experiment {exp!r} in {path}")
            if result.get("batch_id") != batch_id:
                raise ValueError(f"Batch mismatch in {path}")
            found.add(exp)
            recorded.append({
                "series_id": series.name, "repeat_index": rep_index,
                "git_commit": entry.get("git_commit", ""),
                "manifest_sha256": hash_, **result
            })
        for exp in BASELINES:
            if exp not in found:
                recorded.append({
                    "series_id": series.name, "repeat_index": rep_index,
                    "git_commit": entry.get("git_commit", ""),
                    "manifest_sha256": hash_,
                    "batch_id": batch_id, "experiment_id": exp,
                    "status": "pending" if not path.is_file() else "missing",
                    "completed_documents": 0, "total_documents": 400,
                })

    field_order = ["series_id", "repeat_index", "batch_id", "run_id",
                   "experiment_id", "status", "completed_documents",
                   "official_kie_f1", "mean_latency_s", "mean_ocr_time_s",
                   "mean_lm_time_s", "gpu_name", "git_commit",
                   "manifest_sha256"]
    for row in recorded:
        for name in row:
            if name not in field_order:
                field_order.append(name)
    recorded.sort(key=lambda r: (int(r["repeat_index"]),
                                 BASELINES.index(r["experiment_id"])))
    details_path = series / "repeated_runs.csv"
    write_csv(details_path, recorded, field_order)

    summaries = []
    total = len(submitted)
    for exp in BASELINES:
        rows = [r for r in recorded if r["experiment_id"] == exp]
        successful = [r for r in rows if r.get("status") == "scored"
                      and str(r.get("completed_documents")) == "400"
                      and r.get("official_kie_f1") not in ("", None)]
        devices = sorted({r.get("gpu_name") or "unknown" for r in successful})
        revisions = sorted({r.get("model_revision") or "unknown" for r in successful})
        if len(revisions) > 1:
            raise ValueError(f"{exp} uses inconsistent model revisions: {revisions}")
        result = {"series_id": series.name, "experiment_id": exp,
                  "repeats_requested": total, "repeats_scored": len(successful),
                  "model_revision": revisions[0] if revisions else "",
                  "gpu_names": "; ".join(devices)}
        for metric in METRICS:
            values = [float(r[metric]) for r in successful
                      if r.get(metric) not in ("", None)]
            avg, sd = measure(values)
            result["avg_" + metric] = avg
            result["sd_" + metric] = sd
        summaries.append(result)

    summary_path = series / "repeat_summary.csv"
    write_csv(summary_path, summaries, list(summaries[0]))
    print(f"Wrote {details_path}")
    print(f"Wrote {summary_path}")
    for result in summaries:
        avg = result["avg_official_kie_f1"]
        msg = f"{float(avg):.4f}" if avg != "" else "pending"
        print(f"  {result['experiment_id']}: "
              f"{result['repeats_scored']}/{total} scored, mean F1={msg}")
        if len(result["gpu_names"].split("; ")) > 1:
            print(f"  WARNING: {result['experiment_id']} used multiple GPU names."
                  " Do not combine latencies across hardware without adjustment.")
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("series_dir", type=Path,
                        help="One runs/repeated_baselines/<series-id> directory")
    args = parser.parse_args()
    try:
        aggregate(args.series_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
