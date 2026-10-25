"""SLURM entry point. GPU inference and OCR are isolated from CPU scoring."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import sys

from .storage import read_json, write_json, key_for

def run_model(batch, run_id):
    from .adapters import InferenceAdapter
    from .dataset import image_for
    specs = [m for m in batch["runs"] if m["run_id"] == run_id]
    if len(specs) != 1:
        raise ValueError(f"Unknown run_id {run_id}")
    spec = specs[0]
    run_dir = Path(batch["output_root"]) / run_id
    status_path = run_dir / "run.json"
    prediction_path = run_dir / "predictions.json"
    records = read_json(batch["documents_file"])
    model = InferenceAdapter(spec)
    current = read_json(status_path)
    if current["config"] != spec or current["manifest_sha256"] != batch["manifest_sha256"]:
        raise ValueError("Cannot resume a run with different model settings or benchmark manifest")
    current.update(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                   slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    current.pop("error", None)  # do not retain old initialization failures
    write_json(status_path, current)
    existing = {}
    if prediction_path.exists():
        saved = read_json(prediction_path)
        if not isinstance(saved, list):
            raise ValueError("Existing predictions must be a list")
        for row in saved:
            key = key_for(row)
            if key in existing:
                raise ValueError(f"Duplicate saved prediction: {key}")
            existing[key] = row
        expected = {key_for(row) for row in records}
        if set(existing) - expected:
            raise ValueError("Saved predictions include other benchmark records")
    to_run = [record for record in records
              if key_for(record) not in existing or existing[key_for(record)].get("error")]
    try:
        ocr_index = None
        if spec.get("ocr") is not None:
            engine = spec["ocr"]
            cache_path = Path(batch["output_root"]) / "batches" / batch["batch_id"] / "ocr" / engine / "index.json"
            cache = read_json(cache_path)
            if cache.get("manifest_sha256") != batch["manifest_sha256"]:
                raise ValueError("OCR cache does not match the benchmark manifest")
            if engine == "paddleocr_layout" and cache.get("format") != "paddleocr-layout-v1":
                raise ValueError("Layout OCR cache format mismatch")
            if engine == "ppstructurev3":
                from .ppstructure import FORMAT, SETTINGS, RECOVERY, FALLBACK_SETTINGS
                if cache.get("format") != FORMAT or cache.get("settings") != SETTINGS or
                    cache.get("recovery") != RECOVERY or
                    cache.get("fallback_settings") != FALLBACK_SETTINGS:
                    raise ValueError("PP-StructureV3 Markdown cache configuration mismatch")
            ocr_index = cache["records"]
            if len(ocr_index) != 400 or any(key_for(r) not in ocr_index for r in records):
                raise ValueError(f"Shared {engine} cache is incomplete; not loading the language model")
        if to_run:
            model.load()
            environment = model.environment()
            write_json(run_dir / "metadata.json", {
                "run_id": run_id, "experiment_id": spec["id"], "config": spec,
                "manifest_sha256": batch["manifest_sha256"],
                "annotation_sha256": batch["annotation_sha256"],
                "started_utc": current["started_utc"],
                "slurm_job_id": current["slurm_job_id"], **environment
            })
        else:
            print(f"Already have {len(existing)} predictions; skipping model loading")
        for position, record in enumerate(records, 1):
            key = key_for(record)
            if key in existing and not existing[key].get("error"):
                continue
            entry = dict(record)
            entry["document_key"] = key
            try:
                if spec["pipeline"] == "vlm":
                    outcome = model.infer(record["question"],
                                          image_path=image_for(record, batch["image_directory"]))
                    entry["ocr_seconds"] = None
                else:
                    ocr_info = ocr_index[key]
                    content = Path(ocr_info["text_path"]).read_text(encoding="utf-8")
                    outcome = model.infer(record["question"], ocr_text=content)
                    entry["ocr_seconds"] = float(ocr_info["ocr_seconds"])
                entry.update(outcome)
                entry["inference_seconds"] = round(
                    outcome["lm_seconds"] + (entry["ocr_seconds"] or 0), 4
                )
                entry.pop("error", None)
                msg = "OK"
            except Exception as exc:
                entry.update(predict="", error=f"{type(exc).__name__}: {exc}",
                             inference_seconds=None, lm_seconds=None)
                msg = "ERROR"
            existing[key] = entry
            write_json(prediction_path, [existing[key_for(row)] for row in records
                                         if key_for(row) in existing])
            print(f"[{position}/400] {msg} {spec['id']} {key}", flush=True)
        failures = [row for row in existing.values() if row.get("error")]
        current.update(
            status="completed" if len(existing) == 400 and not failures else "failed",
            completed_documents=len(existing) - len(failures),
            failures=len(failures),
            finished_utc=datetime.now(timezone.utc).isoformat()
        )
        write_json(status_path, current)
        if failures:
            raise RuntimeError(f"{len(failures)} extraction failures: check predictions.json")
    except Exception as exc:
        current.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, current)
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ulm_extractor.worker")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("ocr", "ocr-layout", "ppstructure", "run", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--batch", type=Path, required=True)
        if name == "run":
            p.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    batch = read_json(args.batch)
    if args.action in {"ocr", "ocr-layout"}:
        from .ocr import run_ocr
        run_ocr(batch, engine="paddleocr_layout" if args.action == "ocr-layout" else "paddleocr")
    elif args.action == "ppstructure":
        from .ppstructure import run_ppstructure
        run_ppstructure(batch)
    elif args.action == "run":
        run_model(batch, args.run_id)
    else:
        from .aggregate import collect
        collect(batch)

if __name__ == "__main__":
    sys.exit(main())
