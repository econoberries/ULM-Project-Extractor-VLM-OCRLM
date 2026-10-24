"""Resume one immutable SLURM batch, preserving model UUIDs and saved predictions.

Use --ocr-only to guarantee NO direct VLM GPU jobs are submitted.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from .scheduler import ROOT, _sbatch
from .storage import key_for, read_json, write_json


def _completed_predictions(path, documents):
    """Require exactly the original 400 valid outputs before skipping inference."""
    if not path.is_file():
        return False
    predictions = read_json(path)
    if not isinstance(predictions, list) or len(predictions) != len(documents):
        return False
    expected = {key_for(item) for item in documents}
    keys = [key_for(item) for item in predictions]
    if len(set(keys)) != len(documents) or set(keys) != expected:
        raise ValueError(f"Predictions do not match batch manifest: {path}")
    return all(
        isinstance(item.get("predict"), str)
        and not item.get("error")
        and isinstance(item.get("inference_seconds"), (float, int))
        for item in predictions
    )


def _ocr_cache_complete(batch, documents):
    index_path = (
        Path(batch["output_root"]) / "batches" / batch["batch_id"]
        / "ocr" / "paddleocr" / "index.json"
    )
    if not index_path.is_file():
        return False
    index = read_json(index_path)
    if index.get("manifest_sha256") != batch["manifest_sha256"]:
        raise ValueError("Existing PaddleOCR cache has a different dataset manifest")
    try:
        installed_version = version("paddleocr")
    except PackageNotFoundError as exc:
        raise ValueError("Install PaddleOCR in the project environment before resuming") from exc
    if index.get("paddleocr_version") != installed_version:
        raise ValueError("Existing PaddleOCR cache uses a different PaddleOCR version")
    records = index.get("records")
    expected = {key_for(item) for item in documents}
    if not isinstance(records, dict) or set(records) - expected:
        raise ValueError("Existing PaddleOCR cache has invalid document keys")
    return len(records) == len(documents) and all(
        key in records and Path(records[key]["text_path"]).is_file()
        for key in expected
    )


def resume_batch(batch_path, args):
    """Submit only unfinished experiments; never generate a new batch/run UUID."""
    batch_path = Path(batch_path).expanduser().resolve()
    batch = read_json(batch_path)
    outroot = Path(batch["output_root"]).resolve()
    batchdir = outroot / "batches" / batch["batch_id"]
    if batch_path != (batchdir / "batch.json").resolve():
        raise ValueError("--batch must point to the original batch.json in its output directory")
    documents = read_json(batch["documents_file"])
    if not isinstance(documents, list) or len(documents) != 400:
        raise ValueError("Original 400-document batch manifest is missing or incomplete")
    document_keys = [key_for(row) for row in documents]
    if len(set(document_keys)) != 400:
        raise ValueError("Batch manifest contains duplicate document keys")
    if not Path(batch["evaluator"]).is_file():
        raise FileNotFoundError(f"Official OCRBench evaluator missing: {batch['evaluator']}")

    pending = []
    skipped = []
    for spec in batch["runs"]:
        run_id = spec["run_id"]
        state_path = outroot / run_id / "run.json"
        state = read_json(state_path)
        if (state.get("run_id") != run_id
                or state.get("batch_id") != batch["batch_id"]
                or state.get("config") != spec
                or state.get("manifest_sha256") != batch["manifest_sha256"]):
            raise ValueError(f"Run metadata does not match original batch: {run_id}")

        # Strong explicit protection: don't schedule ANY direct VLM under --ocr-only.
        if args.ocr_only and spec["pipeline"] == "vlm":
            skipped.append((spec["id"], "VLM excluded (--ocr-only)"))
            continue
        predictions_path = outroot / run_id / "predictions.json"
        all_done = _completed_predictions(predictions_path, documents)
        status = state.get("status")
        if status in {"scored", "scoring_failed", "completed"}:
            if not all_done:
                raise ValueError(
                    f"{spec['id']} is marked {status} but lacks 400 valid predictions"
                )
            skipped.append((spec["id"], "already inferred; CPU scoring only"))
            continue
        if status in {"submitted", "running"}:
            raise ValueError(
                f"{spec['id']} is {status}; check squeue/sacct and wait for its "
                "earlier SLURM job to finish before resuming"
            )
        if status != "failed":
            raise ValueError(f"Unrecognized or missing run status for {spec['id']}: {status}")
        pending.append(spec)

    for experiment, why in skipped:
        print(f"SKIP: {experiment} — {why}", flush=True)
    print(f"Resume batch {batch['batch_id']}: {len(pending)} GPU model jobs selected", flush=True)

    # A complete cache can be reused. Otherwise the existing OCR worker resumes
    # from its per-document text index, and OCR-LM workers depend on that job.
    ocr_needed = any(s["pipeline"] == "ocr_lm" for s in pending)
    if ocr_needed and _ocr_cache_complete(batch, documents):
        print("Shared PaddleOCR cache is complete; no OCR GPU job needed", flush=True)
        ocr_needed = False

    gpu_base = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--gres", args.gres, "--cpus-per-task", str(args.cpus),
        "--mem", args.mem, "--time", args.time, "--nodes", "1",
        "--export", f"ALL,EXTRACTOR_EXPECT_GPU={batch['gpu_family']}",
        "--chdir", str(ROOT),
    ]
    logpath = str(batchdir / "logs" / "%x-%j.out")
    jobs = {}
    batcharg = str(batch_path)
    if ocr_needed:
        jobs["paddleocr"] = _sbatch([
            *gpu_base, "--job-name", "ulm-ocr-resume", "--output", logpath,
            str(ROOT / "slurm" / "worker.sbatch"), "ocr", "--batch", batcharg
        ], args.dry_run)
    for spec in pending:
        command = [
            *gpu_base, "--job-name", f"ulm-{spec['id'][:30]}-resume",
            "--output", logpath,
        ]
        if spec["pipeline"] == "ocr_lm" and ocr_needed:
            command += ["--dependency", f"afterany:{jobs['paddleocr']}"]
        command += [
            str(ROOT / "slurm" / "worker.sbatch"),
            "run", "--batch", batcharg, "--run-id", spec["run_id"],
        ]
        jobs[spec["run_id"]] = _sbatch(command, args.dry_run)

    # Run a CPU collector after all resubmitted jobs, retaining scores for
    # previously completed models. Original batch and jobs.json stay unchanged.
    deps = [value for value in jobs.values() if value != "DRY_RUN"]
    command = [
        "sbatch", "--parsable", "--partition", args.collector_partition,
        "--cpus-per-task", "1", "--mem", "4G", "--time", "00:30:00",
        "--chdir", str(ROOT), "--job-name", "ulm-collect-resume",
        "--output", logpath,
    ]
    if deps:
        command += ["--dependency", "afterany:" + ":".join(deps)]
    command += [str(ROOT / "slurm" / "collector.sbatch"), "collect", "--batch", batcharg]
    jobs["collector"] = _sbatch(command, args.dry_run)
    if not args.dry_run:
        # Audit history of resubmissions; never overwrite jobs.json or batch.json.
        resume_id = str(uuid4())
        write_json(batchdir / "resumes" / f"{resume_id}.json", {
            "batch_id": batch["batch_id"], "resumed_runs": [s["run_id"] for s in pending],
            "skipped": skipped, "jobs": jobs,
        })
    print(f"Batch summary: {batchdir / 'benchmark_summary.csv'}")
    return {"batch_id": batch["batch_id"], "jobs": jobs, "resumed": pending, "skipped": skipped}
