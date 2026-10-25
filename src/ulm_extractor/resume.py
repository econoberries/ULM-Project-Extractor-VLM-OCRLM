"""Resume one immutable SLURM batch, preserving model UUIDs and saved predictions.

Use --ocr-only to guarantee NO direct VLM GPU jobs are submitted.
"""
from pathlib import Path
import subprocess
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


def _ocr_cache_complete(batch, documents, engine="paddleocr"):
    index_path = (
        Path(batch["output_root"]) / "batches" / batch["batch_id"]
        / "ocr" / engine / "index.json"
    )
    if not index_path.is_file():
        return False
    index = read_json(index_path)
    if index.get("manifest_sha256") != batch["manifest_sha256"]:
        raise ValueError("Existing PaddleOCR cache has a different dataset manifest")
    if engine == "paddleocr_layout" and index.get("format") != "paddleocr-layout-v1":
        raise ValueError("Existing layout OCR cache has an incompatible format")
    if engine == "ppstructurev3":
        from .ppstructure import FORMAT, SETTINGS
        if index.get("format") != FORMAT or index.get("settings") != SETTINGS:
            raise ValueError("Existing PP-StructureV3 cache format/settings mismatch")
    # The resume CLI runs from .venv (PyTorch), but the OCR model runs
    # exclusively from .venv-ocr. Check the *correct* PaddleOCR installation.
    env_dir = (".venv-ppstructure" if engine == "ppstructurev3" else
               ".venv-la-ocr" if engine == "paddleocr_layout" else ".venv-ocr")
    ocr_python = ROOT / env_dir / "bin" / "python"
    if not ocr_python.is_file():
        raise ValueError(f"Missing {env_dir}/bin/python; create OCR environment first")
    try:
        proc = subprocess.run([
            str(ocr_python), "-c",
            "from importlib.metadata import version; print(version('paddleocr'))"
        ], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Cannot inspect PaddleOCR in {env_dir}: {exc.stderr}"
        ) from exc
    installed_version = proc.stdout.strip()
    if index.get("paddleocr_version") != installed_version:
        raise ValueError("Existing PaddleOCR cache uses a different PaddleOCR version")
    if engine == "ppstructurev3":
        installed_paddlex = subprocess.run([
            str(ocr_python), "-c",
            "from importlib.metadata import version; print(version('paddlex'))"
        ], capture_output=True, text=True, check=True).stdout.strip()
        if index.get("paddlex_version") != installed_paddlex:
            raise ValueError("Existing PP-StructureV3 cache uses a different PaddleX version")
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

    # Each OCR representation has its own cache and resume dependencies.
    required_engines = {s.get("ocr") for s in pending if s.get("ocr")}
    ocr_needed = set()
    for engine in ("paddleocr", "paddleocr_layout", "ppstructurev3"):
        if engine not in required_engines:
            continue
        if _ocr_cache_complete(batch, documents, engine):
            print(f"Shared {engine} cache complete; no OCR GPU job needed", flush=True)
        else:
            ocr_needed.add(engine)

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
    for engine, action, jobname in (
        ("paddleocr", "ocr", "ulm-ocr-resume"),
        ("paddleocr_layout", "ocr-layout", "ulm-la-ocr-resume"),
        ("ppstructurev3", "ppstructure", "ulm-ppstructure-resume"),
    ):
        if engine in ocr_needed:
            jobs[engine] = _sbatch([
                *gpu_base, "--job-name", jobname, "--output", logpath,
                str(ROOT / "slurm" / "worker.sbatch"), action, "--batch", batcharg
            ], args.dry_run)
    for spec in pending:
        command = [
            *gpu_base, "--job-name", f"ulm-{spec['id'][:30]}-resume",
            "--output", logpath,
        ]
        if spec.get("ocr") in ocr_needed:
            command += ["--dependency", f"afterany:{jobs[spec['ocr']]}"]
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
