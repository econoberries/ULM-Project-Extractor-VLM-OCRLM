"""Submit isolated SLURM GPU jobs and a dependency-aware CPU collector."""
from pathlib import Path
from uuid import uuid4
import subprocess

from .storage import write_json

ROOT = Path(__file__).resolve().parents[2]

def _sbatch(command, dry_run):
    print("+", " ".join(map(str, command)), flush=True)
    if dry_run:
        return "DRY_RUN"
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    # --parsable returns jobid or jobid;cluster
    job_id = proc.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Cannot parse sbatch job ID: {proc.stdout!r}")
    return job_id

def submit_batch(data, models, args):
    batch_id = str(uuid4())
    outroot = Path(args.output_root).expanduser().resolve()
    batchdir = outroot / "batches" / batch_id
    docspath = batchdir / "documents.json"
    specpath = batchdir / "batch.json"
    runs = [{**model, "run_id": str(uuid4())} for model in models]
    spec = {
        "batch_id": batch_id, "documents_file": str(docspath),
        "image_directory": data["image_directory"],
        "annotations": data["annotations"],
        "annotation_sha256": data["annotation_sha256"],
        "manifest_sha256": data["manifest_sha256"],
        "evaluator": str(Path(args.evaluator).expanduser().resolve()),
        "output_root": str(outroot), "runs": runs,
        "gpu_family": args.gpu_family,
    }
    if not args.dry_run and not Path(spec["evaluator"]).is_file():
        raise FileNotFoundError(
            f"Official OCRBench evaluator missing: {spec['evaluator']}. "
            "Clone Yuliang-Liu/MultimodalOCR or use --evaluator PATH."
        )
    if not args.dry_run:
        batchdir.joinpath("logs").mkdir(parents=True, exist_ok=True)
        write_json(docspath, data["documents"])
        write_json(specpath, spec)
        for model in runs:
            rundir = outroot / model["run_id"]
            write_json(rundir / "run.json", {
                "run_id": model["run_id"], "batch_id": batch_id,
                "experiment_id": model["id"], "config": model,
                "manifest_sha256": spec["manifest_sha256"],
                "status": "submitted",
            })
    gpu_base = [
        "sbatch", "--parsable", "--partition", args.partition,
        "--gres", args.gres, "--cpus-per-task", str(args.cpus),
        "--mem", args.mem, "--time", args.time, "--nodes", "1",
        "--export", f"ALL,EXTRACTOR_EXPECT_GPU={args.gpu_family}",
        "--chdir", str(ROOT),
    ]
    logpath = str(batchdir / "logs" / "%x-%j.out")
    batcharg = str(specpath)
    jobs = {}
    for engine, action, job_name in (
        ("paddleocr", "ocr", "ulm-ocr"),
        ("paddleocr_layout", "ocr-layout", "ulm-la-ocr"),
    ):
        if any(x.get("ocr") == engine for x in runs):
            jobs[engine] = _sbatch([
                *gpu_base, "--job-name", job_name, "--output", logpath,
                str(ROOT / "slurm" / "worker.sbatch"),
                action, "--batch", batcharg,
            ], args.dry_run)
    for model in runs:
        command = [
            *gpu_base, "--job-name", f"ulm-{model['id'][:30]}",
            "--output", logpath,
        ]
        if model.get("ocr") is not None:
            command += ["--dependency", f"afterany:{jobs[model['ocr']]}"]
        command += [
            str(ROOT / "slurm" / "worker.sbatch"),
            "run", "--batch", batcharg, "--run-id", model["run_id"]
        ]
        jobs[model["run_id"]] = _sbatch(command, args.dry_run)
    # Collect after ALL jobs finish, regardless of exit codes. Always CPU-only.
    dependencies = [jid for jid in jobs.values() if jid != "DRY_RUN"]
    collector = [
        "sbatch", "--parsable", "--partition", args.collector_partition,
        "--cpus-per-task", "1", "--mem", "4G", "--time", "00:30:00",
        "--chdir", str(ROOT),
        "--job-name", "ulm-collect", "--output", logpath,
    ]
    if dependencies:
        collector += ["--dependency", "afterany:" + ":".join(dependencies)]
    collector += [str(ROOT / "slurm" / "collector.sbatch"), "collect", "--batch", batcharg]
    jobs["collector"] = _sbatch(collector, args.dry_run)
    if not args.dry_run:
        write_json(batchdir / "jobs.json", jobs)
    print(f"Batch UUID: {batch_id}")
    print(f"Summary: {batchdir / 'benchmark_summary.csv'}")
    return {"batch_id": batch_id, "batch_path": str(specpath), "jobs": jobs, "runs": runs}
