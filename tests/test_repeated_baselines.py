"""Offline tests for four-model repeated submissions and aggregation."""
import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from ulm_extractor.config import load_config
from ulm_extractor.storage import write_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_repeated_baselines.py"
spec = importlib.util.spec_from_file_location("repeat_aggregation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RepeatBaselineTests(unittest.TestCase):
    def test_only_original_four_models(self):
        models = load_config(ROOT / "model-baselines.yml")
        self.assertEqual([x["id"] for x in models], list(module.BASELINES))
        self.assertEqual([x["pipeline"] for x in models],
                         ["vlm", "vlm", "ocr_lm", "ocr_lm"])
        self.assertEqual([x["ocr"] for x in models],
                         [None, None, "paddleocr", "paddleocr"])
        self.assertTrue(all(x["max_new_tokens"] == 384 for x in models))
        self.assertTrue(all(x["do_sample"] is False for x in models))

    def make_series(self, base, scores=(0.60, 0.65)):
        series = base / "series"
        series.mkdir()
        with (series / "series_manifest.tsv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=[
                "repeat_index", "batch_id", "output_root", "collector_job_id", "git_commit"])
            writer.writeheader()
            for rep, score in enumerate(scores, 1):
                root = series / f"repeat-{rep:03d}"
                batchid = f"batch-{rep}"
                folder = root / "batches" / batchid
                folder.mkdir(parents=True)
                write_json(folder / "batch.json", {
                    "batch_id": batchid, "manifest_sha256": "a" * 64
                })
                with (folder / "benchmark_summary.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as result:
                    fields = ["batch_id", "experiment_id", "status",
                              "completed_documents", "official_kie_f1",
                              "mean_latency_s", "mean_ocr_time_s",
                              "mean_lm_time_s", "model_revision", "gpu_name"]
                    w = csv.DictWriter(result, fieldnames=fields)
                    w.writeheader()
                    for exp in module.BASELINES:
                        w.writerow({
                            "batch_id": batchid, "experiment_id": exp,
                            "status": "scored", "completed_documents": 400,
                            "official_kie_f1": score,
                            "mean_latency_s": 4 + rep,
                            "mean_ocr_time_s": 0.2, "mean_lm_time_s": 3 + rep,
                            "model_revision": "b" * 40, "gpu_name": "NVIDIA A100",
                        })
                writer.writerow({
                    "repeat_index": rep, "batch_id": batchid,
                    "output_root": str(root), "collector_job_id": f"100{rep}",
                    "git_commit": "c" * 40
                })
        return series

    def test_aggregate_correct_means_and_no_old_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            series = self.make_series(Path(tmp))
            module.aggregate(series)
            with (series / "repeat_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual(int(rows[0]["repeats_scored"]), 2)
            self.assertAlmostEqual(float(rows[0]["avg_official_kie_f1"]), 0.625)
            self.assertAlmostEqual(float(rows[0]["avg_mean_latency_s"]), 5.5)
            with (series / "repeated_runs.csv").open() as f:
                individual = list(csv.DictReader(f))
            self.assertEqual(len(individual), 8)
            self.assertEqual({x["repeat_index"] for x in individual}, {"1", "2"})
            self.assertEqual({x["experiment_id"] for x in individual},
                             set(module.BASELINES))

    def test_pending_jobs_are_never_counted_as_zero_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            series = self.make_series(Path(tmp))
            (series / "repeat-002" / "batches" / "batch-2" / "benchmark_summary.csv").unlink()
            module.aggregate(series)
            with (series / "repeat_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(int(rows[0]["repeats_scored"]), 1)
            self.assertAlmostEqual(float(rows[0]["avg_official_kie_f1"]), 0.6)
            with (series / "repeated_runs.csv").open() as f:
                individual = list(csv.DictReader(f))
            self.assertEqual(sum(r["status"] == "pending" for r in individual), 4)

    def test_mismatched_documents_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            series = self.make_series(Path(tmp))
            path = series / "repeat-002" / "batches" / "batch-2" / "batch.json"
            write_json(path, {"batch_id": "batch-2", "manifest_sha256": "z" * 64})
            with self.assertRaisesRegex(ValueError, "different 400-document"):
                module.aggregate(series)

    def test_bash_dry_run_submits_nothing_and_validates_n(self):
        """Runs a stub CLI, not real Slurm, and checks exactly N calls."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts" / "run_repeated_baselines.sh",
                            root / "scripts" / "run_repeated_baselines.sh")
            (root / "model-baselines.yml").write_text("version: 1\n")
            bin_dir = root / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "python").symlink_to(Path(sys.executable))
            extractor = bin_dir / "extractor"
            extractor.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\"\n")
            extractor.chmod(0o755)
            input_dir = root / "OCRBench_v2" / "EN_part"
            input_dir.mkdir(parents=True)
            environment = dict(os.environ, EN_PART=str(input_dir))
            command = ["bash", str(root / "scripts" / "run_repeated_baselines.sh")]
            run = subprocess.run(command + ["3", "--dry-run"],
                                 cwd=root, env=environment, capture_output=True,
                                 text=True, check=True)
            self.assertEqual(run.stdout.count("===== Repetition"), 3)
            self.assertEqual(run.stdout.count("--dry-run"), 3)
            self.assertIn("model-baselines.yml", run.stdout)
            self.assertNotIn("model-laocr.yml", run.stdout)
            self.assertNotIn("model-ppstructure.yml", run.stdout)
            self.assertFalse((root / "runs").exists())
            invalid = subprocess.run(command + ["0"], cwd=root, env=environment,
                                     capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("must be an integer", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
