"""Offline contract checks: no GPUs, model downloads or SLURM allocations."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ulm_extractor.config import load_config
from ulm_extractor.dataset import load_documents
from ulm_extractor.scheduler import submit_batch
from ulm_extractor.aggregate import collect
from ulm_extractor.storage import read_json, write_json

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.images = self.base / "OCRBench_v2" / "EN_part"
        self.images.mkdir(parents=True)
        self.original = []
        for i in range(400):
            group = (["FUNSD"] * 48 + ["POIE"] * 177 + ["Receipt"] * 25 + ["SROIE"] * 150)[i]
            relative = f"OCRBench_v2/EN_part/{group}/{i:04d}.jpg"
            file = self.images / group / f"{i:04d}.jpg"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"test image")
            self.original.append({
                "id": i, "type": "key information extraction en",
                "dataset_name": group, "image_path": relative,
                "question": f"Extract field {i} as a dictionary.",
                "answers": ['{"field": ["value"]}'],
            })
        write_json(self.images.parent / "OCRBench_v2.json", self.original)
        self.yml = self.base / "model.yml"
        self.yml.write_text(
            "version: 1\ninference:\n  max_new_tokens: 384\nmodels:\n"
            "  - id: qwen35_direct\n    pipeline: vlm\n    adapter: qwen35\n"
            "    checkpoint: Qwen/Qwen3.5-9B\n"
            "  - id: qwen35_ocr\n    pipeline: ocr_lm\n    adapter: qwen35\n"
            "    checkpoint: Qwen/Qwen3.5-9B\n    ocr: paddleocr\n",
            encoding="utf-8"
        )
        self.evaluator = self.base / "eval.py"
        self.evaluator.write_text("# fake evaluator placeholder\n")

    def scheduler_args(self, dry=False):
        return argparse.Namespace(
            output_root=self.base / "runs", evaluator=self.evaluator,
            gpu_family="A100", partition="gpu-single", gres="gpu:A100:1",
            collector_partition="cpu-single", cpus=4, mem="64G",
            time="02:00:00", dry_run=dry
        )

    def test_model_config_and_pins(self):
        models = load_config(self.yml)
        self.assertEqual([m["id"] for m in models], ["qwen35_direct", "qwen35_ocr"])
        self.assertEqual(models[0]["max_new_tokens"], 384)
        self.assertEqual(len(models[0]["revision"]), 40)
        self.assertEqual(models[1]["ocr"], "paddleocr")

    def test_reject_unknown_adapter(self):
        body = self.yml.read_text().replace("adapter: qwen35", "adapter: not_real")
        self.yml.write_text(body)
        with self.assertRaises(ValueError):
            load_config(self.yml)

    def test_400_annotations_and_images(self):
        data = load_documents(self.images)
        self.assertEqual(len(data["documents"]), 400)
        self.assertEqual(data["documents"][0]["question"], self.original[0]["question"])
        self.assertEqual(len(data["manifest_sha256"]), 64)
        (self.images / "FUNSD" / "0000.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            load_documents(self.images)

    def test_submit_dependencies_and_unique_uuids(self):
        data = load_documents(self.images)
        models = load_config(self.yml)
        submitted = []
        def fake_sbatch(command, dry_run):
            submitted.append(command)
            return str(1000 + len(submitted))
        with patch("ulm_extractor.scheduler._sbatch", side_effect=fake_sbatch):
            batch = submit_batch(data, models, self.scheduler_args())
        self.assertEqual(len(submitted), 4)  # OCR, VLM, OCR->LM, collector
        self.assertIn("afterany:1001", submitted[2])
        self.assertIn("afterany:1001:1002:1003", submitted[3])
        self.assertIn("cpu-single", submitted[3])
        self.assertEqual(len({model["run_id"] for model in batch["runs"]}), 2)
        spec = read_json(batch["batch_path"])
        self.assertEqual(spec["manifest_sha256"], data["manifest_sha256"])
        self.assertEqual(read_json(self.base / "runs" / batch["runs"][0]["run_id"] / "run.json")["status"],
                         "submitted")

    def test_collector_retries_failed_scoring_without_gpu(self):
        import subprocess
        data = load_documents(self.images)
        with patch("ulm_extractor.scheduler._sbatch", side_effect=["201", "202", "203", "204"]):
            submission = submit_batch(data, load_config(self.yml), self.scheduler_args())
        batch = read_json(submission["batch_path"])
        experiment = batch["runs"][0]
        run_dir = self.base / "runs" / experiment["run_id"]
        status_path = run_dir / "run.json"
        status = read_json(status_path)
        status["status"] = "scoring_failed"
        write_json(status_path, status)
        predictions = [{**record, "predict": '{"field": "value"}',
                        "inference_seconds": 1.0, "lm_seconds": 1.0,
                        "ocr_seconds": None, "output_tokens": 15,
                        "hit_token_limit": False,
                        "peak_allocated_gpu_bytes": [1024**3]}
                       for record in self.original]
        write_json(run_dir / "predictions.json", predictions)
        other = batch["runs"][1]
        other_status = read_json(self.base / "runs" / other["run_id"] / "run.json")
        other_status["status"] = "failed"
        write_json(self.base / "runs" / other["run_id"] / "run.json", other_status)

        def fake_failure(command, **kwargs):
            raise subprocess.CalledProcessError(
                1, command, output="checking dependencies", stderr="ModuleNotFoundError: No module named 'distance'"
            )
        with patch("ulm_extractor.aggregate.subprocess.run", side_effect=fake_failure):
            collect(batch)
        self.assertIn("ModuleNotFoundError",
                      (run_dir / "scoring.log").read_text(encoding="utf-8"))
        self.assertEqual(read_json(status_path)["status"], "scoring_failed")

        def fake_success(command, **kwargs):
            out = command[command.index("--output_path") + 1]
            source = read_json(command[command.index("--input_path") + 1])
            write_json(out, [{**r, "score": 0.75} for r in source])
        with patch("ulm_extractor.aggregate.subprocess.run", side_effect=fake_success):
            csvpath = collect(batch)
        import pandas as pd
        row = pd.read_csv(csvpath).iloc[0]
        self.assertEqual(row["status"], "scored")
        self.assertAlmostEqual(row["official_kie_f1"], 0.75)
        self.assertEqual(read_json(status_path)["status"], "scored")
        self.assertEqual(len(read_json(run_dir / "predictions.json")), 400)

    def test_dry_run_does_not_write_or_submit(self):
        data = load_documents(self.images)
        with patch("ulm_extractor.scheduler.subprocess.run", side_effect=AssertionError("sbatch called")):
            result = submit_batch(data, load_config(self.yml), self.scheduler_args(True))
        self.assertFalse((self.base / "runs").exists())
        self.assertEqual(result["jobs"]["collector"], "DRY_RUN")

    def test_collect_one_row_per_run(self):
        data = load_documents(self.images)
        with patch("ulm_extractor.scheduler._sbatch", side_effect=["123", "124", "125", "126"]):
            submission = submit_batch(data, load_config(self.yml), self.scheduler_args())
        batch = read_json(submission["batch_path"])
        for cfg in batch["runs"]:
            path = self.base / "runs" / cfg["run_id"]
            run = read_json(path / "run.json")
            run["status"] = "completed"
            write_json(path / "run.json", run)
            preds = [{**r, "predict": '{"field": "value"}', "inference_seconds": 1.0,
                      "lm_seconds": 1.0, "ocr_seconds": None if cfg["pipeline"] == "vlm" else 0.2,
                      "output_tokens": 10, "hit_token_limit": False,
                      "peak_allocated_gpu_bytes": [1024**3]} for r in self.original]
            write_json(path / "predictions.json", preds)
            write_json(path / "metadata.json", {"gpu_names": ["NVIDIA A100"],
                                                "model_load_time_s": 2.5})

        def score_fake(command, **kwargs):
            scored_path = command[command.index("--output_path") + 1]
            source = command[command.index("--input_path") + 1]
            rows = read_json(source)
            write_json(scored_path, [{**row, "score": 1.0} for row in rows])

        with patch("ulm_extractor.aggregate.subprocess.run", side_effect=score_fake):
            output = collect(batch)
        import pandas as pd
        table = pd.read_csv(output)
        self.assertEqual(len(table), 2)
        self.assertEqual(set(table["run_id"]), {m["run_id"] for m in batch["runs"]})
        self.assertEqual(list(table["official_kie_f1"]), [1.0, 1.0])
        self.assertTrue((self.base / "runs" / "benchmark_summary.csv").is_file())

if __name__ == "__main__":
    unittest.main()
