"""Offline regression tests for the coordinate-preserving OCR -> LM mode."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ulm_extractor.config import load_config
from ulm_extractor.layout import FORMAT, format_layout_json
from ulm_extractor.resume import _ocr_cache_complete
from ulm_extractor.scheduler import submit_batch
from ulm_extractor.storage import key_for, write_json

ROOT = Path(__file__).resolve().parents[1]


class LayoutPipelineTests(unittest.TestCase):
    def test_layout_json_has_accurate_text_box_score_alignment(self):
        output = {"res": {
            "rec_texts": ["Protein", "8 g"],
            "rec_boxes": [[10, 20, 100, 40], [250, 20, 300, 40]],
            "rec_scores": [0.99, 0.87],
        }}
        encoded = format_layout_json([output], (640, 480))
        result = json.loads(encoded)
        self.assertEqual(result["format"], FORMAT)
        self.assertEqual(result["image_size"], {"width": 640, "height": 480})
        self.assertEqual(result["box_format"], "xyxy")
        self.assertEqual(result["lines"][0], {
            "text": "Protein", "bbox": [10, 20, 100, 40], "confidence": 0.99
        })
        self.assertEqual(result["lines"][1]["bbox"], [250, 20, 300, 40])
        self.assertEqual(result["lines"][1]["confidence"], 0.87)

    def test_layout_json_accepts_json_string_and_rejects_bad_box_alignment(self):
        result = {"res": {"rec_texts": ["Label"], "rec_boxes": [[1, 2, 9, 10]],
                          "rec_scores": [0.9]}}
        self.assertEqual(len(json.loads(format_layout_json([json.dumps(result)], (20, 20)))["lines"]), 1)
        result["res"]["rec_boxes"] = []
        with self.assertRaisesRegex(ValueError, "different lengths"):
            format_layout_json([result], (20, 20))
        result["res"].pop("rec_boxes")
        with self.assertRaisesRegex(ValueError, "requires rec_texts and rec_boxes"):
            format_layout_json([result], (20, 20))

    def test_model_files_and_pipeline_validation(self):
        selected = load_config(ROOT / "model-laocr.yml")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], "qwen35_laocr")
        self.assertEqual(selected[0]["pipeline"], "la_ocr_lm")
        self.assertEqual(selected[0]["ocr"], "paddleocr_layout")
        all_models = load_config(ROOT / "model.yml")
        self.assertEqual(len(all_models), 6)
        self.assertEqual(len({x["id"] for x in all_models}), 6)
        self.assertEqual([m["id"] for m in all_models if m["pipeline"] == "la_ocr_lm"],
                         ["qwen35_laocr"])

    def test_both_ocr_jobs_and_dependencies_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            evaluator = base / "eval.py"
            evaluator.write_text("# placeholder")
            models = [x for x in load_config(ROOT / "model.yml")
                      if x["id"] in {"qwen35_ocr", "qwen35_laocr"}]
            data = {
                "documents": [{"id": i, "dataset_name": "POIE", "image_path": f"{i}.jpg"}
                              for i in range(400)],
                "image_directory": str(base),
                "annotations": str(base / "annotations.json"),
                "annotation_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
            }
            args = argparse.Namespace(
                output_root=base / "runs", evaluator=evaluator,
                gpu_family="A100", partition="gpu-single", gres="gpu:A100:1",
                collector_partition="cpu-single", cpus=4, mem="64G",
                time="02:00:00", dry_run=True,
            )
            commands = []
            def pretend_sbatch(command, dry):
                commands.append(command)
                return str(3000 + len(commands))
            with patch("ulm_extractor.scheduler._sbatch", side_effect=pretend_sbatch):
                submit_batch(data, models, args)
            self.assertEqual(len(commands), 5)
            self.assertIn("ocr", commands[0])
            self.assertIn("ocr-layout", commands[1])
            self.assertIn("afterany:3001", commands[2])
            self.assertIn("afterany:3002", commands[3])
            self.assertIn("afterany:3001:3002:3003:3004", commands[4])
            self.assertFalse((base / "runs").exists())

    def test_layout_cache_uses_only_layout_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            batch = {
                "output_root": str(base / "runs"),
                "batch_id": "batch-1",
                "manifest_sha256": "m" * 64,
            }
            records = [{"id": 1, "dataset_name": "POIE",
                        "image_path": "EN_part/poie/1.jpg"}]
            key = key_for(records[0])
            cache_dir = base / "runs" / "batches" / "batch-1" / "ocr" / "paddleocr_layout"
            text_file = cache_dir / "text" / (key + ".json")
            text_file.parent.mkdir(parents=True)
            text_file.write_text(format_layout_json([{
                "res": {"rec_texts": ["A"], "rec_boxes": [[0, 0, 4, 4]]}
            }], (50, 50)))
            write_json(cache_dir / "index.json", {
                "manifest_sha256": batch["manifest_sha256"],
                "paddleocr_version": "3.7.0",
                "format": FORMAT,
                "records": {key: {"text_path": str(text_file), "ocr_seconds": 1.0}},
            })
            fake_root = base / "repo"
            python = fake_root / ".venv-la-ocr" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            with (patch("ulm_extractor.resume.ROOT", fake_root),
                  patch("ulm_extractor.resume.subprocess.run", return_value=Mock(stdout="3.7.0\n"))):
                self.assertTrue(_ocr_cache_complete(batch, records, "paddleocr_layout"))
                self.assertFalse(_ocr_cache_complete(batch, records, "paddleocr"))
            idx = json.loads((cache_dir / "index.json").read_text())
            idx.pop("format")
            write_json(cache_dir / "index.json", idx)
            with self.assertRaisesRegex(ValueError, "incompatible format"):
                _ocr_cache_complete(batch, records, "paddleocr_layout")

    def test_slurm_and_venv_route_layout_worker(self):
        script = (ROOT / "slurm" / "worker.sbatch").read_text()
        self.assertIn('ocr-layout) python_bin=".venv-la-ocr/bin/python"', script)
        self.assertIn('run) python_bin=".venv/bin/python"', script)
        checker = (ROOT / "scripts" / "check_dependencies.py").read_text()
        self.assertIn('"la-ocr"', checker)
        self.assertIn("la_python", checker)
        self.assertIn("/.venv-la-ocr/", (ROOT / ".gitignore").read_text())


if __name__ == "__main__":
    unittest.main()
