"""Offline contracts for the full 400-image PP-StructureV3 -> Markdown -> Qwen mode."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ulm_extractor.config import load_config
from ulm_extractor.ppstructure import FORMAT, SETTINGS, RECOVERY, FALLBACK_SETTINGS, extract_markdown, run_ppstructure
from ulm_extractor.resume import _ocr_cache_complete
from ulm_extractor.scheduler import submit_batch
from ulm_extractor.storage import key_for, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


class PPStructureContracts(unittest.TestCase):
    def test_config_registers_six_models_and_standalone_full_run(self):
        full = load_config(ROOT / "model.yml")
        self.assertEqual(len(full), 6)
        only = load_config(ROOT / "model-ppstructure.yml")
        self.assertEqual(len(only), 1)
        self.assertEqual(only[0]["pipeline"], "ppstructure_lm")
        self.assertEqual(only[0]["ocr"], "ppstructurev3")
        self.assertEqual(only[0]["checkpoint"], "Qwen/Qwen3.5-9B")

    def test_markdown_extraction_checks_api_and_nonempty_content(self):
        self.assertEqual(extract_markdown(SimpleNamespace(markdown={
            "markdown_texts": "| Key | Value |\n|---|---|\n| Name | Alice |",
        })), "| Key | Value |\n|---|---|\n| Name | Alice |")
        with self.assertRaises(ValueError):
            extract_markdown({"markdown_texts": " "})
        with self.assertRaises(TypeError):
            extract_markdown({"markdown_texts": ["invalid", "list"]})
        with self.assertRaises(TypeError):
            extract_markdown({})

    def test_parser_writes_per_document_markdown_and_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            records = [{
                "id": i, "dataset_name": "POIE",
                "image_path": f"OCRBench_v2/EN_part/POIE/{i}.jpg"
            } for i in range(2)]
            docs = base / "documents.json"
            write_json(docs, records)
            batch = {
                "batch_id": "pp-batch",
                "documents_file": str(docs),
                "output_root": str(base / "runs"),
                "image_directory": str(base / "EN_part"),
                "manifest_sha256": "x" * 64,
            }
            class FakeResult:
                markdown = {
                    "markdown_texts": "| Field | Value |\\n|---|---|\\n| Name | Alice |",
                    "markdown_images": {},
                }
                json = {"res": {"parsing_res_list": [], "overall_ocr_res": {
                    "rec_texts": ["Name", "Alice"],
                    "rec_boxes": [[1, 1, 10, 10], [20, 1, 40, 10]],
                }}}
                def save_to_json(self, save_path):
                    Path(save_path).write_text(json.dumps(self.json))
            class FakeParser:
                count = 0
                def predict(self, path):
                    self.count += 1
                    return [FakeResult()]
            class FakeOCR:
                def predict(self, path):
                    return [{"res": {"rec_texts": ["Name", "Alice", "Invoice total 10"],
                             "rec_boxes": [[1, 1, 10, 10], [20, 1, 40, 10],
                                           [1, 30, 50, 40]]}}]
            fake = FakeParser()
            with (patch.dict(sys.modules, {
                      "paddleocr": SimpleNamespace(
                          PPStructureV3=lambda **kw: fake,
                          PaddleOCR=lambda **kw: FakeOCR())
                  }),
                  patch("ulm_extractor.ppstructure.version",
                        side_effect=lambda name: {
                            "paddleocr": "3.7.0", "paddlex": "3.7.2"}[name]),
                  patch("ulm_extractor.ppstructure.synchronize_paddle")):
                run_ppstructure(batch)
                run_ppstructure(batch)
            self.assertEqual(fake.count, 2)  # the second call reuses cached Markdown
            folder = base / "runs" / "batches" / "pp-batch" / "ocr" / "ppstructurev3"
            index = read_json(folder / "index.json")
            self.assertEqual(index["format"], FORMAT)
            self.assertEqual(index["settings"], SETTINGS)
            self.assertEqual(len(index["records"]), 2)
            for record in records:
                data = index["records"][key_for(record)]
                markdown = Path(data["text_path"])
                self.assertEqual(markdown.suffix, ".md")
                self.assertIn("| Name | Alice |", markdown.read_text())
                self.assertIn("Invoice total 10", markdown.read_text())
                self.assertIn("Additional OCR-recognized text", markdown.read_text())
                self.assertTrue(Path(data["raw_json_paths"][0]).is_file())
                self.assertTrue(Path(data["original_markdown_path"]).is_file())
                self.assertTrue(Path(data["audit_path"]).is_file())
                self.assertIsInstance(data["ocr_seconds"], float)

    def test_resume_rejects_mismatched_ppstructure_cache_config(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            record = {"id": 10, "dataset_name": "POIE", "image_path": "EN_part/10.jpg"}
            batch = {"batch_id": "pp-batch", "output_root": str(base / "runs"),
                     "manifest_sha256": "z" * 64}
            root = base / "runs" / "batches" / "pp-batch" / "ocr" / "ppstructurev3"
            md = root / "markdown" / (key_for(record) + ".md")
            md.parent.mkdir(parents=True)
            md.write_text("hello")
            index = {"format": FORMAT, "settings": SETTINGS,
                     "manifest_sha256": batch["manifest_sha256"],
                     "paddleocr_version": "3.7.0", "paddlex_version": "3.7.2",
                     "recovery": RECOVERY, "fallback_settings": FALLBACK_SETTINGS,
                     "records": {key_for(record): {
                         "text_path": str(md), "ocr_seconds": 1.0}}}
            write_json(root / "index.json", index)
            interpreter = base / "repo" / ".venv-ppstructure" / "bin" / "python"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            with (patch("ulm_extractor.resume.ROOT", base / "repo"),
                  patch("ulm_extractor.resume.subprocess.run",
                        side_effect=[Mock(stdout="3.7.0\n"), Mock(stdout="3.7.2\n")])):
                self.assertTrue(_ocr_cache_complete(batch, [record], "ppstructurev3"))
            index["settings"] = {"lang": "other"}
            write_json(root / "index.json", index)
            with self.assertRaisesRegex(ValueError, "format/settings mismatch"):
                _ocr_cache_complete(batch, [record], "ppstructurev3")

    def test_scheduler_orders_parser_model_and_collector_for_all_400(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            fake_eval = base / "eval.py"
            fake_eval.write_text("# placeholder")
            args = argparse.Namespace(
                output_root=base / "runs", evaluator=fake_eval,
                gpu_family="A100", partition="gpu-single", gres="gpu:A100:1",
                collector_partition="cpu-single", cpus=4, mem="64G",
                time="04:00:00", dry_run=True)
            docs = [{
                "id": i, "dataset_name": "POIE",
                "image_path": f"EN_part/img{i}.jpg"
            } for i in range(400)]
            data = {"documents": docs,
                    "image_directory": str(base),
                    "annotations": str(base / "OCRBench_v2.json"),
                    "annotation_sha256": "a" * 64,
                    "manifest_sha256": "b" * 64}
            calls = []
            def fake_sbatch(command, dry):
                calls.append(command)
                return str(7000 + len(calls))
            with patch("ulm_extractor.scheduler._sbatch", side_effect=fake_sbatch):
                submit_batch(data, load_config(ROOT / "model-ppstructure.yml"), args)
            self.assertEqual(len(calls), 3)
            self.assertIn("ppstructure", calls[0])
            self.assertIn("afterany:7001", calls[1])
            self.assertIn("afterany:7001:7002", calls[2])
            self.assertIn("04:00:00", calls[0])
            self.assertFalse((base / "runs").exists())

    def test_worker_environment_and_install_extras(self):
        worker = (ROOT / "slurm" / "worker.sbatch").read_text()
        self.assertIn('ppstructure) python_bin=".venv-ppstructure/bin/python"', worker)
        self.assertIn('run) python_bin=".venv/bin/python"', worker)
        smoke = (ROOT / "slurm" / "check_environment.sbatch").read_text()
        self.assertIn('pp-paddle|ppstructure)', smoke)
        self.assertIn('scripts/check_ppstructure.py', smoke)
        self.assertIn('paddleocr[doc-parser]==3.7.0', (ROOT / "pyproject.toml").read_text())
        self.assertIn("/.venv-ppstructure/", (ROOT / ".gitignore").read_text())


if __name__ == "__main__":
    unittest.main()
