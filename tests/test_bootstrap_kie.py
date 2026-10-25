"""Offline statistical and data-integrity tests for paired OCRBench bootstrap."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ulm_extractor.storage import write_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bootstrap_kie", ROOT / "scripts" / "bootstrap_kie.py"
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def documents():
    groups = [("FUNSD", 48), ("POIE", 177), ("Receipt", 25), ("SROIE", 150)]
    result = []
    for group, count in groups:
        for i in range(count):
            result.append({
                "dataset_name": group, "id": str(len(result)),
                "image_path": f"EN_part/{group}/{i}.jpg",
                "question": f"Extract field {i}",
            })
    return result


def create_fixture(root, ids=("qwen35_vlm", "minicpm_vlm",
                             "qwen35_ocr", "qwen25_ocr"),
                   score_func=None, batch_id="batch-001"):
    batch_dir = root / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    docs = documents()
    write_json(batch_dir / "documents.json", docs)
    runs = []
    for ix, exp in enumerate(ids):
        run_id = f"run-{batch_id}-{ix}"
        run_path = root / run_id
        run_path.mkdir()
        values = [
            score_func(exp, k, row) if score_func else
            (1.0 if ix == 0 else 0.5)
            for k, row in enumerate(docs)
        ]
        write_json(run_path / "predictions_scored.json", [
            {**row, "score": value, "predict": "{}"} for row, value in zip(docs, values)
        ])
        write_json(run_path / "run.json",
                   {"status": "scored", "batch_id": batch_id})
        write_json(run_path / "metrics.json",
                   {"official_kie_f1": sum(values) / len(values)})
        runs.append({
            "id": exp, "run_id": run_id, "checkpoint": exp,
            "revision": "a" * 40, "pipeline": "vlm", "ocr": None,
        })
    write_json(batch_dir / "batch.json", {
        "batch_id": batch_id, "manifest_sha256": "f" * 64,
        "output_root": "/obsolete/home/path", "runs": runs,
    })
    return batch_dir / "batch.json"


class BootstrapTests(unittest.TestCase):
    def test_matched_bootstrap_constant_difference(self):
        docs = documents()
        keys = [bootstrap.identity(r) for r in docs]
        base = np.linspace(0, 0.7, 400)
        scores = {"A": base + 0.3, "B": base}
        result, pairs, aligned = bootstrap.make_results(
            keys, docs, scores, ["A", "B"], draws=500, seed=42
        )
        overall = next(r for r in pairs if r["dataset"] == "ALL")
        self.assertAlmostEqual(overall["mean_difference_a_minus_b"], 0.3)
        self.assertAlmostEqual(overall["ci_low"], 0.3, places=9)
        self.assertAlmostEqual(overall["ci_high"], 0.3, places=9)
        self.assertTrue(overall["familywise_ci_excludes_zero"])
        self.assertEqual(len(aligned), 400)
        self.assertEqual(len([r for r in pairs if r["dataset"] != "ALL"]), 4)
        self.assertEqual(len([r for r in result if r["dataset"] == "ALL"]), 2)

    def test_identical_predictions_show_zero_difference_and_repeatability(self):
        docs = documents()
        keys = [bootstrap.identity(r) for r in docs]
        vec = np.asarray([(i % 10) / 10 for i in range(400)])
        a, b, _ = bootstrap.make_results(keys, docs, {"A": vec, "B": vec},
                                          ["A", "B"], draws=500, seed=4)
        c, d, _ = bootstrap.make_results(keys, docs, {"A": vec, "B": vec},
                                          ["A", "B"], draws=500, seed=4)
        self.assertEqual(a, c)
        self.assertEqual(b, d)
        self.assertEqual(b[0]["ci_low"], 0.0)
        self.assertEqual(b[0]["ci_high"], 0.0)
        self.assertFalse(b[0]["ci_excludes_zero"])

    def test_end_to_end_reads_archive_portably_and_writes_all_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            batch = create_fixture(root)
            out = Path(tmp) / "analysis"
            dest = bootstrap.analyze([batch], output_dir=out, draws=200, seed=2026)
            self.assertEqual(dest, out)
            self.assertTrue((dest / "bootstrap_models.csv").is_file())
            self.assertTrue((dest / "bootstrap_pairs.csv").is_file())
            self.assertTrue((dest / "paired_document_scores.csv").is_file())
            self.assertTrue((dest / "analysis_metadata.json").is_file())
            with (dest / "bootstrap_pairs.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5 * 6)
            # 4 models -> 6 paired comparisons, + 4 subset comparisons each
            value = [row for row in rows if row["dataset"] == "ALL"
                     and row["model_a"] == "qwen35_vlm"
                     and row["model_b"] == "minicpm_vlm"][0]
            self.assertAlmostEqual(float(value["mean_difference_a_minus_b"]), 0.5)
            self.assertTrue((dest / "analysis_metadata.json").is_file())
            with self.assertRaises(FileExistsError):
                bootstrap.analyze([batch], output_dir=out, draws=200)

    def test_order_of_scored_predictions_does_not_change_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            batch = create_fixture(root, score_func=lambda exp, i, row:
                                   float((i % 17) / 17))
            data = json.loads(batch.read_text())
            run = root / data["runs"][0]["run_id"] / "predictions_scored.json"
            scored = json.loads(run.read_text())
            write_json(run, list(reversed(scored)))
            ids, docs, scores, details, _ = bootstrap.collect_models(
                [batch], bootstrap.BASELINES
            )
            self.assertEqual(len(ids), 400)
            self.assertAlmostEqual(scores["qwen35_vlm"][0], 0.0)
            self.assertAlmostEqual(scores["qwen35_vlm"][399], (399 % 17) / 17)

    def test_reject_duplicate_and_missing_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            batch = create_fixture(root)
            data = json.loads(batch.read_text())
            path = root / data["runs"][1]["run_id"] / "predictions_scored.json"
            records = json.loads(path.read_text())
            records[1]["image_path"] = records[0]["image_path"]
            records[1]["id"] = records[0]["id"]
            records[1]["question"] = records[0]["question"]
            write_json(path, records)
            with self.assertRaisesRegex(ValueError, "Duplicate document"):
                bootstrap.collect_models([batch], bootstrap.BASELINES)

    def test_reject_duplicated_model_ids_across_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            a = create_fixture(root, batch_id="first")
            b = create_fixture(root, batch_id="second")
            with self.assertRaisesRegex(ValueError, "multiple supplied batches"):
                bootstrap.collect_models([a, b], bootstrap.BASELINES)

    def test_reject_unscored_run_and_changed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            a = create_fixture(root, ids=("qwen35_vlm",), batch_id="first")
            b = create_fixture(root, ids=("minicpm_vlm",), batch_id="second")
            content = json.loads(b.read_text())
            content["manifest_sha256"] = "x" * 64
            write_json(b, content)
            with self.assertRaisesRegex(ValueError, "manifest hashes differ"):
                bootstrap.collect_models([a, b], ("qwen35_vlm", "minicpm_vlm"))

    def test_reject_invalid_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            batch = create_fixture(root, ids=("qwen35_vlm",))
            data = json.loads(batch.read_text())
            path = root / data["runs"][0]["run_id"] / "predictions_scored.json"
            rows = json.loads(path.read_text())
            rows[0]["score"] = None
            write_json(path, rows)
            with self.assertRaisesRegex(ValueError, "invalid official score"):
                bootstrap.collect_models([batch], ("qwen35_vlm",))


if __name__ == "__main__":
    unittest.main()
