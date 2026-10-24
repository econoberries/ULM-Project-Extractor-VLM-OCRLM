"""Regression checks for the packages needed by the SLURM GPU workers."""
from pathlib import Path
import ast
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]

class HelixEnvironmentContract(unittest.TestCase):
    def test_project_extras_cover_the_previous_failures(self):
        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = cfg["project"]["optional-dependencies"]
        self.assertTrue(any(x.startswith("torchvision") for x in extras["vlm"]))
        self.assertTrue(any(x.startswith("paddleocr") for x in extras["ocr"]))
        self.assertTrue(cfg["project"]["requires-python"].startswith(">=3.12"))

    def test_cuda_130_profile_is_explicit(self):
        pytorch = (ROOT / "requirements" / "helix-pytorch-cu130.txt").read_text()
        paddle = (ROOT / "requirements" / "helix-paddle-cu130.txt").read_text()
        self.assertIn("torch==2.14.1+cu130", pytorch)
        self.assertIn("torchvision==0.29.1+cu130", pytorch)
        self.assertIn("paddlepaddle-gpu==3.3.0", paddle)
        self.assertIn("/cu130/", paddle)
        self.assertIn("paddlepaddle-gpu", paddle)
        self.assertNotIn("paddlepaddle==", paddle)
        self.assertIn("paddleocr==3.7.0", tomllib.loads(
            (ROOT / "pyproject.toml").read_text()
        )["project"]["optional-dependencies"]["ocr"])

    def test_kie_only_evaluator_requirements(self):
        path = ROOT / "requirements" / "ocrbench-v2-kie-eval.txt"
        packages = [line.strip().lower() for line in path.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        self.assertIn("apted", packages)
        self.assertIn("distance", packages)
        self.assertIn("ipdb", packages)
        self.assertIn("lxml", packages)
        self.assertNotIn("polygon3", packages)

    def test_environment_checker_parses(self):
        path = ROOT / "scripts" / "check_dependencies.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('find_spec("paddle")', source)
        self.assertIn("paddle.is_compiled_with_cuda()", source)

if __name__ == "__main__":
    unittest.main()
