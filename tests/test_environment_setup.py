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

    def test_repair_requirements_match_pytorch(self):
        repair = (ROOT / "requirements" / "helix-torch-repair-cu130.txt").read_text()
        self.assertIn("nvidia-cudnn-cu13==9.24.0.43", repair)
        self.assertIn("nvidia-nccl-cu13==2.30.7", repair)

    def test_isolated_ocr_venv_is_gitignored(self):
        ignore = (ROOT / ".gitignore").read_text()
        self.assertIn("/.venv-ocr/", ignore)

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
        self.assertIn("nvidia-cudnn-cu13", source)
        self.assertIn("nvidia-nccl-cu13", source)
        self.assertIn(".venv-ocr", source)

    def test_gpu_workers_use_disjoint_environments(self):
        worker = (ROOT / "slurm" / "worker.sbatch").read_text(encoding="utf-8")
        smoke = (ROOT / "slurm" / "check_environment.sbatch").read_text(encoding="utf-8")
        self.assertIn('ocr) python_bin=".venv-ocr/bin/python"', worker)
        self.assertIn('run) python_bin=".venv/bin/python"', worker)
        self.assertIn('paddle|ocr) python_bin=".venv-ocr/bin/python"', smoke)
        self.assertIn('torch|model) python_bin=".venv/bin/python"', smoke)
        self.assertIn(".venv-ocr", (ROOT / "src" / "ulm_extractor" / "resume.py").read_text())
        self.assertIn(".venv-ocr", (ROOT / "requirements" / "helix-paddle-cu130.txt").read_text())

if __name__ == "__main__":
    unittest.main()
