"""Validate separately installed Helix environments without using a GPU.

Examples:
  .venv/bin/python scripts/check_dependencies.py --stage vlm
  .venv-ocr/bin/python scripts/check_dependencies.py --stage ocr
  .venv/bin/python scripts/check_dependencies.py --stage all

"all" checks .venv for Torch and delegates OCR checks to .venv-ocr.
"""
from argparse import ArgumentParser
from importlib.metadata import PackageNotFoundError, version
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "base": ("ulm-extractor", "PyYAML", "pandas"),
    "vlm": ("torch", "torchvision", "transformers", "accelerate", "Pillow"),
    "ocr": ("paddlepaddle-gpu", "paddleocr", "paddlex"),
}
HELIX_EXPECTED = {
    "torch": "2.14.1+cu130",
    "torchvision": "0.29.1+cu130",
    "paddlepaddle-gpu": "3.3.0",
    "paddleocr": "3.7.0",
    "paddlex": "3.7.2",
}
# Shared runtime libraries that the user's Paddle install downgraded.
TORCH_LIBRARIES = {
    "nvidia-cudnn-cu13": "9.24.0.43",
    "nvidia-nccl-cu13": "2.30.7",
}


def _installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def check(stage="all"):
    if stage == "all":
        main_python = ROOT / ".venv" / "bin" / "python"
        ocr_python = ROOT / ".venv-ocr" / "bin" / "python"
        if not main_python.is_file() or not ocr_python.is_file():
            print("Missing .venv or .venv-ocr; see README installation instructions")
            return 1
        for env, step in ((main_python, "vlm"), (ocr_python, "ocr")):
            print(f"\nChecking {env.name} in {env.parent.parent.name}: {step}", flush=True)
            result = subprocess.run([str(env), str(Path(__file__).resolve()), "--stage", step])
            if result.returncode:
                return 1
        return 0

    if sys.version_info[:2] != (3, 12):
        print(f"ERROR: Python 3.12 required; using {sys.version.split()[0]}")
        return 1
    is_ocr = stage == "ocr"
    expected_env = ".venv-ocr" if is_ocr else ".venv"
    if Path(sys.prefix).name != expected_env:
        print(f"ERROR: {stage} checks must run in {expected_env}, got {sys.prefix}")
        return 1

    required = list(REQUIRED["base"])
    if stage in {"vlm", "ocr"}:
        required += REQUIRED[stage]
    problems = []
    for dist in required:
        installed = _installed(dist)
        if installed is None:
            print(f"MISSING  {dist}")
            problems.append(dist)
            continue
        if dist == "torch":
            # CUDA-local suffixes may be absent from wheel metadata on Helix.
            # The imported runtime reports the actual build, e.g. 2.14.1+cu130.
            try:
                import torch
                installed = torch.__version__
            except Exception as exc:
                print(f"ERROR    import torch: {type(exc).__name__}: {exc}")
                problems.append("torch import")
                continue
        expected = HELIX_EXPECTED.get(dist)
        if expected and installed != expected:
            print(f"CHECK    {dist} {installed} (expected {expected})")
            problems.append(dist)
        else:
            print(f"OK       {dist} {installed}")

    if is_ocr:
        if _installed("paddlepaddle"):
            print("CONFLICT CPU-only paddlepaddle must not coexist with paddlepaddle-gpu")
            problems.append("paddlepaddle")
        if _installed("torch"):
            print("CONFLICT torch installed in the PaddleOCR virtual environment")
            problems.append("torch")
        if importlib.util.find_spec("paddle") is None:
            print("MISSING  Python module 'paddle' (install paddlepaddle-gpu)")
            problems.append("paddle")
        else:
            try:
                import paddle
                gpu_build = bool(paddle.is_compiled_with_cuda())
                print(f"OK       import paddle {paddle.__version__}; CUDA build: {gpu_build}")
                if not gpu_build:
                    problems.append("paddle CUDA")
            except Exception as exc:
                print(f"ERROR    import paddle: {type(exc).__name__}: {exc}")
                problems.append("paddle import")
    else:
        if _installed("paddlepaddle-gpu"):
            print("CONFLICT paddlepaddle-gpu installed in PyTorch environment")
            problems.append("paddlepaddle-gpu")
        if stage == "vlm":
            for dist, expected in TORCH_LIBRARIES.items():
                actual = _installed(dist)
                if actual != expected:
                    print(f"CHECK    {dist} {actual} (PyTorch requires {expected})")
                    problems.append(dist)

    if problems:
        print("\nEnvironment is NOT ready. Check the install steps; do not resume yet.")
        return 1
    print(f"\n{expected_env} dependency check PASSED. Run GPU smoke tests next.")
    return 0


def main(argv=None):
    parser = ArgumentParser(description="Verify separate Helix PyTorch/Paddle environments")
    parser.add_argument("--stage", choices=("base", "vlm", "ocr", "all"), default="all")
    return check(parser.parse_args(argv).stage)


if __name__ == "__main__":
    raise SystemExit(main())
