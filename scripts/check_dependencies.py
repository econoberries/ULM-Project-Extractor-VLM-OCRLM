"""Fast login-node dependency check; it does NOT test GPU execution.

Usage:
  .venv/bin/python scripts/check_dependencies.py --stage all
  .venv/bin/python scripts/check_dependencies.py --stage vlm
  .venv/bin/python scripts/check_dependencies.py --stage ocr
"""
from argparse import ArgumentParser
from importlib.metadata import PackageNotFoundError, version
import sys

REQUIRED = {
    "base": ("ulm-extractor", "PyYAML", "pandas"),
    "vlm": ("torch", "torchvision", "transformers", "accelerate", "Pillow"),
    "ocr": ("paddlepaddle-gpu", "paddleocr"),
}
HELIX_EXPECTED = {
    "torch": "2.14.1+cu130",
    "torchvision": "0.29.1+cu130",
    "paddlepaddle-gpu": "3.3.0",
}

def check(stage="all"):
    if sys.version_info[:2] != (3, 12):
        print(f"ERROR: Python 3.12 required; using {sys.version.split()[0]}")
        return 1
    requested = ("base", "vlm", "ocr") if stage == "all" else ("base", stage)
    missing = []
    mismatch = []
    seen = set()
    for group in requested:
        for dist in REQUIRED[group]:
            if dist in seen:
                continue
            seen.add(dist)
            try:
                installed = version(dist)
            except PackageNotFoundError:
                print(f"MISSING  {dist}")
                missing.append(dist)
                continue
            expected = HELIX_EXPECTED.get(dist)
            if expected and installed != expected:
                print(f"CHECK    {dist} {installed} (Helix CUDA 13.0 recipe expects {expected})")
                mismatch.append(dist)
            else:
                print(f"OK       {dist} {installed}")
    if missing or mismatch:
        print("\nEnvironment not ready for the selected Helix profile.")
        print("See README installation steps; do not submit the full benchmark yet.")
        return 1
    print("\nInstalled-package check passed. This does NOT prove GPU access.")
    print("Next: submit the Torch/Paddle/PaddleOCR SLURM GPU smoke tests.")
    return 0

def main(argv=None):
    parser = ArgumentParser(description="Check Helix CUDA 13.0 package dependencies without GPU")
    parser.add_argument("--stage", choices=("base", "vlm", "ocr", "all"), default="all")
    return check(parser.parse_args(argv).stage)

if __name__ == "__main__":
    raise SystemExit(main())
