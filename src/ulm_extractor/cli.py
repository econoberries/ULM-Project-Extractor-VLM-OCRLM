"""Extractor command line: two required inputs, SLURM-first execution."""
import argparse
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .dataset import load_documents
from .scheduler import ROOT, submit_batch

def main(argv=None):
    parser = argparse.ArgumentParser(prog="extractor")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("benchmark", help="Submit all configured experiments through SLURM")
    p.add_argument("--input", type=Path, required=True, help="OCRBench_v2/EN_part directory")
    p.add_argument("--models", type=Path, required=True, help="Explicit model.yml configurations")
    p.add_argument("--annotations", type=Path, default=None, help="Optional OCRBench_v2.json override")
    p.add_argument("--evaluator", type=Path,
                   default=ROOT.parent / "MultimodalOCR" / "OCRBench_v2" / "eval_scripts" / "eval.py")
    p.add_argument("--output-root", type=Path, default=ROOT / "runs")
    p.add_argument("--partition", default="gpu-single", help="GPU SLURM partition")
    p.add_argument("--collector-partition", default="cpu-single", help="CPU collector partition")
    p.add_argument("--gres", default="gpu:A100:1", help="SLURM GPU request")
    p.add_argument("--gpu-family", default="A100", help="Check allocated GPU name")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--dry-run", action="store_true", help="Validate and print jobs without writing or submitting")
    args = parser.parse_args(argv)
    if args.action == "benchmark":
        try:
            if args.cpus < 1:
                raise ValueError("--cpus must be positive")
            models = load_config(args.models)
            dataset = load_documents(args.input, args.annotations)
            print(f"Found {len(dataset['documents'])} English KIE records; "
                  f"manifest SHA256 {dataset['manifest_sha256']}", flush=True)
            submit_batch(dataset, models, args)
            return 0
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

if __name__ == "__main__":
    raise SystemExit(main())
