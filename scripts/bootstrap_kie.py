#!/usr/bin/env python3
"""Paired, dataset-stratified bootstrap for official OCRBench v2 English KIE.

Reads existing predictions_scored.json files. It NEVER calls the models or
reruns the official evaluator. Resamples the same documents jointly across
all models (not 400 independent documents per model).

Example:
 .venv/bin/python scripts/bootstrap_kie.py \
   --batch runs/batches/<BATCH_UUID>/batch.json --n-bootstrap 10000 --seed 2026

Choose arbitrary experiment IDs with --models. For models scored in different
batches, repeat --batch; all batches must have identical manifest hashes.
Repeated experiment IDs from different batches are rejected deliberately.
"""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

BASELINES = ("qwen35_vlm", "minicpm_vlm", "qwen35_ocr", "qwen25_ocr")
DATASETS = ("FUNSD", "POIE", "Receipt", "SROIE")
EXPECTED_COUNTS = {"FUNSD": 48, "POIE": 177, "Receipt": 25, "SROIE": 150}


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def as_batch_path(raw):
    file = Path(raw).expanduser().resolve()
    if file.is_dir():
        file = file / "batch.json"
    if not file.is_file() or file.name != "batch.json":
        raise FileNotFoundError(f"Expected an existing batch.json: {file}")
    if file.parent.parent.name != "batches":
        raise ValueError(f"Batch must be under runs/batches/<batch-id>/: {file}")
    return file


def identity(row):
    return (str(row["dataset_name"]), str(row["id"]),
            str(row["image_path"]), str(row["question"]))


def read_scored(path, expected_documents):
    predictions = load_json(path)
    if not isinstance(predictions, list) or len(predictions) != 400:
        raise ValueError(f"Expected 400 scored predictions in {path}")
    scores = {}
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError(f"Non-object prediction in {path}")
        key = identity(row)
        if key in scores:
            raise ValueError(f"Duplicate document in {path}: {key[:3]}")
        value = row.get("score")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Missing/invalid official score in {path}: {key[:3]}")
        if row.get("error"):
            raise ValueError(f"Failed prediction cannot be bootstrapped: {path}")
        scores[key] = float(value)
    if set(scores) != expected_documents:
        missing = list(expected_documents - set(scores))[:2]
        extra = list(set(scores) - expected_documents)[:2]
        raise ValueError(
            f"Document identities do not match across runs for {path}. "
            f"Examples missing={missing}, extra={extra}"
        )
    return scores


def collect_models(batch_files, requested):
    """Return aligned document keys, scores and metadata from selected batches."""
    selected = {}
    dataset_hash = None
    canonical = None
    docs = None
    details = []
    for batch_path in batch_files:
        batch_path = as_batch_path(batch_path)
        batch = load_json(batch_path)
        batch_id = batch_path.parent.name
        if batch.get("batch_id") != batch_id:
            raise ValueError(f"Invalid batch identifier in {batch_path}")
        sha = batch.get("manifest_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"Missing manifest SHA256 in {batch_path}")
        if dataset_hash is None:
            dataset_hash = sha
        elif sha != dataset_hash:
            raise ValueError("Dataset manifest hashes differ: cannot bootstrap unmatched batches")
        run_root = batch_path.parents[2]  # portable after copying a runs/ archive
        document_file = batch_path.parent / "documents.json"
        if not document_file.is_file():
            raise FileNotFoundError(f"Missing original documents.json: {document_file}")
        original = load_json(document_file)
        if not isinstance(original, list) or len(original) != 400:
            raise ValueError(f"Expected exactly 400 documents in {document_file}")
        ids = [identity(row) for row in original]
        if len(set(ids)) != 400:
            raise ValueError(f"Duplicate documents in {document_file}")
        if canonical is None:
            canonical = ids
            docs = original
        elif set(ids) != set(canonical):
            raise ValueError(f"Document identifiers differ in {batch_path}")
        for model in batch["runs"]:
            name = model["id"]
            if name not in requested:
                continue
            if name in selected:
                raise ValueError(
                    f"Model {name!r} appears in multiple supplied batches. "
                    "Choose only one batch per implementation; never mix "
                    "historical and revised PP versions implicitly."
                )
            run_id = model["run_id"]
            run_dir = run_root / run_id
            state_file = run_dir / "run.json"
            if not state_file.is_file():
                raise FileNotFoundError(f"Missing run status: {state_file}")
            state = load_json(state_file)
            if state.get("status") != "scored":
                raise ValueError(f"{name} is not officially scored ({state.get('status')})")
            if state.get("batch_id") != batch_id:
                raise ValueError(f"Run {run_id} belongs to another batch")
            scored_file = run_dir / "predictions_scored.json"
            if not scored_file.is_file():
                raise FileNotFoundError(f"Missing official scores: {scored_file}")
            scoring = read_scored(scored_file, set(canonical))
            score_vector = np.asarray([scoring[k] for k in canonical], dtype=float)
            # Prevent disagreement between saved official aggregate and scores.
            metrics_path = run_dir / "metrics.json"
            if metrics_path.is_file():
                expected = load_json(metrics_path).get("official_kie_f1")
                if expected is not None and not math.isclose(
                    float(expected), float(score_vector.mean()), rel_tol=0, abs_tol=1e-8
                ):
                    raise ValueError(f"Saved official F1 and document scores disagree for {name}")
            selected[name] = score_vector
            details.append({
                "experiment_id": name, "batch_id": batch_id,
                "run_id": run_id, "pipeline": model.get("pipeline"),
                "ocr_engine": model.get("ocr"),
                "checkpoint": model.get("checkpoint"),
                "model_revision": model.get("revision"),
                "scored_predictions": str(scored_file),
            })
    missing = set(requested) - set(selected)
    if missing:
        raise ValueError(f"Models not found in provided batch(es): {sorted(missing)}")
    return canonical, docs, selected, details, dataset_hash


def bootstrap_indices(groups, n, draws, rng, stratified=True):
    indices = np.empty((draws, n), dtype=np.int32)
    if stratified:
        # Fixed 48/177/25/150 proportions; pairing is retained in each draw.
        for positions in groups.values():
            sample = rng.choice(positions, size=(draws, len(positions)), replace=True)
            indices[:, positions] = sample
    else:
        indices[:] = rng.integers(0, n, size=(draws, n))
    return indices


def ci(values, alpha):
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def make_results(ids, docs, scores, requested, draws=10000, seed=2026,
                 confidence=0.95, stratified=True):
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between 0 and 1")
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    if len(ids) != 400:
        raise ValueError("English KIE comparison must include exactly 400 documents")
    rng = np.random.default_rng(seed)
    groups = {
        dataset: np.asarray(
            [i for i, row in enumerate(docs)
             if str(row["dataset_name"]).casefold() == dataset.casefold()],
            dtype=np.int32
        ) for dataset in DATASETS
    }
    if stratified and {k: len(v) for k, v in groups.items()} != EXPECTED_COUNTS:
        raise ValueError("Unexpected OCRBench v2 dataset composition for stratified bootstrap")
    indices = bootstrap_indices(groups, len(ids), draws, rng, stratified)
    alpha = 1 - confidence
    pairs = list(itertools.combinations(requested, 2))
    adjusted_alpha = alpha / len(pairs)  # conservative Bonferroni family-wise CIs

    models = []
    pairwise = []
    # The overall estimate is always mean document score, matching the official
    # OCRBench KIE collector; not a micro-F1 calculated from pooled token counts.
    scopes = [("ALL", np.arange(len(ids)))] + [
        (dataset, positions) for dataset, positions in groups.items()
    ]
    for dataset, positions in scopes:
        if len(positions) == 0:
            continue
        # Resample within each dataset for subset CIs. For overall CIs, retain
        # benchmark proportions if --stratified (default).
        if dataset == "ALL":
            sample_positions = indices
        else:
            sample_positions = rng.choice(positions, size=(draws, len(positions)),
                                          replace=True)
        means = {}
        boot = {}
        for name in requested:
            values = scores[name][positions]
            estimate = float(np.mean(values))
            sample_means = scores[name][sample_positions].mean(axis=1)
            means[name] = estimate
            boot[name] = sample_means
            low, high = ci(sample_means, alpha)
            models.append({
                "dataset": dataset, "experiment_id": name,
                "documents": len(positions), "official_mean_score": estimate,
                "ci_low": low, "ci_high": high,
                "confidence": confidence, "bootstrap_draws": draws,
            })
        for name_a, name_b in pairs:
            estimate = means[name_a] - means[name_b]
            boot_diff = boot[name_a] - boot[name_b]
            low, high = ci(boot_diff, alpha)
            family_low, family_high = ci(boot_diff, adjusted_alpha)
            pairwise.append({
                "dataset": dataset, "model_a": name_a, "model_b": name_b,
                "documents": len(positions), "mean_difference_a_minus_b": estimate,
                "ci_low": low, "ci_high": high,
                "ci_excludes_zero": low > 0 or high < 0,
                "familywise_ci_low": family_low,
                "familywise_ci_high": family_high,
                "familywise_ci_excludes_zero": family_low > 0 or family_high < 0,
                "confidence": confidence, "bootstrap_draws": draws,
                "number_of_pairwise_comparisons": len(pairs),
            })
    aligned = []
    for i, row in enumerate(docs):
        aligned.append({
            "document_key": row.get("document_key", ""),
            "dataset_name": str(row["dataset_name"]),
            "id": str(row["id"]),
            "image_path": str(row["image_path"]),
            **{name: float(scores[name][i]) for name in requested},
        })
    return models, pairwise, aligned


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Nothing to save: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(batch_paths, models=BASELINES, draws=10000, seed=2026,
            confidence=0.95, stratified=True, output_dir=None):
    requested = tuple(models)
    if len(requested) < 2 or len(set(requested)) != len(requested):
        raise ValueError("Request at least two different experiment IDs")
    ids, docs, scores, details, manifest_sha = collect_models(batch_paths, requested)
    results, pairs, aligned = make_results(
        ids, docs, scores, requested,
        draws=draws, seed=seed, confidence=confidence, stratified=stratified
    )
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # Save next to the first source runs/ (never overwrite past experiments).
        base = as_batch_path(batch_paths[0]).parents[2]
        dest = base / "analysis" / f"paired-bootstrap-{stamp}-{uuid4().hex[:8]}"
    else:
        dest = Path(output_dir).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "bootstrap_models.csv", results)
    write_csv(dest / "bootstrap_pairs.csv", pairs)
    write_csv(dest / "paired_document_scores.csv", aligned)
    with (dest / "analysis_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "method": "paired percentile bootstrap of official per-document KIE scores",
            "sampling": "fixed-count stratified by FUNSD/POIE/Receipt/SROIE"
            if stratified else "unstratified over all 400 documents",
            "n_documents": len(ids), "bootstrap_draws": draws,
            "confidence": confidence, "random_seed": seed,
            "manifest_sha256": manifest_sha,
            "models": details,
            "source_batch_paths": [str(as_batch_path(p)) for p in batch_paths],
            "note": (
                "Scores are official per-document KIE scores, averaged across 400 "
                "matched documents. Each draw resamples identical document indices "
                "for every model. Per-dataset CIs use within-subset resampling. "
                "Family-wise intervals use Bonferroni correction for all listed "
                "pairwise model comparisons. This does not account for decoding "
                "stochasticity, dataset representativeness, or multiple hypothesis "
                "families across dataset-specific exploratory analyses."
            ),
        }, f, indent=2)
        f.write("\n")
    print(f"Document manifest: {manifest_sha}")
    print(f"Analyzed {len(ids)} matched documents; {draws} bootstrap draws; seed={seed}.")
    for row in results:
        if row["dataset"] == "ALL":
            print(f"  {row['experiment_id']:<22} mean={row['official_mean_score']:.4f}"
                  f", {int(confidence*100)}% CI [{row['ci_low']:.4f}, "
                  f"{row['ci_high']:.4f}]")
    print(f"Saved results in {dest}: bootstrap_models.csv, bootstrap_pairs.csv, "
          "paired_document_scores.csv, analysis_metadata.json")
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=Path, action="append", required=True,
                   help="Batch directory or batch.json; repeat for separate batches")
    p.add_argument("--models", nargs="+", default=list(BASELINES),
                   help="Experiment IDs to compare (default: four original models)")
    p.add_argument("--n-bootstrap", type=int, default=10000,
                   help="Paired bootstrap draws (default: 10000)")
    p.add_argument("--seed", type=int, default=2026,
                   help="Random-number generator seed (default: 2026)")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence level (default: 0.95)")
    p.add_argument("--unstratified", action="store_true",
                   help="Sample from all 400 documents instead of within document type")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Distinct output directory (default: new unique runs/analysis/)")
    args = p.parse_args()
    try:
        analyze(args.batch, models=args.models, draws=args.n_bootstrap,
                seed=args.seed, confidence=args.confidence,
                stratified=not args.unstratified, output_dir=args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
