"""Validate the exact OCRBench v2 English KIE task and its images."""
from pathlib import Path
import hashlib
import json

TASK = "key information extraction en"
FIELDS = ("id", "type", "dataset_name", "image_path", "question", "answers")

def load_documents(image_directory, annotations=None):
    images = Path(image_directory).expanduser().resolve()
    if not images.is_dir() or images.name != "EN_part":
        raise ValueError("--input must point to the existing OCRBench_v2/EN_part directory")
    annotation_file = (Path(annotations).expanduser().resolve() if annotations
                       else (images.parent / "OCRBench_v2.json").resolve())
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Missing annotations: {annotation_file}; use --annotations if elsewhere")
    raw = annotation_file.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list):
        raise ValueError("OCRBench annotations must be a JSON list")
    selected = [row for row in records if isinstance(row, dict) and row.get("type") == TASK]
    if len(selected) != 400:
        raise ValueError(f"Expected exactly 400 English KIE records; got {len(selected)}")
    seen = set()
    for row in selected:
        if any(k not in row or row[k] is None for k in FIELDS):
            raise ValueError(f"Missing benchmark fields for record {row.get('id')}")
        rel = Path(row["image_path"])
        if rel.is_absolute() or ".." in rel.parts or "EN_part" not in rel.parts:
            raise ValueError(f"Unsafe or non-English image path: {rel}")
        parts = rel.parts
        suffix = parts[parts.index("EN_part") + 1:]
        if not suffix:
            raise ValueError(f"Missing image filename: {rel}")
        img = images.joinpath(*suffix).resolve()
        if not img.is_relative_to(images) or not img.is_file():
            raise FileNotFoundError(f"Missing English image: {img}")
        key = (str(row["dataset_name"]), str(row["id"]), row["image_path"])
        if key in seen:
            raise ValueError(f"Duplicate benchmark record: {key}")
        seen.add(key)
        if not isinstance(row["question"], str) or not isinstance(row["answers"], list):
            raise ValueError(f"Invalid question or answers for {key}")
    hash_manifest = hashlib.sha256(json.dumps(selected, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {
        "documents": selected,
        "image_directory": str(images),
        "annotations": str(annotation_file),
        "annotation_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": hash_manifest,
    }

def image_for(document, image_directory):
    rel = Path(document["image_path"])
    suffix = rel.parts[rel.parts.index("EN_part") + 1:]
    return Path(image_directory).joinpath(*suffix)
