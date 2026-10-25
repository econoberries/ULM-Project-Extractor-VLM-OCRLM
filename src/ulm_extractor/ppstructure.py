"""PP-StructureV3 -> OCR-complete Markdown -> existing Qwen3.5 adapter.

This updates the *same* ppstructurev3/qwen35_ppstructure architecture.
We retain original PP tables/headings, recover dropped text from PP global OCR
and the established English PaddleOCR recognizer, and resolve PP image crops.
Both original and reconstructed Markdown and raw predictions remain auditable.
"""
from importlib.metadata import version
from pathlib import Path
import time

from .dataset import image_for
from .markdown_recovery import (
    FORMAT, RECOVERY, ocr_lines, resolve_markdown,
)
from .storage import key_for, read_json, write_json

SETTINGS = {
    "lang": "en",
    "device": "gpu:0",
    "use_table_recognition": True,
    "use_formula_recognition": False,
    "use_chart_recognition": False,
    "use_seal_recognition": False,
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
}

# Exactly the same English OCR constructor as the flat PaddleOCR baseline.
FALLBACK_SETTINGS = {
    "lang": "en",
    "device": "gpu:0",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
}


def extract_markdown(result, allow_empty=False):
    """Read PP's rendered Markdown, but permit empty results for OCR recovery."""
    value = result.markdown if hasattr(result, "markdown") else result
    if callable(value):
        value = value()
    if not isinstance(value, dict):
        raise TypeError("PP-StructureV3 Result has no markdown mapping")
    text = value.get("markdown_texts")
    if not isinstance(text, str):
        raise TypeError("PP-StructureV3 markdown_texts must be a string")
    if not text.strip() and not allow_empty:
        raise ValueError("PP-StructureV3 produced empty Markdown")
    return text


def synchronize_paddle():
    import paddle
    if paddle.is_compiled_with_cuda():
        paddle.device.synchronize()


def image_crop_ocr(result, base_markdown, recognizer, destination):
    """OCR PP's actual embedded image crops and keep them for reproducibility."""
    import numpy as np
    info = result.markdown
    images = info.get("markdown_images") or {}
    if not isinstance(images, dict):
        raise TypeError("Unexpected markdown_images mapping")
    recognized = {}
    for raw_name, picture in images.items():
        name = str(raw_name)
        if name not in base_markdown or picture is None:
            continue
        filename = Path(name).name
        crop_path = destination / filename
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(picture, "convert"):
            picture = picture.convert("RGB")
            picture.save(crop_path)
            pixels = np.asarray(picture)
        else:
            pixels = np.asarray(picture)
            from PIL import Image
            Image.fromarray(pixels).save(crop_path)
        result_lines = []
        for output in recognizer.predict(pixels):
            result_lines += ocr_lines(output, source="pp-image-crop")
        recognized[name] = result_lines
    return recognized


def run_ppstructure(batch):
    from paddleocr import PaddleOCR, PPStructureV3

    records = read_json(batch["documents_file"])
    root = Path(batch["output_root"]) / "batches" / batch["batch_id"] / "ocr" / "ppstructurev3"
    index_path = root / "index.json"
    versions = {"paddleocr_version": version("paddleocr"),
                "paddlex_version": version("paddlex")}
    expected = {
        "manifest_sha256": batch["manifest_sha256"],
        "format": FORMAT, "settings": SETTINGS,
        "recovery": RECOVERY, "fallback_settings": FALLBACK_SETTINGS,
        **versions,
    }
    if index_path.exists():
        index = read_json(index_path)
        for field, val in expected.items():
            if index.get(field) != val:
                raise ValueError(
                    f"PP-StructureV3 cache {field} mismatch: this batch uses "
                    "an older Markdown format. Start a new batch; keep old results."
                )
    else:
        index = {**expected, "records": {}}

    pending = [row for row in records
               if key_for(row) not in index["records"]
               or not Path(index["records"][key_for(row)]["text_path"]).is_file()]
    if not pending:
        print("PP-StructureV3 recovered Markdown cache complete.", flush=True)
        return

    parser = PPStructureV3(**SETTINGS)
    recognizer = PaddleOCR(**FALLBACK_SETTINGS)

    for position, row in enumerate(pending, 1):
        key = key_for(row)
        image_path = image_for(row, batch["image_directory"])
        markdown_path = root / "markdown" / (key + ".md")
        original_path = root / "original_markdown" / (key + ".md")
        ocr_path = root / "fallback_ocr" / (key + ".txt")
        audit_path = root / "audit" / (key + ".json")
        raw_folder = root / "raw_json"
        crop_folder = root / "images" / key

        synchronize_paddle()
        start = time.perf_counter()
        results = list(parser.predict(str(image_path)))
        if not results:
            raise ValueError(f"PP-StructureV3 returned no pages: {image_path}")

        # Full-image English OCR is needed because PP may omit complete regions
        # (e.g. nutrition facts with no image placeholder).
        fallback = []
        for output in recognizer.predict(str(image_path)):
            fallback.extend(ocr_lines(output, source="flat-ocr"))
        ocr_path.parent.mkdir(parents=True, exist_ok=True)
        ocr_path.write_text("\n".join(row["text"] for row in fallback), encoding="utf-8")

        pages = []
        originals = []
        audits = []
        raw_paths = []
        for page, result in enumerate(results):
            raw_path = raw_folder / f"{key}-page{page}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            result.save_to_json(save_path=str(raw_path))
            raw_paths.append(str(raw_path))
            base_markdown = extract_markdown(result, allow_empty=True)
            originals.append(base_markdown)

            crops = image_crop_ocr(result, base_markdown, recognizer, crop_folder)
            recovered, diagnostics = resolve_markdown(
                base_markdown, result, fallback if page == 0 else [], crops
            )
            pages.append(recovered)
            audits.append(diagnostics)

        synchronize_paddle()
        seconds = time.perf_counter() - start
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_text("\n\n".join(originals), encoding="utf-8")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("\n\n".join(pages), encoding="utf-8")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(audit_path, {"document_key": key, "pages": audits})

        # Publish the cache index last: interrupted runs cannot use partial files.
        index["records"][key] = {
            "text_path": str(markdown_path),
            "original_markdown_path": str(original_path),
            "fallback_ocr_path": str(ocr_path),
            "raw_json_paths": raw_paths,
            "audit_path": str(audit_path),
            "ocr_seconds": round(seconds, 4),
        }
        write_json(index_path, index)
        print(f"[PP-StructureV3 {position}/{len(pending)}] {key} "
              f"{seconds:.3f}s, {sum(a['additional_ocr_lines'] for a in audits)} "
              "unplaced OCR lines", flush=True)
    print(f"Shared PP-StructureV3 recovered Markdown cache: {index_path}", flush=True)
