"""Run pretrained PP-StructureV3 once per image, caching Markdown for the LM.

This is independent of PP-OCRv6 flat and coordinate-JSON caches. Only the
structured Markdown (not parser JSON or image pixels) is passed to Qwen.
"""
from importlib.metadata import version
from pathlib import Path
import time

from .dataset import image_for
from .storage import key_for, read_json, write_json

FORMAT = "ppstructurev3-markdown-v1"
# Explicit, fixed architecture so resumed caches cannot silently change.
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


def extract_markdown(result):
    """Get a page's text Markdown from the documented PaddleOCR Result API."""
    value = result.markdown if hasattr(result, "markdown") else result
    if callable(value):
        value = value()
    if not isinstance(value, dict):
        raise TypeError("PP-StructureV3 Result has no markdown mapping")
    text = value.get("markdown_texts")
    if not isinstance(text, str):
        raise TypeError("PP-StructureV3 markdown_texts must be a string")
    if not text.strip():
        raise ValueError("PP-StructureV3 produced empty Markdown")
    return text


def synchronize_paddle():
    import paddle
    if paddle.is_compiled_with_cuda():
        paddle.device.synchronize()


def run_ppstructure(batch):
    from paddleocr import PPStructureV3

    records = read_json(batch["documents_file"])
    root = Path(batch["output_root"]) / "batches" / batch["batch_id"] / "ocr" / "ppstructurev3"
    index_path = root / "index.json"
    versions = {"paddleocr_version": version("paddleocr"),
                "paddlex_version": version("paddlex")}
    if index_path.exists():
        index = read_json(index_path)
        for field, expected in {
            "manifest_sha256": batch["manifest_sha256"],
            "format": FORMAT,
            "settings": SETTINGS,
            **versions,
        }.items():
            if index.get(field) != expected:
                raise ValueError(f"PP-StructureV3 cache {field} mismatch")
    else:
        index = {
            "manifest_sha256": batch["manifest_sha256"],
            "format": FORMAT,
            "settings": SETTINGS,
            **versions,
            "records": {},
        }

    pending = [
        row for row in records
        if key_for(row) not in index["records"]
        or not Path(index["records"][key_for(row)]["text_path"]).is_file()
    ]
    if not pending:
        print("PP-StructureV3 Markdown cache complete; nothing to parse.", flush=True)
        return

    parser = PPStructureV3(**SETTINGS)
    for i, row in enumerate(pending, 1):
        key = key_for(row)
        image_path = image_for(row, batch["image_directory"])
        markdown_path = root / "markdown" / (key + ".md")
        synchronize_paddle()
        start = time.perf_counter()
        results = list(parser.predict(str(image_path)))
        synchronize_paddle()
        if not results:
            raise ValueError(f"PP-StructureV3 produced no result: {image_path}")
        markdown = "\n\n".join(extract_markdown(result) for result in results)
        seconds = time.perf_counter() - start
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        index["records"][key] = {
            "text_path": str(markdown_path), "ocr_seconds": round(seconds, 4)
        }
        write_json(index_path, index)
        print(f"[PP-StructureV3 {i}/{len(pending)}] {key} {seconds:.3f}s", flush=True)
    print(f"Shared PP-StructureV3 Markdown cache: {index_path}", flush=True)
