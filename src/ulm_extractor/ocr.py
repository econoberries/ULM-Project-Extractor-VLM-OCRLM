"""Run and separately cache flat or coordinate-preserving PaddleOCR once per batch."""
from pathlib import Path
import importlib.metadata
import json
import time

from .dataset import image_for
from .layout import FORMAT, format_layout_json
from .storage import key_for, read_json, write_json

ENGINES = {"paddleocr", "paddleocr_layout"}


def synchronize_paddle():
    import paddle
    if paddle.is_compiled_with_cuda():
        paddle.device.synchronize()


def extract_text(result):
    # The original flat-text baseline is deliberately unchanged.
    value = result.json if hasattr(result, "json") else result
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("Unexpected PaddleOCR result representation")
    source = value.get("res", value)
    texts = source.get("rec_texts")
    if not isinstance(texts, list):
        raise ValueError("PaddleOCR output missing rec_texts")
    return "\n".join(str(line) for line in texts)


def run_ocr(batch, engine="paddleocr"):
    if engine not in ENGINES:
        raise ValueError(f"Unknown OCR engine: {engine}")
    from paddleocr import PaddleOCR
    if engine == "paddleocr_layout":
        from PIL import Image

    records = read_json(batch["documents_file"])
    root = Path(batch["output_root"]) / "batches" / batch["batch_id"] / "ocr" / engine
    indexfile = root / "index.json"
    version = importlib.metadata.version("paddleocr")
    if indexfile.exists():
        index = read_json(indexfile)
        if (index.get("manifest_sha256") != batch["manifest_sha256"]
                or index.get("paddleocr_version") != version):
            raise ValueError("OCR cache version/manifest mismatch")
        if engine == "paddleocr_layout" and index.get("format") != FORMAT:
            raise ValueError("Layout OCR cache format mismatch")
    else:
        index = {"manifest_sha256": batch["manifest_sha256"],
                 "paddleocr_version": version, "records": {}}
        if engine == "paddleocr_layout":
            index["format"] = FORMAT

    pending = [
        record for record in records
        if key_for(record) not in index["records"]
        or not Path(index["records"][key_for(record)]["text_path"]).is_file()
    ]
    if not pending:
        print(f"{engine} cache complete; nothing to extract.", flush=True)
        return
    # Same recognition model/parameters for flat OCR and layout-preserved OCR.
    ocr = PaddleOCR(lang="en", device="gpu:0",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    for n, record in enumerate(pending, 1):
        key = key_for(record)
        ext = ".json" if engine == "paddleocr_layout" else ".txt"
        text_path = root / "text" / (key + ext)
        image_path = image_for(record, batch["image_directory"])
        synchronize_paddle()
        start = time.perf_counter()
        response = ocr.predict(str(image_path))
        synchronize_paddle()
        if engine == "paddleocr_layout":
            with Image.open(image_path) as image:
                content = format_layout_json(response, image.size)
        else:
            content = "\n".join(extract_text(entry) for entry in response)
        elapsed = time.perf_counter() - start
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(content, encoding="utf-8")
        index["records"][key] = {
            "text_path": str(text_path), "ocr_seconds": round(elapsed, 4)
        }
        write_json(indexfile, index)
        print(f"[{engine} {n}/{len(pending)}] {key} {elapsed:.3f}s", flush=True)
    print(f"Shared {engine} cache: {indexfile}", flush=True)
