"""Run PaddleOCR once per batch and cache exactly the selected 400 documents."""
from pathlib import Path
import importlib.metadata
import json
import time

from .dataset import image_for
from .storage import key_for, read_json, write_json

def synchronize_paddle():
    import paddle
    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()

def extract_text(result):
    # PaddleOCR 3.x returns Result objects exposing a JSON-compatible .json view.
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

def run_ocr(batch):
    from paddleocr import PaddleOCR
    records = read_json(batch["documents_file"])
    root = Path(batch["output_root"]) / "batches" / batch["batch_id"] / "ocr" / "paddleocr"
    indexfile = root / "index.json"
    # Batch-level cache: never reuse OCR from a different manifest or run context.
    version = importlib.metadata.version("paddleocr")
    if indexfile.exists():
        index = read_json(indexfile)
        if index.get("manifest_sha256") != batch["manifest_sha256"] or index.get("paddleocr_version") != version:
            raise ValueError("OCR cache version/manifest mismatch")
    else:
        index = {"manifest_sha256": batch["manifest_sha256"],
                 "paddleocr_version": version, "records": {}}
    pending = [record for record in records
               if key_for(record) not in index["records"] or
               not Path(index["records"][key_for(record)]["text_path"]).is_file()]
    if not pending:
        print("OCR cache complete; nothing to extract.")
        return
    ocr = PaddleOCR(lang="en", device="gpu:0",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    for n, record in enumerate(pending, 1):
        doc_key = key_for(record)
        text_path = root / "text" / (doc_key + ".txt")
        image_path = image_for(record, batch["image_directory"])
        synchronize_paddle()
        start = time.perf_counter()
        response = ocr.predict(str(image_path))
        synchronize_paddle()
        elapsed = time.perf_counter() - start
        content = "\n".join(extract_text(entry) for entry in response)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(content, encoding="utf-8")
        index["records"][doc_key] = {
            "text_path": str(text_path), "ocr_seconds": round(elapsed, 4)
        }
        write_json(indexfile, index)
        print(f"[OCR {n}/{len(pending)}] {doc_key} {elapsed:.3f}s", flush=True)
    print(f"Shared OCR cache: {indexfile}")
