"""One-image smoke test for coordinate-preserving PaddleOCR."""
import argparse
import json
from pathlib import Path
from PIL import Image
from ulm_extractor.layout import FORMAT, format_layout_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    import paddle
    from paddleocr import PaddleOCR
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("An allocated PaddlePaddle GPU is required")
    engine = PaddleOCR(
        lang="en", device="gpu:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    response = engine.predict(str(image_path))
    with Image.open(image_path) as image:
        encoded = format_layout_json(response, image.size)
    parsed = json.loads(encoded)
    if parsed.get("format") != FORMAT or not parsed["lines"]:
        raise RuntimeError("Layout output had no recognized text with boxes")
    print("Layout OCR lines:", len(parsed["lines"]))
    print("Example JSON line:", json.dumps(parsed["lines"][0], ensure_ascii=False))
    print("PaddleOCR layout single-image inference: PASSED", flush=True)


if __name__ == "__main__":
    main()
