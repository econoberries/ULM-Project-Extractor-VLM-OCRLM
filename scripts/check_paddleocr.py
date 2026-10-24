"""Run the exact PaddleOCR 3.x constructor used by the benchmark on ONE image."""
import argparse
import json
from pathlib import Path

def recognized_lines(output):
    """Extract recognized text from PaddleOCR 3.x Result or JSON-like output."""
    value = output.json if hasattr(output, "json") else output
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("PaddleOCR result is neither a dictionary nor JSON")
    data = value.get("res", value)
    if not isinstance(data, dict) or not isinstance(data.get("rec_texts"), list):
        raise ValueError("PaddleOCR result is missing the rec_texts list")
    return [str(line) for line in data["rec_texts"]]

def main(argv=None):
    parser = argparse.ArgumentParser(description="PaddleOCR A100 single-image smoke test")
    parser.add_argument("--image", required=True, type=Path,
                        help="An existing JPEG or PNG (e.g. an OCRBench EN_part image)")
    args = parser.parse_args(argv)
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    import paddle
    from paddleocr import PaddleOCR
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
        raise RuntimeError("GPU PaddlePaddle is not available. Run the paddle check first.")

    print("=== PaddleOCR one-image inference ===", flush=True)
    print("Image:", image_path)
    print("PaddlePaddle version:", paddle.__version__, flush=True)
    # Keep constructor consistent with src/ulm_extractor/ocr.py.
    engine = PaddleOCR(
        lang="en",
        device="gpu:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    outputs = engine.predict(str(image_path))
    if not outputs:
        raise RuntimeError("PaddleOCR returned no result objects")
    lines = [line for output in outputs for line in recognized_lines(output)]
    print("Result objects:", len(outputs))
    print("Recognized lines:", len(lines))
    print("First five lines:", lines[:5])
    print("PaddleOCR single-image inference: PASSED", flush=True)

if __name__ == "__main__":
    main()
