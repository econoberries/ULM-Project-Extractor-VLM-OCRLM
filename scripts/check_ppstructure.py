"""Real PP-StructureV3 + English OCR -> text-only Markdown GPU smoke test."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from ulm_extractor.ppstructure import (
    SETTINGS, FALLBACK_SETTINGS, extract_markdown, image_crop_ocr,
)
from ulm_extractor.markdown_recovery import ocr_lines, resolve_markdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        parser.error(f"Image does not exist: {image}")
    import paddle
    from paddleocr import PaddleOCR, PPStructureV3
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PP-StructureV3 smoke test requires allocated CUDA GPU")
    structure = PPStructureV3(**SETTINGS)
    recognizer = PaddleOCR(**FALLBACK_SETTINGS)
    results = list(structure.predict(str(image)))
    if not results:
        raise ValueError("PP-StructureV3 produced no document result")
    fallback = []
    for out in recognizer.predict(str(image)):
        fallback.extend(ocr_lines(out))
    rendered = []
    with TemporaryDirectory() as directory:
        for res in results:
            base = extract_markdown(res, allow_empty=True)
            crops = image_crop_ocr(res, base, recognizer, Path(directory))
            content, audit = resolve_markdown(base, res, fallback, crops)
            rendered.append(content)
            print("OCR completeness audit:", audit)
    markdown = "\n\n".join(rendered)
    print("Resolved Markdown characters:", len(markdown))
    print("Resolved Markdown preview:\n", markdown[:1600])
    print("PP-StructureV3 Markdown single-image inference: PASSED", flush=True)


if __name__ == "__main__":
    main()
