"""Real GPU smoke test: PP-StructureV3 produces nonempty Markdown from one image."""
import argparse
from pathlib import Path

from ulm_extractor.ppstructure import SETTINGS, extract_markdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        parser.error(f"Image does not exist: {image}")
    import paddle
    from paddleocr import PPStructureV3
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PP-StructureV3 smoke test requires allocated CUDA GPU")
    pipeline = PPStructureV3(**SETTINGS)
    results = list(pipeline.predict(str(image)))
    if not results:
        raise ValueError("PP-StructureV3 produced no document result")
    markdown = "\n\n".join(extract_markdown(result) for result in results)
    print("Markdown characters:", len(markdown))
    print("Markdown preview:\n", markdown[:1200])
    print("PP-StructureV3 Markdown single-image inference: PASSED", flush=True)


if __name__ == "__main__":
    main()
