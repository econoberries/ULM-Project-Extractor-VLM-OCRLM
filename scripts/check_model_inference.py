"""OPTIONAL: one-document VLM smoke test using the actual benchmark adapter."""
import argparse
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description="One-document direct VLM smoke test")
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to the full OCRBench_v2/EN_part directory")
    parser.add_argument("--models", type=Path, default=Path("model.yml"))
    parser.add_argument("--experiment-id", default="qwen35_vlm")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Smaller cap for a quick functionality check, not a benchmark score")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    from ulm_extractor.config import load_config
    from ulm_extractor.dataset import image_for, load_documents
    from ulm_extractor.adapters import InferenceAdapter

    dataset = load_documents(args.input)
    matches = [m for m in load_config(args.models) if m["id"] == args.experiment_id]
    if len(matches) != 1:
        parser.error(f"Model ID not found in {args.models}: {args.experiment_id}")
    config = dict(matches[0])
    if config["pipeline"] != "vlm":
        parser.error("This optional check currently supports only direct VLM experiments")

    config["max_new_tokens"] = args.max_new_tokens
    document = dataset["documents"][0]
    print("=== Single-document VLM adapter check ===", flush=True)
    print("Experiment:", config["id"], flush=True)
    print("Checkpoint:", config["checkpoint"], flush=True)
    print("Document ID:", document["id"], flush=True)
    model = InferenceAdapter(config)
    model.load()
    answer = model.infer(
        question=document["question"],
        image_path=image_for(document, dataset["image_directory"]),
    )
    print("GPU:", ", ".join(model.environment()["gpu_names"]))
    print("Tokens generated:", answer["output_tokens"])
    print("Prediction sample:", answer["predict"][:500])
    print("Single-document model inference: PASSED", flush=True)

if __name__ == "__main__":
    main()
