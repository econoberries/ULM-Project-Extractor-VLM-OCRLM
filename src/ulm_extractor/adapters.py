"""Model-specific input processors; original OCRBench questions are never rewritten."""
import time
from pathlib import Path

class InferenceAdapter:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.torch = None
        self.model_load_seconds = None

    def load(self):
        import torch
        from transformers import (
            AutoModelForMultimodalLM, AutoModelForImageTextToText,
            AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("Model inference requires a CUDA device")
        self.torch = torch
        start = time.perf_counter()
        adapter = self.config["adapter"]
        model_id, revision = self.config["checkpoint"], self.config["revision"]
        if adapter == "qwen25":
            self.processor = AutoTokenizer.from_pretrained(model_id, revision=revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, device_map="auto", dtype="auto"
            )
        elif adapter == "minicpm":
            self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, revision=revision, device_map="auto", dtype="auto"
            )
        else:
            self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_id, revision=revision, device_map="auto", dtype="auto"
            )
        self.model.eval()
        self.model_load_seconds = round(time.perf_counter() - start, 4)

    def _synchronize(self):
        for i in range(self.torch.cuda.device_count()):
            self.torch.cuda.synchronize(i)

    def infer(self, question, image_path=None, ocr_text=None):
        """Return raw text and standardized per-document performance attributes.

        Timing includes model-specific input preprocessing, generation and decoding.
        It excludes model loading and writing result files.
        """
        from PIL import Image
        torch = self.torch
        max_tokens = self.config["max_new_tokens"]
        is_vlm = self.config["pipeline"] == "vlm"
        adapter = self.config["adapter"]
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        self._synchronize()
        start = time.perf_counter()
        if is_vlm:
            if not image_path:
                raise ValueError("VLM requires an image")
            picture = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": picture},
                {"type": "text", "text": question},
            ]}]
        else:
            if ocr_text is None:
                raise ValueError("OCR->LM requires extracted text")
            payload = ocr_text + "\n\n" + question
            content = payload if adapter == "qwen25" else [{"type": "text", "text": payload}]
            messages = [{"role": "user", "content": content}]
        options = dict(tokenize=True, add_generation_prompt=True,
                       return_dict=True, return_tensors="pt")
        if adapter == "minicpm":
            options.update(enable_thinking=False,
                           processor_kwargs={"downsample_mode": "16x", "max_slice_nums": 36})
        elif adapter == "qwen35":
            options["enable_thinking"] = False
        inputs = self.processor.apply_chat_template(messages, **options).to(self.model.device)
        generate_options = {"max_new_tokens": max_tokens, "do_sample": False}
        if adapter == "minicpm":
            generate_options.update(downsample_mode="16x", num_beams=1)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_options)
        self._synchronize()
        num_prompt_tokens = int(inputs["input_ids"].shape[-1])
        tokens = output[0][num_prompt_tokens:]
        if adapter == "minicpm":
            prediction = self.processor.batch_decode(
                [tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            prediction = self.processor.decode(tokens, skip_special_tokens=True)
        self._synchronize()
        elapsed = time.perf_counter() - start
        mem = [int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())]
        return {
            "predict": prediction,
            "lm_seconds": round(elapsed, 4),
            "output_tokens": int(tokens.numel()),
            "hit_token_limit": int(tokens.numel()) >= max_tokens,
            "peak_allocated_gpu_bytes": mem,
        }

    def environment(self):
        import transformers
        return {
            "gpu_names": [self.torch.cuda.get_device_name(i)
                          for i in range(self.torch.cuda.device_count())],
            "torch_version": self.torch.__version__,
            "transformers_version": transformers.__version__,
            "model_load_time_s": self.model_load_seconds,
            "dtype": str(self.model.dtype),
        }
