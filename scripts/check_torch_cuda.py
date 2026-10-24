"""Small real PyTorch CUDA computation; run only inside an allocated SLURM GPU job."""
import sys

def main():
    import torch
    import transformers

    print("=== PyTorch CUDA environment ===", flush=True)
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the allocated GPU (check CUDA-enabled wheel and driver)")
    print("GPU count:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
    a = torch.randn((512, 512), device="cuda")
    result = a @ a
    torch.cuda.synchronize()
    if result.device.type != "cuda" or not torch.isfinite(result).all().item():
        raise RuntimeError("PyTorch GPU matrix multiplication failed")
    print("Matrix result device:", result.device)
    print("PyTorch CUDA: PASSED", flush=True)

if __name__ == "__main__":
    main()
