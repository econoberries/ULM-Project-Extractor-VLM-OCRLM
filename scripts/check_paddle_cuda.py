"""Small real PaddlePaddle CUDA computation; separate process from PyTorch."""
import sys

def main():
    import paddle

    print("=== PaddlePaddle CUDA environment ===", flush=True)
    print("Python:", sys.version.split()[0])
    print("PaddlePaddle:", paddle.__version__)
    print("CUDA compiled:", paddle.is_compiled_with_cuda())
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("CPU-only PaddlePaddle detected. Install a Helix-compatible paddlepaddle-gpu.")
    count = paddle.device.cuda.device_count()
    print("GPU count:", count)
    if count < 1:
        raise RuntimeError("PaddlePaddle has CUDA support but cannot see the allocated GPU")
    paddle.set_device("gpu:0")
    a = paddle.randn([512, 512])
    result = paddle.matmul(a, a)
    paddle.device.cuda.synchronize()
    place = str(result.place).lower()
    if "gpu" not in place:
        raise RuntimeError(f"PaddlePaddle did not calculate on GPU: {result.place}")
    print("Matrix result device:", result.place)
    print("PaddlePaddle CUDA: PASSED", flush=True)

if __name__ == "__main__":
    main()
