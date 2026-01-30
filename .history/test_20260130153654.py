# test_gpu.py
import torch
import time

print("=== GPU / CUDA TEST ===")
print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name        :", torch.cuda.get_device_name(0))
    print("VRAM total (GB) :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

    # petit calcul GPU
    x = torch.randn(10000, 10000, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    y = x @ x
    torch.cuda.synchronize()
    print("Matrix mul time :", round(time.time() - t0, 3), "sec")
else:
    print("❌ CUDA non détecté")
