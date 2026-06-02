"""
Dump PTX of PyTorch's fused AdamW kernel to inspect the weight decay FMA pattern.
"""
import torch
import numpy as np
import os

# Create a simple test to trigger the fused adam kernel
shape = (64,)
param = torch.randn(shape, dtype=torch.float32, device="cuda", requires_grad=True)
param.grad = torch.randn(shape, dtype=torch.float32, device="cuda")

opt = torch.optim.AdamW([param], lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, fused=True)

# Enable CUDA profiling/PTX dump
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Use torch.cuda.get_gencode_flags() if available
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# Run one step to compile the kernel
opt.step()

# Try to get the compiled kernel info
# Use cuobjdump on the cached kernels
print("\nLooking for cached CUDA kernels...")
cache_dir = os.path.expanduser("~/.cache/torch/kernels")
if os.path.exists(cache_dir):
    for f in os.listdir(cache_dir):
        if "adam" in f.lower():
            print(f"  Found: {f}")
else:
    print("  Cache dir not found")

# Alternative: use TORCH_SHOW_DISPATCH_TRACE or similar
print("\nDone.")
