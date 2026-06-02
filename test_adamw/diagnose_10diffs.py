"""
Diagnostic: find the exact 10 elements that differ and analyze why.
"""
import os
import numpy as np

os.environ["PADDLE_ADAMW_TORCH_COMPAT"] = "1"
os.environ["FLAGS_use_accuracy_compatible_kernel"] = "1"

import torch
import paddle

BASE = "/home/ningzhengsheng/src/adam_diff_test"
WEIGHT_S1 = f"{BASE}/hf-weights-step1-start"
GRAD_S1 = f"{BASE}/hf-grad-before-optimizer/step1"
GRAD_S2 = f"{BASE}/hf-grad-before-optimizer/step2"

LR = 1e-5
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
WEIGHT_DECAY = 0.01

name = "model.embed_tokens.weight"

# Load data
w0 = np.load(f"{WEIGHT_S1}/{name}.npy").astype(np.float32)
g1 = np.load(f"{GRAD_S1}/{name}_grad.npy").astype(np.float32)
g2 = np.load(f"{GRAD_S2}/{name}_grad.npy").astype(np.float32)

print(f"Shape: {w0.shape}, total elements: {w0.size}")

# --- PyTorch 2-step ---
pt_param = torch.tensor(w0, dtype=torch.float32, device="cuda", requires_grad=True)
pt_opt = torch.optim.AdamW([pt_param], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY, fused=True)

pt_param.grad = torch.tensor(g1, device="cuda")
pt_opt.step()
pt_opt.zero_grad()

pt_param.grad = torch.tensor(g2, device="cuda")
pt_opt.step()

pt_result = pt_param.detach().cpu().numpy().astype(np.float32)

# --- Paddle 2-step ---
pd_param = paddle.to_tensor(w0, dtype="float32", place=paddle.CUDAPlace(0))
pd_param.stop_gradient = False
pd_opt = paddle.optimizer.AdamW(
    learning_rate=LR, beta1=BETA1, beta2=BETA2, epsilon=EPS, weight_decay=WEIGHT_DECAY,
    parameters=[pd_param],
)

pd_param.grad = paddle.to_tensor(g1, dtype="float32", place=paddle.CUDAPlace(0))
pd_opt.step()
pd_opt.clear_grad()

pd_param.grad = paddle.to_tensor(g2, dtype="float32", place=paddle.CUDAPlace(0))
pd_opt.step()

pd_result = pd_param.numpy().astype(np.float32)

# Find diffs
pt_bits = pt_result.view(np.uint32).flatten()
pd_bits = pd_result.view(np.uint32).flatten()
diff_mask = pt_bits != pd_bits
diff_indices = np.where(diff_mask)[0]

print(f"\nTotal bit diffs: {len(diff_indices)}")
print(f"\nDiffering elements:")
print(f"{'Index':<12} {'PT_hex':<12} {'PD_hex':<12} {'PT_float':<20} {'PD_float':<20} {'abs_diff':<12} {'ULP_diff'}")
for idx in diff_indices:
    pt_v = pt_result.flatten()[idx]
    pd_v = pd_result.flatten()[idx]
    pt_h = pt_bits[idx]
    pd_h = pd_bits[idx]
    ulp = abs(int(pt_h) - int(pd_h))
    print(f"{idx:<12} {pt_h:#010x}  {pd_h:#010x}  {pt_v:<20.15e} {pd_v:<20.15e} {abs(pt_v-pd_v):<12.3e} {ulp}")

# Also dump the initial weight, grad1, grad2 for those indices
print(f"\n--- Initial values for differing elements ---")
print(f"{'Index':<12} {'w0':<20} {'g1':<20} {'g2':<20}")
w0_flat = w0.flatten()
g1_flat = g1.flatten()
g2_flat = g2.flatten()
for idx in diff_indices:
    print(f"{idx:<12} {w0_flat[idx]:<20.15e} {g1_flat[idx]:<20.15e} {g2_flat[idx]:<20.15e}")

# Now do the computation manually in numpy float64 to check
print(f"\n--- Manual double-precision computation ---")
for idx in diff_indices[:3]:  # just first 3
    w = np.float64(w0_flat[idx])
    grad1 = np.float64(g1_flat[idx])
    grad2 = np.float64(g2_flat[idx])

    # Step 1
    m1 = (1 - BETA1) * grad1
    v1 = (1 - BETA2) * grad1 * grad1
    bc1_1 = 1 - BETA1**1
    bc2_sqrt_1 = np.sqrt(1 - BETA2**1)
    step_size_1 = LR / bc1_1
    denom_1 = np.sqrt(v1) / bc2_sqrt_1 + EPS

    # Weight decay step 1
    w_wd1 = w - LR * WEIGHT_DECAY * w
    w_after_1 = w_wd1 - step_size_1 * m1 / denom_1

    # Step 2
    m2 = BETA1 * m1 + (1 - BETA1) * grad2
    v2 = BETA2 * v1 + (1 - BETA2) * grad2 * grad2
    bc1_2 = 1 - BETA1**2
    bc2_sqrt_2 = np.sqrt(1 - BETA2**2)
    step_size_2 = LR / bc1_2
    denom_2 = np.sqrt(v2) / bc2_sqrt_2 + EPS

    # Weight decay step 2
    w_wd2 = w_after_1 - LR * WEIGHT_DECAY * w_after_1
    w_after_2 = w_wd2 - step_size_2 * m2 / denom_2

    print(f"\nIndex {idx}:")
    print(f"  w0={w:.20e}")
    print(f"  After step1 (f64): {w_after_1:.20e}")
    print(f"  After step2 (f64): {w_after_2:.20e}")
    print(f"  PyTorch:           {np.float64(pt_result.flatten()[idx]):.20e}")
    print(f"  Paddle:            {np.float64(pd_result.flatten()[idx]):.20e}")
    print(f"  F64 as f32:        {np.float32(w_after_2):#010x}" if False else "")
