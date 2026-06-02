"""
Deep dive: check if weight decay produces the same p at step2 boundary.
"""
import os, struct
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
w0 = np.load(f"{WEIGHT_S1}/{name}.npy").astype(np.float32)
g1 = np.load(f"{GRAD_S1}/{name}_grad.npy").astype(np.float32)
g2 = np.load(f"{GRAD_S2}/{name}_grad.npy").astype(np.float32)

# Run both frameworks
pt_param = torch.tensor(w0, dtype=torch.float32, device="cuda", requires_grad=True)
pt_opt = torch.optim.AdamW([pt_param], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY, fused=True)
pt_param.grad = torch.tensor(g1, device="cuda")
pt_opt.step(); pt_opt.zero_grad()
pt_s1 = pt_param.detach().cpu().numpy().astype(np.float32)
pt_param.grad = torch.tensor(g2, device="cuda")
pt_opt.step()
pt_s2 = pt_param.detach().cpu().numpy().astype(np.float32)

pd_param = paddle.to_tensor(w0, dtype="float32", place=paddle.CUDAPlace(0))
pd_param.stop_gradient = False
pd_opt = paddle.optimizer.AdamW(learning_rate=LR, beta1=BETA1, beta2=BETA2, epsilon=EPS, weight_decay=WEIGHT_DECAY, parameters=[pd_param])
pd_param.grad = paddle.to_tensor(g1, dtype="float32", place=paddle.CUDAPlace(0))
pd_opt.step(); pd_opt.clear_grad()
pd_s1 = pd_param.numpy().astype(np.float32)
pd_param.grad = paddle.to_tensor(g2, dtype="float32", place=paddle.CUDAPlace(0))
pd_opt.step()
pd_s2 = pd_param.numpy().astype(np.float32)

# Find the 10 diff indices
pt_bits = pt_s2.view(np.uint32).flatten()
pd_bits = pd_s2.view(np.uint32).flatten()
diff_indices = np.where(pt_bits != pd_bits)[0]

def f2hex(f):
    return struct.pack('>f', f).hex()

print(f"Analyzing {len(diff_indices)} diffs\n")

# For each diff, manually compute what should happen at step 2
# Given: p_after_s1 (identical), m1_s1 (identical), m2_s1 (identical), g2
# Step 2 computation:
#   1. weight_decay: p_wd = p - lr * wd * p
#   2. m1 = beta1 * m1_old + (1-beta1) * g2
#   3. m2 = beta2 * m2_old + (1-beta2) * g2 * g2
#   4. bc1 = float(1 - pow(beta1, 2.0))  (computed on device)
#   5. bc2_sqrt = float(sqrt(1 - pow(beta2, 2.0)))
#   6. step_size = float(lr_double / bc1)
#   7. denom = float(float(sqrt(m2) / bc2_sqrt) + eps)
#   8. p = p_wd - step_size * m1 / denom

# Since m1, m2, bc1, bc2_sqrt, step_size, denom are ALL identical between PT/PD,
# the only possible diff is in step 1 (weight_decay).

# Let's simulate weight_decay with different FMA patterns
D = np.float64(LR) * np.float64(WEIGHT_DECAY)  # 1e-7

for idx in diff_indices:
    p = pt_s1.flatten()[idx]  # same for both
    p_d = np.float64(p)

    # Pattern 1: fma(-D, p, p) = p + (-D * p) = p - D*p
    # In exact double: p - D*p = p * (1 - D)
    fma_result = np.float64(p) - np.float64(D) * np.float64(p)  # this is the double-precision result
    p_wd_f32 = np.float32(fma_result)

    # Pattern 2: the actual float32 result using different intermediate roundings
    # The key: is the weight-decayed p the same for PT and PD?
    # We know p_s1 is the same. So if weight decay gives the same p_wd,
    # then the diff must be from the param update.

    pt_final = pt_s2.flatten()[idx]
    pd_final = pd_s2.flatten()[idx]

    print(f"idx={idx}: p_s1={p:.15e} ({f2hex(p)})")
    print(f"  D*p = {D*np.float64(p):.20e}")
    print(f"  p_wd(f32) = {p_wd_f32:.15e} ({f2hex(p_wd_f32)})")
    print(f"  PT_final={pt_final:.15e} ({f2hex(pt_final)})")
    print(f"  PD_final={pd_final:.15e} ({f2hex(pd_final)})")

    # Check: is there any f32 rounding ambiguity in weight decay?
    # p - D*p in double, then truncate to float32
    exact = np.float64(p) - D * np.float64(p)
    # Get the two candidate float32 values
    f32_val = np.float32(exact)
    f32_bits = np.array([f32_val], dtype=np.float32).view(np.uint32)[0]
    f32_up = np.array([f32_bits + 1], dtype=np.uint32).view(np.float32)[0]
    f32_down = np.array([f32_bits - 1], dtype=np.uint32).view(np.float32)[0]

    dist_self = abs(np.float64(f32_val) - exact)
    dist_up = abs(np.float64(f32_up) - exact)
    dist_down = abs(np.float64(f32_down) - exact)

    print(f"  exact={exact:.20e}")
    print(f"  f32_val={f32_val:.15e} dist={dist_self:.5e}")
    print(f"  f32_up ={f32_up:.15e}  dist={dist_up:.5e}")
    print(f"  f32_down={f32_down:.15e} dist={dist_down:.5e}")

    # Is this an exact midpoint?
    is_midpoint = (dist_self == dist_up or dist_self == dist_down)
    print(f"  MIDPOINT: {is_midpoint}")
    print()
