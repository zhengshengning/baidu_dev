"""
Test: isolate EXACTLY which step causes the 10 bit diffs.
Strategy: Compare after step1, then compare moments after step2 (before param update).
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

w0 = np.load(f"{WEIGHT_S1}/{name}.npy").astype(np.float32)
g1 = np.load(f"{GRAD_S1}/{name}_grad.npy").astype(np.float32)
g2 = np.load(f"{GRAD_S2}/{name}_grad.npy").astype(np.float32)

print(f"Shape: {w0.shape}, elements: {w0.size}")

# === PyTorch: step by step with state access ===
pt_param = torch.tensor(w0, dtype=torch.float32, device="cuda", requires_grad=True)
pt_opt = torch.optim.AdamW([pt_param], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY, fused=True)

# Step 1
pt_param.grad = torch.tensor(g1, device="cuda")
pt_opt.step()
pt_opt.zero_grad()

pt_after_s1 = pt_param.detach().cpu().numpy().astype(np.float32)
pt_m1_s1 = pt_opt.state[pt_param]["exp_avg"].cpu().numpy().astype(np.float32)
pt_m2_s1 = pt_opt.state[pt_param]["exp_avg_sq"].cpu().numpy().astype(np.float32)

# Step 2
pt_param.grad = torch.tensor(g2, device="cuda")
pt_opt.step()

pt_after_s2 = pt_param.detach().cpu().numpy().astype(np.float32)
pt_m1_s2 = pt_opt.state[pt_param]["exp_avg"].cpu().numpy().astype(np.float32)
pt_m2_s2 = pt_opt.state[pt_param]["exp_avg_sq"].cpu().numpy().astype(np.float32)

# === Paddle: step by step ===
pd_param = paddle.to_tensor(w0, dtype="float32", place=paddle.CUDAPlace(0))
pd_param.stop_gradient = False
pd_opt = paddle.optimizer.AdamW(
    learning_rate=LR, beta1=BETA1, beta2=BETA2, epsilon=EPS, weight_decay=WEIGHT_DECAY,
    parameters=[pd_param],
)

# Step 1
pd_param.grad = paddle.to_tensor(g1, dtype="float32", place=paddle.CUDAPlace(0))
pd_opt.step()
pd_opt.clear_grad()

pd_after_s1 = pd_param.numpy().astype(np.float32)
# Get moments from optimizer state
state = pd_opt._accumulators
# find moment names
moment1_key = None
moment2_key = None
for k, v in state.items():
    if 'moment1' in k:
        moment1_key = k
    if 'moment2' in k:
        moment2_key = k

print(f"Paddle accumulator keys: {list(state.keys())}")

pd_m1_s1 = None
pd_m2_s1 = None
for k, v in state.items():
    if 'moment1' in k.lower():
        for pk, pv in v.items():
            pd_m1_s1 = pv.numpy().astype(np.float32)
    if 'moment2' in k.lower() and 'max' not in k.lower():
        for pk, pv in v.items():
            pd_m2_s1 = pv.numpy().astype(np.float32)

# Step 2
pd_param.grad = paddle.to_tensor(g2, dtype="float32", place=paddle.CUDAPlace(0))
pd_opt.step()

pd_after_s2 = pd_param.numpy().astype(np.float32)
for k, v in state.items():
    if 'moment1' in k.lower():
        for pk, pv in v.items():
            pd_m1_s2 = pv.numpy().astype(np.float32)
    if 'moment2' in k.lower() and 'max' not in k.lower():
        for pk, pv in v.items():
            pd_m2_s2 = pv.numpy().astype(np.float32)

def bit_diff(a, b):
    return int((a.view(np.uint32) != b.view(np.uint32)).sum())

# === Compare step by step ===
print(f"\n=== After Step 1 ===")
print(f"  param    bit_diff: {bit_diff(pt_after_s1, pd_after_s1)}")
print(f"  moment1  bit_diff: {bit_diff(pt_m1_s1, pd_m1_s1)}")
print(f"  moment2  bit_diff: {bit_diff(pt_m2_s1, pd_m2_s1)}")

print(f"\n=== After Step 2 ===")
print(f"  param    bit_diff: {bit_diff(pt_after_s2, pd_after_s2)}")
print(f"  moment1  bit_diff: {bit_diff(pt_m1_s2, pd_m1_s2)}")
print(f"  moment2  bit_diff: {bit_diff(pt_m2_s2, pd_m2_s2)}")

# Check if the 10 diff elements have different moments
pt_bits = pt_after_s2.view(np.uint32).flatten()
pd_bits = pd_after_s2.view(np.uint32).flatten()
diff_indices = np.where(pt_bits != pd_bits)[0]

print(f"\n=== Detailed analysis of {len(diff_indices)} differing elements ===")
for idx in diff_indices[:5]:
    print(f"\nIndex {idx}:")
    print(f"  param_s1:  PT={pt_after_s1.flatten()[idx]:.15e}  PD={pd_after_s1.flatten()[idx]:.15e}  same={pt_after_s1.flatten()[idx]==pd_after_s1.flatten()[idx]}")
    print(f"  moment1_s1: PT={pt_m1_s1.flatten()[idx]:.15e}  PD={pd_m1_s1.flatten()[idx]:.15e}  same={pt_m1_s1.flatten()[idx]==pd_m1_s1.flatten()[idx]}")
    print(f"  moment2_s1: PT={pt_m2_s1.flatten()[idx]:.15e}  PD={pd_m2_s1.flatten()[idx]:.15e}  same={pt_m2_s1.flatten()[idx]==pd_m2_s1.flatten()[idx]}")
    print(f"  moment1_s2: PT={pt_m1_s2.flatten()[idx]:.15e}  PD={pd_m1_s2.flatten()[idx]:.15e}  same={pt_m1_s2.flatten()[idx]==pd_m1_s2.flatten()[idx]}")
    print(f"  moment2_s2: PT={pt_m2_s2.flatten()[idx]:.15e}  PD={pd_m2_s2.flatten()[idx]:.15e}  same={pt_m2_s2.flatten()[idx]==pd_m2_s2.flatten()[idx]}")
    print(f"  param_s2:  PT={pt_after_s2.flatten()[idx]:.15e}  PD={pd_after_s2.flatten()[idx]:.15e}")

    # Compute what weight decay does
    p1_pt = pt_after_s1.flatten()[idx]
    D = np.float64(LR) * np.float64(WEIGHT_DECAY)
    # In float32
    p1_f32 = np.float32(p1_pt)
    p_wd = np.float32(np.float64(p1_f32) - D * np.float64(p1_f32))
    print(f"  weight_decay(p1): {p_wd:.15e}  (D={D:.20e})")
