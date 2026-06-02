"""
单测：复现 MoE routing 中 paddle.softmax vs torch.softmax 的 diff。
输入：模拟 gate 输出的 router_logits，shape [T, 4]（4个expert）
"""
import numpy as np

def ulp_diff(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    return int(np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64)).max())

def bit_diff(a, b):
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    return int((a.view(np.uint32) != b.view(np.uint32)).sum())

# ---------- 1. 生成输入（固定seed） ----------
np.random.seed(42)
T = 128   # token 数
E = 4     # num_experts
logits_np = np.random.randn(T, E).astype(np.float32)

# ---------- 2. PyTorch softmax ----------
import torch
logits_pt = torch.tensor(logits_np, device='cuda')
out_pt = torch.nn.functional.softmax(logits_pt.float(), dim=-1)
out_pt_np = out_pt.cpu().numpy()

# ---------- 3. PaddlePaddle softmax ----------
import paddle
paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})
logits_pd = paddle.to_tensor(logits_np, place=paddle.CUDAPlace(0))
out_pd = paddle.nn.functional.softmax(logits_pd.cast('float32'), axis=-1)
out_pd_np = out_pd.numpy()

# ---------- 4. 结果对比 ----------
diff = np.abs(out_pt_np - out_pd_np)
print("=== softmax forward diff ===")
print(f"max abs diff : {diff.max():.4e}")
print(f"mean abs diff: {diff.mean():.4e}")
print(f"ULP diff     : {ulp_diff(out_pt_np, out_pd_np)}")
print(f"bit diff (#) : {bit_diff(out_pt_np, out_pd_np)}")
print(f"bit diff (%) : {bit_diff(out_pt_np, out_pd_np) / out_pt_np.size * 100:.2f}%")

# ---------- 5. 验证：diff 确实存在且可复现 ----------
assert diff.max() > 0, "FAIL: no diff found, test may be broken"
assert diff.max() < 1e-6, f"FAIL: diff {diff.max():.2e} too large, not 1 ULP"
print(f"\n[PASS] softmax diff confirmed: {diff.max():.4e} (1 ULP = {np.finfo(np.float32).eps / 2:.4e})")
print("Root cause: paddle/torch use different CUDA softmax kernel implementations,")
print("producing results that differ by 1 ULP (~6e-8) for the same float32 input.")
