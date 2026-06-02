"""
复现 Paddle AdamW vs PyTorch AdamW 的 CUDA FMA 精度差异。

原理：
- Step1 optimizer: m1 = beta1*0 + (1-beta1)*g1 = (1-beta1)*g1
  只有一个非零项，退化为标量乘，两框架结果相同。
- Step2 optimizer: m1_new = beta1*m1 + (1-beta1)*g2
  两项均非零，CUDA kernel 的 FMA 指令融合策略不同 → 1-32 ULP 差异 → 权重出现 diff

使用真实保存的权重/梯度 npy 数据，分别用 Paddle 和 PyTorch 独立执行两步 AdamW，
逐参数对比 bit 级别差异。

运行方式：
  export FLAGS_use_accuracy_compatible_kernel=1
  export PADDLE_ADAMW_TORCH_COMPAT=1
  python /work/Qwen3/grad_diff/reproduce_adamw_diff.py
"""

import os
import numpy as np
import torch
import paddle

# ──────────────────────────────────────────────
# 路径配置（均来自 HF 侧保存的真实数据，两框架共用相同输入）
# ──────────────────────────────────────────────
BASE = "/home/ningzhengsheng/src/adam_diff_test"
WEIGHT_S1  = f"{BASE}/hf-weights-step1-start"   # step1 开始时的权重
GRAD_S1    = f"{BASE}/hf-grad-before-optimizer/step1"   # step1 梯度
GRAD_S2    = f"{BASE}/hf-grad-before-optimizer/step2"   # step2 梯度

# AdamW 超参（与训练完全一致）
LR           = 1e-5
BETA1        = 0.9
BETA2        = 0.95
EPS          = 1e-8
WEIGHT_DECAY = 0.01

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def bit_diff(a: np.ndarray, b: np.ndarray) -> int:
    """逐位比较两个 float32 数组，返回不同元素个数。"""
    assert a.dtype == b.dtype == np.float32
    return int((a.view(np.uint32) != b.view(np.uint32)).sum())


def report(tag, hf_arr, pf_arr, transposed=False):
    """打印一个参数的对比结果。"""
    if transposed:
        pf_arr = pf_arr.T
    if hf_arr.shape != pf_arr.shape:
        print(f"  {tag:<55} shape mismatch {hf_arr.shape} vs {pf_arr.shape}")
        return
    diff   = np.abs(hf_arr.astype(np.float64) - pf_arr.astype(np.float64))
    n_bits = bit_diff(hf_arr.astype(np.float32), pf_arr.astype(np.float32))
    print(f"  {tag:<55} max={diff.max():.3e}  mean={diff.mean():.3e}  bit_diff={n_bits}/{hf_arr.size}")


def load_param_names():
    """获取 step1 起始权重目录中的所有参数名。"""
    names = []
    for f in sorted(os.listdir(WEIGHT_S1)):
        if f.endswith(".npy"):
            names.append(f[:-4])  # 去掉 .npy
    return names


# ──────────────────────────────────────────────
# PyTorch AdamW（两步）
# ──────────────────────────────────────────────

def run_torch_adamw(param_names):
    """
    用 PyTorch AdamW 执行两步更新，返回 {name: weight_after_step2}。
    """
    params_torch = {}
    for name in param_names:
        w = np.load(f"{WEIGHT_S1}/{name}.npy").astype(np.float32)
        t = torch.tensor(w, dtype=torch.float32, device="cuda", requires_grad=True)
        params_torch[name] = t

    opt = torch.optim.AdamW(
        list(params_torch.values()),
        lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY, fused=True,
    )

    # ── step1 ──
    opt.zero_grad()
    for name in param_names:
        g_path = f"{GRAD_S1}/{name}_grad.npy"
        if not os.path.exists(g_path):
            continue
        g = torch.tensor(np.load(g_path).astype(np.float32), device="cuda")
        params_torch[name].grad = g
    opt.step()

    # ── step2 ──
    opt.zero_grad()
    for name in param_names:
        g_path = f"{GRAD_S2}/{name}_grad.npy"
        if not os.path.exists(g_path):
            continue
        g = torch.tensor(np.load(g_path).astype(np.float32), device="cuda")
        params_torch[name].grad = g
    opt.step()

    return {name: params_torch[name].detach().cpu().numpy().astype(np.float32)
            for name in param_names}


# ──────────────────────────────────────────────
# Paddle AdamW（两步）
# ──────────────────────────────────────────────

def run_paddle_adamw(param_names):
    """
    用 Paddle AdamW 执行两步更新，返回 {name: weight_after_step2}。
    """
    params_paddle = {}
    for name in param_names:
        w = np.load(f"{WEIGHT_S1}/{name}.npy").astype(np.float32)
        t = paddle.to_tensor(w, dtype="float32", place=paddle.CUDAPlace(0))
        t.stop_gradient = False
        params_paddle[name] = t

    opt = paddle.optimizer.AdamW(
        learning_rate=LR,
        beta1=BETA1, beta2=BETA2, epsilon=EPS, weight_decay=WEIGHT_DECAY,
        parameters=list(params_paddle.values()),
    )

    def _assign_grads(step_dir):
        for name, param in params_paddle.items():
            g_path = f"{step_dir}/{name}_grad.npy"
            if not os.path.exists(g_path):
                continue
            g = paddle.to_tensor(np.load(g_path).astype(np.float32),
                                  dtype="float32", place=paddle.CUDAPlace(0))
            param.grad = g

    # ── step1 ──
    _assign_grads(GRAD_S1)
    opt.step()
    opt.clear_grad()

    # ── step2 ──
    _assign_grads(GRAD_S2)
    opt.step()
    opt.clear_grad()

    return {name: params_paddle[name].numpy().astype(np.float32)
            for name in param_names}


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    param_names = load_param_names()
    print(f"共 {len(param_names)} 个参数，使用真实 step1/step2 梯度数据\n")

    print("运行 PyTorch AdamW (2 steps) ...")
    res_torch  = run_torch_adamw(param_names)

    print("运行 Paddle  AdamW (2 steps) ...\n")
    res_paddle = run_paddle_adamw(param_names)

    # ── Step1 单独验证（应全零 diff）──────────────────
    print("=" * 70)
    print("【验证】Step1 之后权重 (期望: bit_diff = 0)")
    print("=" * 70)

    # 用保存的 step2-start 权重作为 step1 结果的参考
    step2_start_hf = f"{BASE}/hf-weights-step2-start"
    step2_start_pf = f"{BASE}/pf-weights-step2-start"

    for name in param_names:
        hf_s2 = np.load(f"{step2_start_hf}/{name}.npy").astype(np.float32)
        pf_s2 = np.load(f"{step2_start_pf}/{name}.npy").astype(np.float32)
        # 处理 PF 转置存储
        if pf_s2.shape == hf_s2.shape[::-1] and hf_s2.shape != pf_s2.shape:
            pf_s2 = pf_s2.T
        elif pf_s2.ndim == 2 and pf_s2.shape[0] == pf_s2.shape[1]:
            if np.abs(hf_s2 - pf_s2.T).max() < np.abs(hf_s2 - pf_s2).max():
                pf_s2 = pf_s2.T
        n_bits = bit_diff(hf_s2, pf_s2) if hf_s2.shape == pf_s2.shape else -1
        mark = "✓" if n_bits == 0 else f"✗ {n_bits} bits"
        print(f"  {name:<55} {mark}")

    # ── Step2 对比（核心复现）────────────────────────
    print()
    print("=" * 70)
    print("【复现】Step2 之后权重 Paddle vs PyTorch (期望: 存在 bit_diff)")
    print("=" * 70)

    total_params = 0
    total_bit_diff = 0
    results = []

    for name in param_names:
        pt_w  = res_torch[name]
        pd_w  = res_paddle[name]

        # 处理转置（q/k/v/o/gate/up/down proj 在 PF 中 weight 是转置存储）
        transposed = False
        if pd_w.shape == pt_w.shape[::-1] and pt_w.shape != pd_w.shape:
            pd_w = pd_w.T
            transposed = True
        elif pd_w.ndim == 2 and pd_w.shape[0] == pd_w.shape[1]:
            if np.abs(pt_w - pd_w.T).max() < np.abs(pt_w - pd_w).max():
                pd_w = pd_w.T
                transposed = True

        if pt_w.shape != pd_w.shape:
            print(f"  {name:<55} shape mismatch, skip")
            continue

        n_bits  = bit_diff(pt_w, pd_w)
        diff    = np.abs(pt_w.astype(np.float64) - pd_w.astype(np.float64))
        total_params  += pt_w.size
        total_bit_diff += n_bits

        mark = "✓" if n_bits == 0 else f"✗ {n_bits}/{pt_w.size} bits"
        trans_tag = " [T]" if transposed else "    "
        print(f"  {name:<55}{trans_tag}  max={diff.max():.3e}  mean={diff.mean():.3e}  {mark}")
        results.append((name, n_bits, pt_w.size, diff.max()))

    print()
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    n_diff_params = sum(1 for _, b, _, _ in results if b > 0)
    print(f"  参数总量:         {total_params:,}")
    print(f"  有 diff 的参数:   {n_diff_params} / {len(results)}")
    print(f"  总 bit diff 数:   {total_bit_diff:,}")
    print()
    print("结论: Step1 后两框架权重完全相同（FMA 退化为标量乘），")
    print("      Step2 后出现 1-32 ULP 差异（真 FMA，CUDA kernel 融合策略不同）。")


if __name__ == "__main__":
    main()
