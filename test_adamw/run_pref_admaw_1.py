import csv
import os
import random
import time

import numpy as np
import paddle
import torch
from torch.optim.adamw import adamw

paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': 1})

# os.environ["PADDLE_ADAMW_TORCH_COMPAT"] = "1"

# 设置了 Paddle 和 PyTorch 打印张量时的显示格式
precision = 15
paddle.set_printoptions(precision=precision)
torch.set_printoptions(
    precision=precision,
    sci_mode=False,
    linewidth=200,
    threshold=10_000,
)

results = []


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42


def paddle_device_to_torch_device(device: str) -> str:
    """Convert Paddle device name to PyTorch device name."""
    if device == "gpu":
        return "cuda"
    if device.startswith("gpu:"):
        return f"cuda:{device.split(':', 1)[1]}"
    return device


def calc_diff(np_pd, np_torch):
    diff = np_pd - np_torch
    return np.max(np.abs(diff)), np.mean(np.abs(diff))


def np_to_torch(np_data, dtype=torch.float, device="cuda") -> torch.Tensor:
    return torch.from_numpy(np_data).to(dtype).to(paddle_device_to_torch_device(device))


def torch_to_np(torch_data: torch.Tensor):
    return torch_data.float().detach().cpu().numpy()


def np_to_paddle(np_data, dtype="float32", device="gpu") -> paddle.Tensor:
    return paddle.to_tensor(np_data, dtype=dtype).to(device)


def paddle_to_np(paddle_data: paddle.Tensor):
    return paddle_data.astype("float32").detach().cpu().numpy()


def paddle_adamw_step(
    param_np,
    grad_np,
    moment1_np,
    moment2_np,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    beta1_pow_val=None,
    beta2_pow_val=None,
    device="gpu",
    amsgrad=False,
    warmup_iters=100,
    test_iters=1000,
):
    """
    Use paddle._C_ops.adamw_ to perform AdamW update steps.
    Returns updated param, moment1, moment2 as numpy arrays and timing in ms.
    """
    param = np_to_paddle(param_np, "float32", device)
    grad = np_to_paddle(grad_np, "float32", device)
    moment1 = np_to_paddle(moment1_np, "float32", device)
    moment2 = np_to_paddle(moment2_np, "float32", device)
    moment2_max = paddle.zeros_like(param)

    lr = np_to_paddle(np.array([learning_rate]).astype("float64"), "float64", device)

    if beta1_pow_val is None:
        beta1_pow_val = beta1
    if beta2_pow_val is None:
        beta2_pow_val = beta2
    beta1_pow = np_to_paddle(np.array([beta1_pow_val]).astype("float32"), "float32", device)
    beta2_pow = np_to_paddle(np.array([beta2_pow_val]).astype("float32"), "float32", device)

    master_weight = None
    find_inf = None
    lazy_mode = False
    lr_ratio = 1.0
    with_decay = True

    # Warmup
    for _ in range(warmup_iters):
        paddle._C_ops.adamw_(
            param,
            grad,
            lr,
            moment1,
            moment2,
            moment2_max,
            beta1_pow,
            beta2_pow,
            master_weight,
            find_inf,
            beta1,
            beta2,
            epsilon,
            lr_ratio,
            weight_decay,
            with_decay,
            lazy_mode,
            1000,  # min_row_size_to_use_multithread
            False,  # multi_precision
            False,  # use_global_beta_pow
            amsgrad,
        )
    paddle.device.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(test_iters):
        paddle._C_ops.adamw_(
            param,
            grad,
            lr,
            moment1,
            moment2,
            moment2_max,
            beta1_pow,
            beta2_pow,
            master_weight,
            find_inf,
            beta1,
            beta2,
            epsilon,
            lr_ratio,
            weight_decay,
            with_decay,
            lazy_mode,
            1000,  # min_row_size_to_use_multithread
            False,  # multi_precision
            False,  # use_global_beta_pow
            amsgrad,
        )
    paddle.device.synchronize()
    end_time = time.time()

    paddle_total_time = (end_time - start_time) * 1000.0  # ms
    paddle_avg_time = paddle_total_time / test_iters

    return (
        paddle_to_np(param),
        paddle_to_np(moment1),
        paddle_to_np(moment2),
        paddle_total_time,
        paddle_avg_time,
    )


def torch_adamw_step(
    param_np,
    grad_np,
    moment1_np,
    moment2_np,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    step=1,
    device="cuda",
    warmup_iters=100,
    test_iters=1000,
):
    """
    Use adamw() functional API to perform AdamW update steps.
    Returns updated param, moment1, moment2 as numpy arrays and timing in ms.
    """
    param = np_to_torch(param_np, torch.float, device=device)
    grad = np_to_torch(grad_np, torch.float, device=device)
    exp_avg = np_to_torch(moment1_np, torch.float, device=device)
    exp_avg_sq = np_to_torch(moment2_np, torch.float, device=device)
    state_step = torch.tensor(float(step), device=paddle_device_to_torch_device(device))

    # Warmup
    for _ in range(warmup_iters):
        adamw(
            params=[param],
            grads=[grad],
            exp_avgs=[exp_avg],
            exp_avg_sqs=[exp_avg_sq],
            max_exp_avg_sqs=[],
            state_steps=[state_step],
            fused=True,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=epsilon,
            maximize=False,
        )
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(test_iters):
        adamw(
            params=[param],
            grads=[grad],
            exp_avgs=[exp_avg],
            exp_avg_sqs=[exp_avg_sq],
            max_exp_avg_sqs=[],
            state_steps=[state_step],
            fused=True,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=epsilon,
            maximize=False,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    torch_total_time = (end_time - start_time) * 1000.0  # ms
    torch_avg_time = torch_total_time / test_iters

    param_out = torch_to_np(param)
    moment1_out = torch_to_np(exp_avg)
    moment2_out = torch_to_np(exp_avg_sq)

    return param_out, moment1_out, moment2_out, torch_total_time, torch_avg_time


def run_benchmark(
    shape,
    learning_rate=1e-5,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    weight_decay=0.01,
    step=1,
    device="gpu",
    warmup_iters=100,
    test_iters=1000,
    data_seed=SEED,
):
    beta1_pow_val = beta1**step
    beta2_pow_val = beta2**step

    # rng = np.random.RandomState(data_seed)
    # param_np = rng.uniform(-1, 1, shape).astype("float32")
    # grad_np = rng.uniform(-1, 1, shape).astype("float32")
    param_np = np.random.uniform(-1, 1, shape).astype("float32")
    grad_np = np.random.uniform(-1, 1, shape).astype("float32")
    moment1_np = np.zeros(shape).astype("float32")
    moment2_np = np.zeros(shape).astype("float32")

    print(f"\n{'=' * 80}")
    print(f"Shape: {shape}, Numel: {np.prod(shape)}, Step: {step}, Data seed: {data_seed}")
    print(f"Warmup iters: {warmup_iters}, Test iters: {test_iters}")
    print(f"{'=' * 80}")

    pd_param, pd_m1, pd_m2, _, _ = paddle_adamw_step(
        param_np,
        grad_np,
        moment1_np,
        moment2_np,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        beta1_pow_val=beta1_pow_val,
        beta2_pow_val=beta2_pow_val,
        device=device,
        warmup_iters=0,
        test_iters=1,
    )

    # NOTE: torch's _fused_adam does torch._foreach_add_(state_steps, 1) before
    # the kernel call, so we pass step-1 so the kernel sees the same step as Paddle.
    th_param, th_m1, th_m2, _, _ = torch_adamw_step(
        param_np,
        grad_np,
        moment1_np,
        moment2_np,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        step=step - 1,
        device=device,
        warmup_iters=0,
        test_iters=1,
    )

    _, _, _, paddle_total_time, paddle_avg_time = paddle_adamw_step(
        param_np,
        grad_np,
        moment1_np,
        moment2_np,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        beta1_pow_val=beta1_pow_val,
        beta2_pow_val=beta2_pow_val,
        device=device,
        warmup_iters=warmup_iters,
        test_iters=test_iters,
    )

    _, _, _, torch_total_time, torch_avg_time = torch_adamw_step(
        param_np,
        grad_np,
        moment1_np,
        moment2_np,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        step=step - 1,
        device=device,
        warmup_iters=warmup_iters,
        test_iters=test_iters,
    )

    param_max_diff, param_mean_diff = calc_diff(pd_param, th_param)
    m1_max_diff, m1_mean_diff = calc_diff(pd_m1, th_m1)
    m2_max_diff, m2_mean_diff = calc_diff(pd_m2, th_m2)
    speedup = torch_avg_time / paddle_avg_time if paddle_avg_time > 0 else 0

    print("[Paddle adamw_]")
    print(f"  Total time ({test_iters} runs): {paddle_total_time:.3f} ms")
    print(f"  Average time per run: {paddle_avg_time:.6f} ms")
    print("[Torch adamw fused]")
    print(f"  Total time ({test_iters} runs): {torch_total_time:.3f} ms")
    print(f"  Average time per run: {torch_avg_time:.6f} ms")
    print("\n[Performance Comparison]")
    print(f"  Speedup (Torch/Paddle): {speedup:.2f}x")
    if speedup > 1:
        print(f"  -> Paddle is {speedup:.2f}x faster than Torch")
    elif speedup > 0:
        print(f"  -> Torch is {1 / speedup:.2f}x faster than Paddle")
    print("\n[Correctness Check: single AdamW step]")
    print(f"  Param max/mean abs diff: {param_max_diff:.8f} / {param_mean_diff:.8f}")
    print(f"  Moment1 max/mean abs diff: {m1_max_diff:.8f} / {m1_mean_diff:.8f}")
    print(f"  Moment2 max/mean abs diff: {m2_max_diff:.8f} / {m2_mean_diff:.8f}")

    results.append(
        {
            "Shape": str(shape),
            "Numel": int(np.prod(shape)),
            "Data_Seed": data_seed,
            "Paddle_Total(ms)": f"{paddle_total_time:.3f}",
            "Paddle_Avg(ms)": f"{paddle_avg_time:.6f}",
            "Torch_Total(ms)": f"{torch_total_time:.3f}",
            "Torch_Avg(ms)": f"{torch_avg_time:.6f}",
            "Speedup": f"{speedup:.2f}",
            "Param_Max_Diff": f"{param_max_diff:.8f}",
            "Param_Mean_Diff": f"{param_mean_diff:.8f}",
            "Moment1_Max_Diff": f"{m1_max_diff:.8f}",
            "Moment1_Mean_Diff": f"{m1_mean_diff:.8f}",
            "Moment2_Max_Diff": f"{m2_max_diff:.8f}",
            "Moment2_Mean_Diff": f"{m2_mean_diff:.8f}",
        }
    )


def save_results_to_csv(filename="adamw_benchmark_results.csv"):
    """保存结果到 CSV 文件"""
    if not results:
        return

    fieldnames = [
        "Shape",
        "Numel",
        "Data_Seed",
        "Paddle_Total(ms)",
        "Paddle_Avg(ms)",
        "Torch_Total(ms)",
        "Torch_Avg(ms)",
        "Speedup",
        "Param_Max_Diff",
        "Param_Mean_Diff",
        "Moment1_Max_Diff",
        "Moment1_Mean_Diff",
        "Moment2_Max_Diff",
        "Moment2_Mean_Diff",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存至: {filename}")


def print_summary():
    if not results:
        print("\n没有成功的 benchmark 结果")
        return

    print("\n" + "=" * 140)
    print("AdamW 性能和正确性测试汇总")
    print("=" * 140)
    print(
        "%-18s %-12s %-14s %-14s %-10s %-12s %-12s %-12s"
        % (
            "Shape",
            "Numel",
            "Paddle(ms)",
            "Torch(ms)",
            "Speedup",
            "Param_Max",
            "M1_Max",
            "M2_Max",
        )
    )
    print("-" * 140)
    for result in results:
        print(
            f"{result['Shape']:<18} {result['Numel']:<12} "
            f"{result['Paddle_Avg(ms)']:<14} {result['Torch_Avg(ms)']:<14} "
            f"{result['Speedup']:<10} {result['Param_Max_Diff']:<12} "
            f"{result['Moment1_Max_Diff']:<12} {result['Moment2_Max_Diff']:<12}"
        )


if __name__ == "__main__":
    # 检查CUDA是否可用
    if paddle.device.is_compiled_with_cuda() and torch.cuda.is_available():
        gpu_id = int(os.environ.get("BENCHMARK_GPU_ID", "6"))
        device = f"gpu:{gpu_id}"
        torch_device = paddle_device_to_torch_device(device)
        print(f"CUDA可用，使用 Paddle {device} / Torch {torch_device} 进行测试")
        paddle.set_device(device)
        torch.cuda.set_device(gpu_id)
        set_deterministic(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    else:
        print("警告: CUDA不可用或PyTorch未检测到CUDA")
        if not paddle.device.is_compiled_with_cuda():
            print("  - Paddle CUDA不可用")
        if not torch.cuda.is_available():
            print("  - PyTorch CUDA不可用")
        exit(1)

    test_shapes = [
        (102, 105),
        (1024, 1024),
        (4096, 768),
        (6996, 1152),
        (8856, 1152),
        (9052, 1152),
    ]

    for shape in test_shapes:
        try:
            run_benchmark(shape, device=device, warmup_iters=50, test_iters=500)
        except Exception as e:
            print(f"测试失败: shape={shape}")
            print(f"错误信息: {e}")

    save_results_to_csv()
    print_summary()
