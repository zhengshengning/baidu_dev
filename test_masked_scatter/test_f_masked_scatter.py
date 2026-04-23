# masked_scatter_benchmark.py
# Paddle vs PyTorch masked_scatter 性能与结果对比

import torch
import paddle
import numpy as np
import time
import csv

results = []

def run_benchmark(B, N, num_warmup=100, num_runs=1000, mask_ratio=0.3):
    """
    对比 PyTorch 和 Paddle 的 masked_scatter 性能和结果
    Args:
        B: batch size
        N: sequence length
        num_warmup: warmup 次数
        num_runs: 正式测试次数
        mask_ratio: mask 中 True 的比例
    """
    print(f"\n{'='*60}")
    print(f"Running test: B={B}, N={N}, Shape=[{B}, {N}], mask_ratio={mask_ratio}")
    print(f"{'='*60}")

    # 使用相同的随机种子确保数据一致
    np.random.seed(42)
    base_np = np.zeros((B, N), dtype=np.float32)
    source_np = np.random.randn(B, N).astype(np.float32)
    mask_np = (np.random.rand(B, N) > (1 - mask_ratio))

    # ==================== PyTorch 测试 ====================
    base_torch = torch.from_numpy(base_np).cuda()
    source_torch = torch.from_numpy(source_np).cuda()
    mask_torch = torch.from_numpy(mask_np).cuda()

    # Warmup
    for _ in range(num_warmup):
        _ = base_torch.masked_scatter(mask_torch, source_torch)
    torch.cuda.synchronize()

    # 性能测试
    torch.cuda.synchronize()
    start_time = time.time()
    # paddle.base.core.nvprof_start() #######################################
    for _ in range(num_runs):
        out_torch = base_torch.masked_scatter(mask_torch, source_torch)
    # paddle.base.core.nvprof_stop() #######################################
    torch.cuda.synchronize()
    end_time = time.time()

    torch_total_time = (end_time - start_time) * 1000.0
    torch_avg_time = torch_total_time / num_runs

    print(f"\n[PyTorch] Total time ({num_runs} runs): {torch_total_time:.3f} ms")
    print(f"[PyTorch] Average time per run: {torch_avg_time:.4f} ms")

    # ==================== Paddle 测试 ====================
    base_paddle = paddle.to_tensor(base_np)
    source_paddle = paddle.to_tensor(source_np)
    mask_paddle = paddle.to_tensor(mask_np)

    # Warmup
    for _ in range(num_warmup):
        _ = base_paddle.masked_scatter(mask_paddle, source_paddle)
    paddle.device.synchronize()

    # 性能测试
    paddle.device.synchronize()
    start_time = time.time()
    # paddle.base.core.nvprof_start() #######################################
    for _ in range(num_runs):
        out_paddle = base_paddle.masked_scatter(mask_paddle, source_paddle)
    # paddle.base.core.nvprof_stop() #######################################
    paddle.device.synchronize()
    end_time = time.time()

    paddle_total_time = (end_time - start_time) * 1000.0
    paddle_avg_time = paddle_total_time / num_runs

    print(f"\n[Paddle] Total time ({num_runs} runs): {paddle_total_time:.3f} ms")
    print(f"[Paddle] Average time per run: {paddle_avg_time:.4f} ms")

    # ==================== 结果对比 ====================
    out_torch_np = out_torch.cpu().numpy()
    out_paddle_np = out_paddle.numpy()

    # 计算结果差异
    max_diff = np.abs(out_torch_np - out_paddle_np).max()
    mean_diff = np.abs(out_torch_np - out_paddle_np).mean()
    is_close = np.allclose(out_torch_np, out_paddle_np, rtol=0, atol=0)

    print(f"\n[结果对比]")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Results match: {is_close}")

    # 性能对比
    speedup = torch_avg_time / paddle_avg_time if paddle_avg_time > 0 else 0
    print(f"\n[性能对比]")
    print(f"  Speedup (Torch/Paddle): {speedup:.3f}x")
    if speedup > 1:
        print(f"  Paddle 更快 {((speedup - 1) * 100):.1f}%")
    else:
        print(f"  PyTorch 更快 {((1/speedup - 1) * 100):.1f}%")

    # 记录结果
    results.append({
        'B': B,
        'N': N,
        'Total Elements': B * N,
        'Mask Ratio': mask_ratio,
        'Torch Time(ms)': f"{torch_avg_time:.4f}",
        'Paddle Time(ms)': f"{paddle_avg_time:.4f}",
        'Speedup(Torch/Paddle)': f"{speedup:.3f}",
        'Max Diff': f"{max_diff:.6e}",
        'Results Match': is_close
    })


if __name__ == "__main__":
    # 检查CUDA是否可用
    if paddle.device.is_compiled_with_cuda() and torch.cuda.is_available():
        print("CUDA可用，使用GPU进行测试")
        paddle.set_device('gpu:2')
        torch.cuda.set_device(2)
    else:
        print("警告: CUDA不可用或PyTorch未检测到CUDA")
        if not paddle.device.is_compiled_with_cuda():
            print("  - Paddle CUDA不可用")
        if not torch.cuda.is_available():
            print("  - PyTorch CUDA不可用")
        exit(1)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Paddle version: {paddle.__version__}")
    print(f"PyTorch CUDA: {torch.cuda.get_device_name(0)}")

    # 测试不同的 shape 组合
    # B (batch size) 和 N (sequence length) 的组合
    test_shapes = [
        # 小规模测试
        (128, 512),
        (256, 512),
        (512, 512),
        (512, 1024),
        (1024, 1024),
        # 中等规模测试
        (1024, 2048),
        (1024, 4096),
        (2048, 2048),
        (2048, 4096),
        (4096, 4096),
        # 大规模测试
        (4096, 8192),
        (8192, 4096),
        (8192, 8192),
        (16384, 4096),
        (4096, 16384),
        # 超大规模测试 (可选，根据显存情况)
        (16384, 8192),
        (8192, 16384),
        (16384, 16384),
    ]

    # 运行所有测试
    for B, N in test_shapes:
        try:
            run_benchmark(B, N, num_warmup=100, num_runs=1000, mask_ratio=0.3)
        except RuntimeError as e:
            print(f"\n跳过 Shape [{B}, {N}]: {e}")
            results.append({
                'B': B,
                'N': N,
                'Total Elements': B * N,
                'Mask Ratio': 0.3,
                'Torch Time(ms)': 'OOM',
                'Paddle Time(ms)': 'OOM',
                'Speedup(Torch/Paddle)': 'N/A',
                'Max Diff': 'N/A',
                'Results Match': 'N/A'
            })

    # 保存结果到CSV
    if results:
        csv_file = 'masked_scatter_f_benchmark_results.csv'
        fieldnames = [
            'B', 'N', 'Total Elements', 'Mask Ratio',
            'Torch Time(ms)', 'Paddle Time(ms)', 'Speedup(Torch/Paddle)',
            'Max Diff', 'Results Match'
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n{'='*60}")
        print(f"结果已保存至: {csv_file}")
        print(f"{'='*60}")

        # 打印汇总表格
        print("\n\n========== 结果汇总 ==========")
        print(f"{'B':>8} {'N':>8} {'Torch(ms)':>12} {'Paddle(ms)':>12} {'Speedup':>10} {'Match':>8}")
        print("-" * 62)
        for r in results:
            print(f"{r['B']:>8} {r['N']:>8} {r['Torch Time(ms)']:>12} {r['Paddle Time(ms)']:>12} {r['Speedup(Torch/Paddle)']:>10} {str(r['Results Match']):>8}")
